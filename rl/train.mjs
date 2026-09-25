#!/usr/bin/env node
// PPO trainer. Workers play rollouts on random remix holes and compute
// gradient shards; this thread sums them, clips, and steps Adam on the shared
// parameters. See docs/TRAINING.md.
//
// usage: node rl/train.mjs [--name run] [--start-prob 0.3] [--iters 300] [--workers N] [--steps 1024]
//        [--lr 3e-4] [--epochs 4] [--mb 4] [--resume runs/x/last.json] [--eval-every 10]
import fs from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { Pool, summarise } from './pool.mjs';
import { Adam, clipGradNorm } from './nn.mjs';
import { mulberry32 } from './sim/loader.mjs';
import { DEFAULT_ARCH } from './policy.mjs';
import { saveCheckpoint, loadCheckpoint } from './checkpoint.mjs';
import { evalSet } from './courses.mjs';

const {values: a} = parseArgs({options: {
    name: {type: 'string', default: 'run'},
    iters: {type: 'string', default: '300'},
    workers: {type: 'string'},
    steps: {type: 'string', default: '1024'},     // env steps per worker per iteration
    lr: {type: 'string', default: '3e-4'},
    'lr-final': {type: 'string', default: '0.1'}, // fraction of lr at the last iteration
    epochs: {type: 'string', default: '4'},
    mb: {type: 'string', default: '4'},           // minibatches per epoch
    clip: {type: 'string', default: '0.2'},
    'ent-cat': {type: 'string', default: '0.01'},
    'ent-cont': {type: 'string', default: '0'},
    vf: {type: 'string', default: '0.5'},
    'max-grad': {type: 'string', default: '0.5'},
    'target-kl': {type: 'string', default: '0.03'},
    gamma: {type: 'string', default: '1'},
    lambda: {type: 'string', default: '0.95'},
    'classic-prob': {type: 'string', default: '0'},
    'start-prob': {type: 'string', default: '0.3'}, // exploring starts (random spot on the hole)
    hidden: {type: 'string', default: DEFAULT_ARCH.hidden.join(',')},
    resume: {type: 'string'},
    'eval-every': {type: 'string', default: '10'},
    'eval-rounds': {type: 'string', default: '4'},
    seed: {type: 'string', default: '1'},
}});
const num = (k)=> +a[k];

const dir = join('runs', a.name);
fs.mkdirSync(dir, {recursive: true});
let arch = {hidden: a.hidden.split(',').map(Number)};
let resume = a.resume && loadCheckpoint(a.resume);
if (resume) arch = resume.arch;
const pool = new Pool({workers: a.workers && +a.workers, arch, seed: num('seed')});
const P = pool.params, N = pool.model.size;
const adam = new Adam(N, {lr: num('lr')});
let startIter = 0, bestEval = Infinity;
if (resume)
{
    P.set(resume.params);
    if (resume.adam) { adam.t = resume.adam.t; adam.m.set(resume.adam.m); adam.v.set(resume.adam.v); }
    startIter = resume.meta.iter ?? 0;
    bestEval = resume.meta.bestEval ?? Infinity;
}
else
    pool.model.init(mulberry32(num('seed')));

const hp = {clip: num('clip'), entCat: num('ent-cat'), entCont: num('ent-cont'), vf: num('vf')};
const iters = num('iters'), epochs = num('epochs'), nMb = num('mb');
const G = new Float32Array(N);
const logFile = join(dir, 'log.csv');
if (!resume || !fs.existsSync(logFile))
    fs.writeFileSync(logFile, 'iter,steps,episodes,toPar,penalties,pickups,pl,vl,ent,kl,clipfrac,gradNorm,lr,sps,evalClassic,evalRemix\n');
fs.writeFileSync(join(dir, 'config.json'), JSON.stringify({args: a, arch, hp}, null, 2));
console.log(`run ${a.name}: ${pool.size} workers, ${N} params, ${num('steps')*pool.size} steps/iter`);

let totalSteps = 0;
const recent = [];
const ckptMeta = (iter, extra = {})=> ({iter, totalSteps, bestEval, ...extra});
for (let it = startIter; it < iters; ++it)
{
    const t0 = performance.now();
    const frac = it/Math.max(1, iters-1);
    adam.lr = num('lr')*(1 - frac*(1 - num('lr-final')));
    const ro = await pool.all('rollout', {steps: num('steps'), classicProb: num('classic-prob'), startProb: num('start-prob'),
        gamma: num('gamma'), lambda: num('lambda')});
    let n = 0, s = 0, s2 = 0;
    for (const r of ro) { n += r.n; s += r.advSum; s2 += r.advSq; recent.push(...r.eps); }
    totalSteps += n;
    const advMean = s/n, advStd = Math.sqrt(Math.max(1e-12, s2/n - advMean*advMean));
    let st = {pl: 0, vl: 0, ent: 0, kl: 0, clipN: 0, B: 0}, gn = 0, kl = 0;
    outer:
    for (let ep = 0; ep < epochs; ++ep)
    {
        let klEp = 0, bEp = 0;
        for (let mb = 0; mb < nMb; ++mb)
        {
            const res = await pool.all('grad', {epoch: ep, mb, nMb, advMean, advStd, hp});
            let B = 0;
            for (const r of res) { B += r.B; for (const k in st) st[k] += r[k]; klEp += r.kl; bEp += r.B; }
            G.fill(0);
            for (const g of pool.gradViews) for (let i = 0; i < N; ++i) G[i] += g[i];
            for (let i = 0; i < N; ++i) G[i] /= B;
            gn = clipGradNorm(G, num('max-grad'));
            adam.step(P, G);
        }
        kl = klEp/bEp;
        if (kl > num('target-kl')*1.5) break outer;
    }
    while (recent.length > 2000) recent.shift();
    const toPar = recent.reduce((q, e)=> q + e.strokes - e.par, 0)/recent.length;
    const pen = recent.reduce((q, e)=> q + e.penalties, 0)/recent.length;
    const pick = recent.reduce((q, e)=> q + !e.holed, 0)/recent.length;
    const sps = n/((performance.now() - t0)/1000);

    let evC = '', evR = '';
    const everyEval = num('eval-every');
    if (everyEval && ((it+1) % everyEval == 0 || it == iters-1))
    {
        const R = num('eval-rounds');
        const c = summarise(await pool.evaluate(evalSet('classic', R)));
        const r = summarise(await pool.evaluate(evalSet('remix', R)));
        evC = c.toPar.toFixed(2); evR = r.toPar.toFixed(2);
        console.log(`  eval classic ${evC} (best ${c.best}, worst ${c.worst}, pen ${c.penalties.toFixed(1)})`
            + `  remix ${evR} (pen ${r.penalties.toFixed(1)}, pickups ${r.pickups.toFixed(2)})`);
        const score = (c.toPar + r.toPar)/2;
        if (score < bestEval)
        {
            bestEval = score;
            saveCheckpoint(join(dir, 'best.json'), {arch, params: P,
                meta: ckptMeta(it+1, {evalClassic: c.toPar, evalRemix: r.toPar})});
            console.log(`  new best ${score.toFixed(2)} -> ${join(dir, 'best.json')}`);
        }
        saveCheckpoint(join(dir, 'last.json'), {arch, params: P, adam, meta: ckptMeta(it+1)});
    }
    const B = st.B || 1;
    const row = [it+1, totalSteps, recent.length, toPar.toFixed(3), pen.toFixed(3), pick.toFixed(3),
        (st.pl/B).toFixed(4), (st.vl/B).toFixed(4), (st.ent/B).toFixed(3), kl.toFixed(4),
        (st.clipN/B).toFixed(3), gn.toFixed(3), adam.lr.toExponential(2), sps|0, evC, evR];
    fs.appendFileSync(logFile, row.join(',') + '\n');
    console.log(`it ${it+1} steps ${totalSteps} toPar/hole ${toPar.toFixed(3)} pen ${pen.toFixed(2)}`
        + ` pick ${pick.toFixed(3)} vl ${(st.vl/B).toFixed(3)} ent ${(st.ent/B).toFixed(2)}`
        + ` kl ${kl.toFixed(4)} clip ${(st.clipN/B).toFixed(3)} gn ${gn.toFixed(2)} ${sps|0} sps`);
}
saveCheckpoint(join(dir, 'last.json'), {arch, params: P, adam, meta: ckptMeta(iters)});
await pool.close();
