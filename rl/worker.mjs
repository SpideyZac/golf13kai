// A training worker thread: owns one GolfEnv, plays rollouts with the current
// shared parameters, keeps its rollout in memory, and computes PPO gradient
// shards over it on request. Parameters are read straight from the shared
// buffer the trainer writes between rounds; gradients go to this worker's own
// shared buffer for the trainer to sum.
import { parentPort, workerData } from 'node:worker_threads';
import { GolfEnv, OBS_DIM, isStuck } from './env.mjs';
import { Model, ppoRowGrad, HEAD } from './policy.mjs';
import { randnFrom } from './nn.mjs';
import { mulberry32 } from './sim/loader.mjs';
import { trainEpisode } from './courses.mjs';

const {paramSAB, gradSAB, arch, envCfg, seed} = workerData;
const params = new Float32Array(paramSAB), grads = new Float32Array(gradSAB);
const model = new Model(arch).bind(params, grads);
const env = new GolfEnv(envCfg);
const rand = mulberry32(seed), randn = randnFrom(rand);

// rollout storage
let cap = 0, n = 0;
let OBS, CLUB, SPIN, AIM, DIST, IMPACT, LOGP, VAL, ADV, RET;
function ensure(k)
{
    if (k <= cap) return;
    cap = Math.max(k, cap*2);
    const grow = (A, T, w = 1)=> { const B = new T(cap*w); A && B.set(A); return B; };
    OBS = grow(OBS, Float32Array, OBS_DIM);
    CLUB = grow(CLUB, Int8Array); SPIN = grow(SPIN, Int8Array);
    AIM = grow(AIM, Float32Array); DIST = grow(DIST, Float32Array); IMPACT = grow(IMPACT, Float32Array);
    LOGP = grow(LOGP, Float32Array); VAL = grow(VAL, Float32Array);
    ADV = grow(ADV, Float32Array); RET = grow(RET, Float32Array);
}

// Play one episode, storing every step from index n on. Returns the episode summary.
function playEpisode(spec, deterministic, store, gamma, lambda, escape = false)
{
    let obs = env.reset(spec);
    const start = n, rewards = [];
    for (;;)
    {
        const last = env.log.at(-1);
        const det = deterministic && !(escape && last && isStuck(env.prev, last.club == 'PT'));
        const a = model.act(obs, rand, randn, det);
        if (store)
        {
            ensure(n+1);
            OBS.set(obs, n*OBS_DIM);
            CLUB[n] = a.club; SPIN[n] = a.spin; AIM[n] = a.aim; DIST[n] = a.dist; IMPACT[n] = a.impact;
            LOGP[n] = a.logp; VAL[n] = a.value;
            ++n;
        }
        const r = env.step(a);
        rewards.push(r.reward);
        if (r.done) break;
        obs = r.obs;
    }
    if (store)
    {
        // GAE; the episode always terminates (holed or picked up), so V(end) = 0
        let gae = 0;
        for (let t = n-1; t >= start; --t)
        {
            const next = t == n-1 ? 0 : VAL[t+1];
            const delta = rewards[t-start] + gamma*next - VAL[t];
            gae = delta + gamma*lambda*gae;
            ADV[t] = gae;
            RET[t] = gae + VAL[t];
        }
    }
    return {seed: spec.seed, remix: spec.remix, hole: spec.hole, round: spec.round,
        strokes: env.strokes, par: env.h.par, penalties: env.penalties,
        holed: env.log.at(-1).result == 'holed', shots: env.log};
}

let perm = null, mbBuf = null, dOut = null, dV = null;

const handlers =
{
    rollout({steps, classicProb, startProb, gamma, lambda})
    {
        n = 0;
        ensure(steps + 64);
        const eps = [];
        while (n < steps)
        {
            const e = playEpisode(trainEpisode(rand, classicProb, startProb), false, true, gamma, lambda);
            delete e.shots;
            eps.push(e);
        }
        let s = 0, s2 = 0;
        for (let i = 0; i < n; ++i) { s += ADV[i]; s2 += ADV[i]*ADV[i]; }
        return {n, eps, advSum: s, advSq: s2};
    },

    // One minibatch shard: gradients SUMMED over this worker's chunk (the
    // trainer divides by the global minibatch size).
    grad({epoch, mb, nMb, advMean, advStd, hp})
    {
        if (mb == 0)
        {
            perm = Int32Array.from({length: n}, (_, i)=> i);
            for (let i = n; i-- > 1;) { const j = Math.floor(rand()*(i+1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
        }
        const lo = Math.floor(n*mb/nMb), hi = Math.floor(n*(mb+1)/nMb), B = hi - lo;
        if (!mbBuf || mbBuf.length < B*OBS_DIM)
        {
            mbBuf = new Float32Array(B*OBS_DIM);
            dOut = new Float32Array(B*HEAD);
            dV = new Float32Array(B);
        }
        for (let i = 0; i < B; ++i)
            mbBuf.set(OBS.subarray(perm[lo+i]*OBS_DIM, (perm[lo+i]+1)*OBS_DIM), i*OBS_DIM);
        grads.fill(0);
        const out = model.actor.forward(mbBuf, B);
        const v = model.critic.forward(mbBuf, B);
        let pl = 0, vl = 0, ent = 0, kl = 0, clipN = 0;
        for (let i = 0; i < B; ++i)
        {
            const k = perm[lo+i];
            const adv = (ADV[k] - advMean)/(advStd + 1e-8);
            const act = {club: CLUB[k], spin: SPIN[k], aim: AIM[k], dist: DIST[k], impact: IMPACT[k]};
            const r = ppoRowGrad(out, i*HEAD, act, LOGP[k], adv, hp, 1, dOut);
            pl += -Math.min(r.ratio*adv, Math.min(Math.max(r.ratio, 1-hp.clip), 1+hp.clip)*adv);
            ent += r.ent;
            kl += (r.ratio - 1) - Math.log(r.ratio);
            clipN += Math.abs(r.ratio - 1) > hp.clip ? 1 : 0;
            const e = v[i] - RET[k];
            vl += e*e;
            dV[i] = hp.vf*e;
        }
        model.actor.backward(dOut, B);
        model.critic.backward(dV, B);
        return {B, pl, vl, ent, kl, clipN};
    },

    eval({episodes, deterministic, escape})
    {
        return episodes.map(spec => playEpisode(spec, deterministic, false, 1, 1, escape));
    },
};

parentPort.on('message', ({id, type, args})=>
{
    try { parentPort.postMessage({id, result: handlers[type](args)}); }
    catch (e) { parentPort.postMessage({id, error: e.stack || String(e)}); }
});
