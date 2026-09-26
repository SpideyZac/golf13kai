// The Rust port (golfsim/) against the game itself: the same holes, the same
// swings, compared with ===. Every hole's layout, every observation float and
// every ball position must match the JS env (which runs the game's own
// source) BIT FOR BIT - the port is only trusted because of this test.
//
// Needs the binary: cd golfsim && cargo build --release. Without it the test
// is skipped. PARITY_HOLES=n plays more holes (default 120).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { GolfEnv, OBS_DIM, N_CLUB_ACTIONS, CLUB_SOLVE } from '../env.mjs';
import { loadModel } from '../checkpoint.mjs';
import { mulberry32 } from '../sim/loader.mjs';
import { randnFrom } from '../nn.mjs';
import { trainEpisode, evalSet } from '../courses.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const BIN = join(ROOT, 'golfsim/target/release', process.platform == 'win32' ? 'golfsim-cli.exe' : 'golfsim-cli');
const HOLES = +(process.env.PARITY_HOLES ?? 120);

// obs index -> block name, for readable failures (layout: docs/SPEC.md)
const BLOCKS = [['lie', 7], ['lieCost', 2], ['wind', 2], ['par', 3], ['round', 3], ['slope', 2], ['pinH', 1],
    ['pinDist', 3], ['club', 11], ['path', 3], ['lastShot', 3], ['rays', 26], ['vsPin', 6], ['putt', 8],
    ['flight', 5], ['centreline', 20], ['gridA', 648], ['gridB', 630], ['solver', 6]];
const blockOf = (i)=> { for (const [n, k] of BLOCKS) { if (i < k) return `${n}[${i}]`; i -= k; } };

class Rust
{
    constructor()
    {
        this.p = spawn(BIN, ['serve'], {stdio: ['pipe', 'pipe', 'inherit']});
        this.waiting = [];
        createInterface({input: this.p.stdout}).on('line', l => this.waiting.shift()(JSON.parse(l)));
    }
    call(req) { return new Promise(r => { this.waiting.push(r); this.p.stdin.write(JSON.stringify(req) + '\n'); }); }
    close() { this.p.stdin.end(); }
}

function sameObs(js, rs, where)
{
    assert.equal(rs.length, OBS_DIM);
    for (let i = 0; i < OBS_DIM; ++i)
        if (!Object.is(js[i], rs[i]))
            assert.fail(`${where}: obs ${blockOf(i)} js ${js[i]} rust ${rs[i]}`);
}

function sameHole(env, h, where)
{
    const G = env.G, H = G.hole;
    const eq = (a, b, what)=> assert.ok(Object.is(a, b), `${where}: ${what} js ${a} rust ${b}`);
    eq(H.par, h.par, 'par'); eq(H.len, h.len, 'len'); eq(H.gr, h.gr, 'gr'); eq(G.noiseSeed, h.noiseSeed, 'noiseSeed');
    eq(H.pin.x, h.pin[0], 'pin.x'); eq(H.pin.z, h.pin[1], 'pin.z');
    eq(H.greenH, h.greenH, 'greenH'); eq(H.teeH, h.teeH, 'teeH');
    eq(H.wind.a, h.wind[0], 'wind.a'); eq(H.wind.s, h.wind[1], 'wind.s');
    eq(H.path.length, h.path.length, 'path length');
    H.path.forEach((p, i)=> { eq(p.x, h.path[i][0], `path[${i}].x`); eq(p.z, h.path[i][1], `path[${i}].z`); eq(p.along, h.path[i][2], `path[${i}].along`); });
    eq(H.bunkers.length, h.bunkers.length, 'bunkers');
    H.bunkers.forEach((b, i)=> ['x', 'z', 'rx', 'rz'].forEach((k, j)=> eq(b[k], h.bunkers[i][j], `bunker[${i}].${k}`)));
    eq(H.waters.length, h.waters.length, 'waters');
    H.waters.forEach((w, i)=> ['x', 'z', 'rx', 'rz', 'h'].forEach((k, j)=> eq(w[k], h.waters[i][j], `water[${i}].${k}`)));
    eq(H.near.length, h.nNear, 'near props');
    if (h.near)
        H.near.forEach((t, i)=> ['x', 'z', 's', 'y'].forEach((k, j)=> eq(t[k], h.near[i][j], `near[${i}].${k}`)));
    ['x', 'y', 'z'].forEach((k, j)=> eq(G.ball[k], h.ball[j], `ball.${k}`));
}

const haveBin = fs.existsSync(BIN);
const skip = haveBin ? false : `no ${BIN} (cd golfsim && cargo build --release)`;

test('jsmath matches V8 bit for bit', {skip}, async ()=>
{
    const rust = new Rust();
    const r = mulberry32(99), x = [], y = [];
    for (let i = 0; i < 20000; ++i)
    {
        // hashN arguments (big), angles, and ordinary magnitudes
        const k = i % 3;
        x.push(k == 0 ? (r()*2 - 1)*2e5 : k == 1 ? (r()*2 - 1)*7 : (r()*2 - 1)*300);
        y.push((r()*2 - 1)*(k == 0 ? 50 : 400));
    }
    const out = await rust.call({cmd: 'math', x, y});
    rust.close();
    const fns = {sin: Math.sin, cos: Math.cos, tanh: Math.tanh, exp: Math.exp, log: v => Math.log(Math.abs(v)),
        log1p: v => Math.log1p(Math.abs(v)), atan2: Math.atan2, hypot2: Math.hypot, hypot3: (a, b)=> Math.hypot(a, b, a - b)};
    for (const [name, f] of Object.entries(fns))
        x.forEach((v, i)=>
        {
            const want = f(v, y[i]), got = out[name][i];
            if (!(Object.is(want, got) || (!isFinite(want) && got === null)))
                assert.fail(`${name}(${v}${f.length > 1 ? ', ' + y[i] : ''}): js ${want} rust ${got}`);
        });
});

test('every classic hole and remix holes generate identically', {skip}, async ()=>
{
    const rust = new Rust(), env = new GolfEnv();
    const specs = [...evalSet('classic', 1), ...evalSet('remix', 2)];
    const r = mulberry32(5);
    for (let i = 0; i < 18; ++i) specs.push(trainEpisode(r, 0, 0));
    for (const spec of specs)
    {
        const obs = env.reset(spec);
        const res = await rust.call({cmd: 'reset', spec, detail: true});
        const where = `${spec.remix ? 'remix' : 'classic'} ${spec.seed} hole ${spec.hole + 1}`;
        sameHole(env, res.hole, where);
        sameObs(obs, res.obs, where);
    }
    rust.close();
});

// HOLES holes under env rules cfg (the Rust side gets them with each reset)
async function playHoles(holes, cfg, seed)
{
    const rust = new Rust(), env = new GolfEnv(cfg);
    const {model} = loadModel(join(ROOT, 'models/agent.json'));
    const rand = mulberry32(seed), randn = randnFrom(rand);
    let strokes = 0, events = {};
    for (let e = 0; e < holes; ++e)
    {
        // half from exploring starts (sand, trees, hillsides), a few classic
        const spec = trainEpisode(rand, .15, .5);
        let obs = env.reset(spec);
        const res = await rust.call({cmd: 'reset', spec, cfg});
        const where = `hole ${e} (${JSON.stringify(spec)})`;
        sameHole(env, res.hole, where);
        sameObs(obs, res.obs, where);
        // the shipped policy, sampled, with every fifth shot random instead
        // (odd clubs and spins, putts from off the green, wild aims) and one
        // in four the solver's shot, so its swings are compared too
        for (let s = 0; ; ++s)
        {
            const u = rand();
            const a = u < .2
                ? {club: Math.floor(rand()*N_CLUB_ACTIONS), spin: Math.floor(rand()*3), aim: (rand()*2 - 1)*3, dist: rand()*6 - 4}
                : u < .45 ? {club: CLUB_SOLVE, spin: Math.floor(rand()*3), aim: 0, dist: 0}
                : model.act(obs, rand, randn, false);
            const action = {club: a.club, spin: a.spin, aim: a.aim, dist: a.dist};
            const js = env.step(action);
            const rs = await rust.call({cmd: 'step', action});
            const at = `${where} stroke ${s + 1} ${JSON.stringify(action)}`;
            const sh = env.log.at(-1);
            assert.equal(rs.result, js.info.result, `${at}: result`);
            assert.equal(rs.reward, js.reward, `${at}: reward`);
            assert.equal(rs.done, js.done, `${at}: done`);
            assert.equal(rs.strokes, env.strokes, `${at}: strokes`);
            ['x', 'y', 'z'].forEach((k, j)=> assert.ok(Object.is(env.G.ball[k], rs.ball[j]),
                `${at}: ball.${k} js ${env.G.ball[k]} rust ${rs.ball[j]}`));
            assert.ok(Object.is(env.prev.moved, rs.prev[0]), `${at}: moved`);
            assert.equal(rs.prev[1], env.prev.tree, `${at}: tree`);
            assert.equal(rs.shot.lie, sh.lie, `${at}: lie`);
            assert.ok(Object.is(rs.shot.power, sh.power) && Object.is(rs.shot.want, sh.want), `${at}: decode`);
            assert.equal(rs.shot.solved, sh.solved, `${at}: solved`);
            events.solved = (events.solved ?? 0) + sh.solved;
            events.solvedIn = (events.solvedIn ?? 0) + (sh.solved && js.info.result == 'holed' ? 1 : 0);
            events[js.info.result] = (events[js.info.result] ?? 0) + 1;
            events.tree = (events.tree ?? 0) + env.prev.tree;
            ++strokes;
            if (js.done) break;
            sameObs(js.obs, rs.obs, at);
            obs = js.obs;
        }
    }
    rust.close();
    console.log(`  parity${Object.keys(cfg).length ? ' ' + JSON.stringify(cfg) : ''}: ${holes} holes, ${strokes} strokes identical`, events);
}

test(`${HOLES} holes of play match shot for shot`, {skip, timeout: 600000}, ()=> playHoles(HOLES, {}, 2026));

test('the optional rules match too: noise-free solver shots, and no noise at all', {skip, timeout: 600000}, async ()=>
{
    await playHoles(Math.ceil(HOLES/3), {exactSolve: true}, 7);
    await playHoles(Math.ceil(HOLES/6), {exactSolve: true, impactNoise: 0, aimNoise: 0}, 8);
});
