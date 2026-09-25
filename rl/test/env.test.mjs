// GolfEnv contract: shapes, determinism, rules.
import test from 'node:test';
import assert from 'node:assert/strict';
import { GolfEnv, OBS_DIM } from '../env.mjs';
import { evalSet, trainEpisode, TRAIN_SEED_MIN } from '../courses.mjs';
import { mulberry32 } from '../sim/loader.mjs';

const play = (env, spec, policy)=>
{
    const obs = [env.reset(spec).slice()];
    let r;
    do { r = env.step(policy(env)); r.obs && obs.push(r.obs.slice()); } while (!r.done);
    return {strokes: env.strokes, obs, log: env.log};
};
const naive = (env)=> ({club: env.G.autoClub(), spin: 1, aim: 0, dist: 0});

test('observation has the declared size and is finite', ()=>
{
    const env = new GolfEnv();
    for (const spec of evalSet('remix', 1).slice(0, 6))
        for (const o of play(env, spec, naive).obs)
        {
            assert.equal(o.length, OBS_DIM);
            assert.ok(o.every(Number.isFinite));
        }
});

test('episodes are deterministic given the spec', ()=>
{
    const a = new GolfEnv(), b = new GolfEnv();
    const spec = {seed: 4242, remix: true, hole: 7, rngSeed: 99};
    assert.deepEqual(play(a, spec, naive).log, play(b, spec, naive).log);
});

test('instances are isolated (interleaved play matches solo play)', ()=>
{
    const s1 = {seed: 1113, remix: false, hole: 2, rngSeed: 5}, s2 = {seed: 77, remix: true, hole: 11, rngSeed: 6};
    const solo = play(new GolfEnv(), s1, naive).log;
    const a = new GolfEnv(), b = new GolfEnv();
    a.reset(s1); b.reset(s2);
    let r;
    do { b.done || b.step(naive(b)); r = a.step(naive(a)); } while (!r.done);
    assert.deepEqual(a.log, solo);
});

test('the mercy rule ends the hole at par+5', ()=>
{
    const env = new GolfEnv();
    // a putter from the tee aimed backwards never gets there
    const {strokes} = play(env, {hole: 0, rngSeed: 1}, ()=> ({club: 10, spin: 1, aim: 9, dist: -9}));
    assert.ok(strokes >= env.h.par + 5 && strokes <= env.h.par + 6);
});

test('training never samples an eval course', ()=>
{
    const rand = mulberry32(1);
    for (let i = 0; i < 1000; ++i)
    {
        const e = trainEpisode(rand);
        assert.ok(e.remix && e.seed >= TRAIN_SEED_MIN && e.hole >= 0 && e.hole < 18);
    }
});

test('exploring starts land on playable ground away from the tee', ()=>
{
    const env = new GolfEnv(), rand = mulberry32(9);
    let away = 0;
    for (let i = 0; i < 60; ++i)
    {
        const spec = {...trainEpisode(rand), start: {u: rand(), v: rand()}};
        env.reset(spec);
        const b = env.G.ball, s = env.G.surfaceAt(b.x, b.z);
        assert.ok(s < env.G.SURF_WATER, `landed on surface ${s}`);
        away += Math.hypot(b.x, b.z) > 20;
    }
    assert.ok(away > 45);
});
