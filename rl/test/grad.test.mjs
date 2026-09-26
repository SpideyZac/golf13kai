// Finite-difference checks of the hand-written backprop and PPO gradients.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MLP, randnFrom } from '../nn.mjs';
import { mulberry32 } from '../sim/loader.mjs';
import { Model, ppoRowGrad, logProb, HEAD } from '../policy.mjs';
import { N_CLUB_ACTIONS, N_SPIN } from '../obs.mjs';

function flat(n) { return [new Float32Array(n), new Float32Array(n)]; }

test('MLP backward matches finite differences', ()=>
{
    let n = 0;
    const mlp = new MLP([5, 7, 3], k => (n += k) - k);
    const [p, g] = flat(n);
    mlp.bind(p, g);
    const rand = mulberry32(3), randn = randnFrom(rand);
    mlp.init(randn, 1, 1);
    const B = 4, X = Float32Array.from({length: B*5}, ()=> randn());
    X[2] = 0; // exercise the zero-skip path
    const C = Float32Array.from({length: B*3}, ()=> randn());
    const loss = ()=> { const y = mlp.forward(X, B); let s = 0; for (let i = 0; i < y.length; ++i) s += y[i]*C[i]; return s; };
    loss();
    mlp.backward(C, B);
    for (let i = 0; i < n; i += 3)
    {
        const e = 1e-2, v = p[i];
        p[i] = v + e; const a = loss();
        p[i] = v - e; const b = loss();
        p[i] = v;
        const fd = (a - b)/(2*e);
        assert.ok(Math.abs(fd - g[i]) < 2e-3 + 2e-2*Math.abs(fd), `param ${i}: fd ${fd} vs ${g[i]}`);
    }
});

test('PPO row gradient matches finite differences of the surrogate', ()=>
{
    const hp = {clip: .2, entCat: .05, entCont: .01};
    const rand = mulberry32(7), randn = randnFrom(rand);
    const out = Float32Array.from({length: HEAD}, ()=> randn()*.5);
    const act = {club: 3, spin: 2, aim: .3, dist: -.4};
    const oldLogp = logProb(out, 0, act) + .05, adv = 1.3;
    // the surrogate PPO minimises for this row (unclipped region), minus entropy
    const entropy = (o)=>
    {
        let H = 0;
        for (const [a, n] of [[0, N_CLUB_ACTIONS], [N_CLUB_ACTIONS, N_SPIN]])
        {
            const m = Math.max(...o.slice(a, a+n)); let s = 0;
            for (let i = 0; i < n; ++i) s += Math.exp(o[a+i]-m);
            const L = m + Math.log(s);
            for (let i = 0; i < n; ++i) H -= Math.exp(o[a+i]-L)*(o[a+i]-L);
        }
        return H;
    };
    const f = (o)=> -Math.exp(logProb(o, 0, act) - oldLogp)*adv - hp.entCat*entropy(o)
        - hp.entCont*(o[HEAD-2] + o[HEAD-1]);
    const d = new Float32Array(HEAD);
    ppoRowGrad(out, 0, act, oldLogp, adv, hp, 1, d);
    for (let i = 0; i < HEAD; ++i)
    {
        const e = 1e-3, v = out[i];
        out[i] = v + e; const a = f(out);
        out[i] = v - e; const b = f(out);
        out[i] = v;
        const fd = (a - b)/(2*e);
        assert.ok(Math.abs(fd - d[i]) < 1e-3 + 1e-2*Math.abs(fd), `out ${i}: fd ${fd} vs ${d[i]}`);
    }
});

test('Model init produces a sane action', ()=>
{
    const m = new Model({hidden: [16, 8]});
    const [p, g] = flat(m.size);
    m.bind(p, g).init(mulberry32(1));
    const rand = mulberry32(2), randn = randnFrom(rand);
    const a = m.act(new Float32Array(m.obsDim).fill(.1), rand, randn);
    assert.ok(a.club >= 0 && a.club < N_CLUB_ACTIONS && a.spin >= 0 && a.spin < 3);
    assert.ok(Number.isFinite(a.logp) && Math.abs(a.value + 4) < 3);
});
