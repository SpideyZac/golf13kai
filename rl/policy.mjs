// The agent: an actor MLP producing a HYBRID action distribution and a
// separate critic MLP, both over one flat parameter buffer.
//
// Actor output layout (HEAD = 18):
//   [0..10]  club logits       categorical over the 11 clubs (10 = putter)
//   [11..13] spin logits       categorical back / none / top
//   [14,15]  mu                Gaussian means of (aim, dist)
//   [16,17]  log std           state dependent, clamped to [LS_MIN, LS_MAX]
import { MLP, randnFrom } from './nn.mjs';
import { OBS_DIM, N_CLUBS, N_SPIN } from './env.mjs';

export const HEAD = N_CLUBS + N_SPIN + 4;
const C0 = 0, S0 = N_CLUBS, MU = N_CLUBS + N_SPIN, LS = MU + 2;
export const LS_MIN = -6, LS_MAX = 1;
const LOG2PI = Math.log(2*Math.PI);
// initial log std of (aim, dist): wide enough to explore doglegs and layups
const LS_INIT = [-.7, -.7];
// critic output offset: returns are about -4 strokes a hole
const V_BIAS = -4;

export const DEFAULT_ARCH = {hidden: [256, 128]};

export class Model
{
    constructor(arch = DEFAULT_ARCH, obsDim = OBS_DIM)
    {
        this.arch = arch; this.obsDim = obsDim;
        let n = 0;
        const alloc = (k)=> { const o = n; n += k; return o; };
        this.actor = new MLP([obsDim, ...arch.hidden, HEAD], alloc);
        this.critic = new MLP([obsDim, ...arch.hidden, 1], alloc);
        this.size = n;
    }

    bind(params, grads)
    {
        this.params = params; this.grads = grads;
        this.actor.bind(params, grads);
        this.critic.bind(params, grads);
        return this;
    }

    init(rand)
    {
        const randn = randnFrom(rand);
        this.actor.init(randn, 1, .01);
        this.critic.init(randn, 1, 1);
        const ab = this.actor.layers.at(-1).b;
        ab[LS] = LS_INIT[0]; ab[LS+1] = LS_INIT[1];
        this.critic.layers.at(-1).b[0] = V_BIAS;
        return this;
    }

    // Sample (or, deterministic, take the mode of) the action for one obs.
    // Returns {club, spin, aim, dist, logp, value}.
    act(obs, rand, randn, deterministic = false)
    {
        const out = this.actor.forward(obs, 1);
        const value = this.critic.forward(obs, 1)[0];
        const club = deterministic ? argmax(out, C0, N_CLUBS) : sampleCat(out, C0, N_CLUBS, rand);
        const spin = deterministic ? argmax(out, S0, N_SPIN) : sampleCat(out, S0, N_SPIN, rand);
        const a = [0, 0];
        for (let d = 0; d < 2; ++d)
            a[d] = out[MU+d] + (deterministic ? 0 : Math.exp(clampLS(out[LS+d]))*randn());
        const act = {club, spin, aim: a[0], dist: a[1]};
        act.logp = logProb(out, 0, act);
        act.value = value;
        return act;
    }

    value(obs) { return this.critic.forward(obs, 1)[0]; }
}

// ---- distribution math on one row of actor output (offset o) ----

const clampLS = (x)=> x < LS_MIN ? LS_MIN : x > LS_MAX ? LS_MAX : x;

function argmax(z, o, n)
{
    let b = 0;
    for (let i = 1; i < n; ++i) if (z[o+i] > z[o+b]) b = i;
    return b;
}

// log-softmax normaliser of z[o..o+n)
function lse(z, o, n)
{
    let mx = -Infinity;
    for (let i = 0; i < n; ++i) mx = Math.max(mx, z[o+i]);
    let s = 0;
    for (let i = 0; i < n; ++i) s += Math.exp(z[o+i] - mx);
    return mx + Math.log(s);
}

function sampleCat(z, o, n, rand)
{
    const L = lse(z, o, n);
    let u = rand();
    for (let i = 0; i < n; ++i)
    {
        u -= Math.exp(z[o+i] - L);
        if (u <= 0) return i;
    }
    return n-1;
}

// row = offset of the row in `out`; act = {club, spin, aim, dist}
export function logProb(out, row, act)
{
    let lp = out[row+C0+act.club] - lse(out, row+C0, N_CLUBS)
           + out[row+S0+act.spin] - lse(out, row+S0, N_SPIN);
    const x = [act.aim, act.dist];
    for (let d = 0; d < 2; ++d)
    {
        const ls = clampLS(out[row+LS+d]), z = (x[d] - out[row+MU+d])/Math.exp(ls);
        lp += -.5*z*z - ls - .5*LOG2PI;
    }
    return lp;
}

// Categorical entropy of z[o..o+n) and, into g[o..], d(-coef*H)/dz.
function catEntropyGrad(z, o, n, coef, g)
{
    const L = lse(z, o, n);
    let H = 0;
    for (let i = 0; i < n; ++i) { const lp = z[o+i] - L; H -= Math.exp(lp)*lp; }
    for (let i = 0; i < n; ++i)
    {
        const lp = z[o+i] - L, p = Math.exp(lp);
        g[o+i] += coef*p*(lp + H);   // -coef * dH/dz_i, dH/dz_i = -p_i(log p_i + H)
    }
    return H;
}

// PPO clipped-surrogate gradient for one sample, written into dOut[row..].
// Returns {ratio, logp, ent} for statistics. `scale` is 1/N of the batch.
export function ppoRowGrad(out, row, act, oldLogp, adv, hp, scale, dOut)
{
    const logp = logProb(out, row, act);
    const ratio = Math.exp(logp - oldLogp);
    const eps = hp.clip;
    const clipped = (adv > 0 && ratio > 1+eps) || (adv < 0 && ratio < 1-eps);
    // d(-min(r*A, clip(r)*A))/d logp
    const g = clipped ? 0 : -adv*ratio*scale;
    const g0 = row;
    for (let i = 0; i < HEAD; ++i) dOut[g0+i] = 0;
    // categorical heads: dlogp/dz = onehot - softmax
    for (const [o, n, k] of [[C0, N_CLUBS, act.club], [S0, N_SPIN, act.spin]])
    {
        const L = lse(out, row+o, n);
        for (let i = 0; i < n; ++i)
            dOut[g0+o+i] += g*((i == k ? 1 : 0) - Math.exp(out[row+o+i] - L));
    }
    let ent = catEntropyGrad(out, row+C0, N_CLUBS, hp.entCat*scale, dOut)
            + catEntropyGrad(out, row+S0, N_SPIN, hp.entCat*scale, dOut);
    const x = [act.aim, act.dist];
    for (let d = 0; d < 2; ++d)
    {
        const raw = out[row+LS+d], ls = clampLS(raw), sd = Math.exp(ls);
        const z = (x[d] - out[row+MU+d])/sd;
        dOut[g0+MU+d] += g*z/sd;
        if (raw > LS_MIN && raw < LS_MAX)
            dOut[g0+LS+d] += g*(z*z - 1) - hp.entCont*scale;
        ent += ls + .5 + .5*LOG2PI;
    }
    return {ratio, logp, ent};
}
