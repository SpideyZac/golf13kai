// Minimal dependency-free MLP with manual backprop and Adam, over ONE flat
// Float32Array of parameters (so it can live in a SharedArrayBuffer and be
// read by every worker thread without copying).
//
// Weights are stored [in][out] (k*out + j): the forward and backward inner
// loops run contiguously over outputs and skip zero inputs, which matters
// because the observation is mostly one-hot grid channels.

export class MLP
{
    // sizes = [in, hidden..., out]; hidden layers are tanh, the output linear.
    // alloc(n) returns the offset of n fresh parameters in the flat buffer.
    constructor(sizes, alloc)
    {
        this.sizes = sizes;
        this.layers = [];
        for (let l = 0; l + 1 < sizes.length; ++l)
        {
            const n = sizes[l], m = sizes[l+1];
            this.layers.push({n, m, wOff: alloc(n*m), bOff: alloc(m)});
        }
        this.cache = new Map();
    }

    // point W/b (and gW/gb) views at flat buffers
    bind(params, grads)
    {
        for (const L of this.layers)
        {
            L.W = params.subarray(L.wOff, L.wOff + L.n*L.m);
            L.b = params.subarray(L.bOff, L.bOff + L.m);
            if (grads)
            {
                L.gW = grads.subarray(L.wOff, L.wOff + L.n*L.m);
                L.gb = grads.subarray(L.bOff, L.bOff + L.m);
            }
        }
    }

    // gains: hidden layers `gain`, output layer `outGain`; biases zero
    init(randn, gain = 1, outGain = .01)
    {
        this.layers.forEach((L, l)=>
        {
            const g = (l == this.layers.length-1 ? outGain : gain)/Math.sqrt(L.n);
            for (let i = 0; i < L.W.length; ++i) L.W[i] = randn()*g;
            L.b.fill(0);
        });
    }

    buffers(B)
    {
        let c = this.cache.get(B);
        if (!c)
        {
            c = {acts: this.sizes.map(s => new Float32Array(B*s)),
                 grads: this.sizes.map(s => new Float32Array(B*s))};
            this.cache.set(B, c);
        }
        return c;
    }

    // X: B*in inputs. Returns the B*out output view (valid until the next
    // forward with the same B).
    forward(X, B)
    {
        const c = this.buffers(B);
        c.acts[0].set(X.subarray(0, B*this.sizes[0]));
        this.layers.forEach((L, l)=>
        {
            const A = c.acts[l], Z = c.acts[l+1], {n, m, W, b} = L;
            const last = l == this.layers.length-1;
            for (let i = 0; i < B; ++i)
            {
                const zo = i*m, ao = i*n;
                for (let j = 0; j < m; ++j) Z[zo+j] = b[j];
                for (let k = 0; k < n; ++k)
                {
                    const x = A[ao+k];
                    if (x === 0) continue;
                    const wo = k*m;
                    for (let j = 0; j < m; ++j) Z[zo+j] += x*W[wo+j];
                }
                if (!last)
                    for (let j = 0; j < m; ++j) Z[zo+j] = Math.tanh(Z[zo+j]);
            }
        });
        return c.acts[this.layers.length];
    }

    // dOut: B*out gradient of the loss w.r.t. the outputs of the LAST forward
    // with this B. Accumulates into gW/gb.
    backward(dOut, B)
    {
        const c = this.buffers(B);
        let dZ = dOut;
        for (let l = this.layers.length; l--;)
        {
            const L = this.layers[l], {n, m, W, gW, gb} = L, A = c.acts[l];
            const dA = l ? c.grads[l] : null;
            for (let i = 0; i < B; ++i)
            {
                const zo = i*m, ao = i*n;
                for (let j = 0; j < m; ++j) gb[j] += dZ[zo+j];
                for (let k = 0; k < n; ++k)
                {
                    const x = A[ao+k], wo = k*m;
                    if (x !== 0)
                        for (let j = 0; j < m; ++j) gW[wo+j] += x*dZ[zo+j];
                    if (dA)
                    {
                        let s = 0;
                        for (let j = 0; j < m; ++j) s += W[wo+j]*dZ[zo+j];
                        dA[ao+k] = s*(1 - x*x); // through the tanh that made A
                    }
                }
            }
            dZ = dA;
        }
    }
}

export class Adam
{
    constructor(n, {lr = 3e-4, b1 = .9, b2 = .999, eps = 1e-8} = {})
    {
        Object.assign(this, {lr, b1, b2, eps, t: 0});
        this.m = new Float32Array(n);
        this.v = new Float32Array(n);
    }

    step(params, grads)
    {
        const {b1, b2, eps, m, v} = this;
        ++this.t;
        const a = this.lr*Math.sqrt(1 - b2**this.t)/(1 - b1**this.t);
        for (let i = 0; i < params.length; ++i)
        {
            const g = grads[i];
            m[i] = b1*m[i] + (1-b1)*g;
            v[i] = b2*v[i] + (1-b2)*g*g;
            params[i] -= a*m[i]/(Math.sqrt(v[i]) + eps);
        }
    }
}

// Scale grads in place so their global L2 norm is at most maxNorm; returns
// the norm before clipping.
export function clipGradNorm(grads, maxNorm)
{
    let s = 0;
    for (let i = 0; i < grads.length; ++i) s += grads[i]*grads[i];
    const norm = Math.sqrt(s);
    if (norm > maxNorm)
    {
        const k = maxNorm/norm;
        for (let i = 0; i < grads.length; ++i) grads[i] *= k;
    }
    return norm;
}

export function randnFrom(rand)
{
    let spare = null;
    return ()=>
    {
        if (spare !== null) { const s = spare; spare = null; return s; }
        let u, v, s;
        do { u = rand()*2 - 1; v = rand()*2 - 1; s = u*u + v*v; } while (!s || s >= 1);
        const k = Math.sqrt(-2*Math.log(s)/s);
        spare = v*k;
        return u*k;
    };
}
