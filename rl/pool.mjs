// Worker-thread pool shared by training and evaluation. Parameters live in a
// SharedArrayBuffer every worker reads; each worker has its own gradient buffer.
import { Worker } from 'node:worker_threads';
import os from 'node:os';
import { Model } from './policy.mjs';

export class Pool
{
    constructor({workers = Math.max(1, os.availableParallelism() - 1), arch, envCfg = {}, seed = 1})
    {
        this.model = new Model(arch);
        this.paramSAB = new SharedArrayBuffer(this.model.size*4);
        this.params = new Float32Array(this.paramSAB);
        this.model.bind(this.params);
        this.gradViews = [];
        this.workers = [];
        this.pending = new Map();
        this.nextId = 0;
        for (let i = 0; i < workers; ++i)
        {
            const gradSAB = new SharedArrayBuffer(this.model.size*4);
            this.gradViews.push(new Float32Array(gradSAB));
            const w = new Worker(new URL('./worker.mjs', import.meta.url), {workerData:
                {paramSAB: this.paramSAB, gradSAB, arch, envCfg, seed: seed*7919 + i*104729 + 1}});
            w.on('message', ({id, result, error})=>
            {
                const p = this.pending.get(id);
                this.pending.delete(id);
                error ? p.reject(new Error(error)) : p.resolve(result);
            });
            w.on('error', e => { for (const p of this.pending.values()) p.reject(e); });
            this.workers.push(w);
        }
    }

    get size() { return this.workers.length; }

    call(i, type, args)
    {
        const id = this.nextId++;
        return new Promise((resolve, reject)=>
        {
            this.pending.set(id, {resolve, reject});
            this.workers[i].postMessage({id, type, args});
        });
    }

    all(type, args) { return Promise.all(this.workers.map((_, i)=> this.call(i, type, args))); }

    // Spread episode specs over the workers; results come back in spec order.
    async evaluate(episodes, deterministic = true, escape = false)
    {
        const k = this.size, chunks = Array.from({length: k}, ()=> []);
        episodes.forEach((e, i)=> chunks[i % k].push(e));
        const res = await Promise.all(chunks.map((c, i)=> this.call(i, 'eval', {episodes: c, deterministic, escape})));
        const out = new Array(episodes.length);
        res.forEach((r, i)=> r.forEach((e, j)=> { out[j*k + i] = e; }));
        return out;
    }

    close() { return Promise.all(this.workers.map(w => w.terminate())); }
}

// Summarise a list of 18-hole-round episode results.
export function summarise(results)
{
    const rounds = new Map();
    for (const e of results)
    {
        const r = rounds.get(e.round) ?? {strokes: 0, par: 0, holes: 0, pen: 0, pickups: 0};
        r.strokes += e.strokes; r.par += e.par; r.holes++; r.pen += e.penalties; r.pickups += !e.holed;
        rounds.set(e.round, r);
    }
    const rs = [...rounds.values()];
    const mean = (f)=> rs.reduce((s, r)=> s + f(r), 0)/rs.length;
    return {rounds: rs.length, strokes: mean(r => r.strokes), toPar: mean(r => r.strokes - r.par),
        penalties: mean(r => r.pen), pickups: mean(r => r.pickups),
        best: Math.min(...rs.map(r => r.strokes - r.par)), worst: Math.max(...rs.map(r => r.strokes - r.par))};
}
