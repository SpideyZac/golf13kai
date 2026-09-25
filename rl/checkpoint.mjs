// Checkpoints: one JSON file holding the architecture, the observation
// version it was trained against, the parameters (base64 float32) and,
// optionally, the Adam state for resuming.
import fs from 'node:fs';
import { OBS_VERSION, OBS_DIM } from './env.mjs';
import { Model } from './policy.mjs';

const enc = (f32)=> Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString('base64');
const dec = (s)=> { const b = Buffer.from(s, 'base64'); return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); };

export function saveCheckpoint(file, {arch, params, adam, meta = {}})
{
    const j = {format: 1, obsVersion: OBS_VERSION, obsDim: OBS_DIM, arch, meta, params: enc(params)};
    if (adam) j.adam = {t: adam.t, m: enc(adam.m), v: enc(adam.v)};
    fs.writeFileSync(file + '.tmp', JSON.stringify(j));
    fs.renameSync(file + '.tmp', file);
}

export function loadCheckpoint(file)
{
    const j = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (j.obsVersion != OBS_VERSION || j.obsDim != OBS_DIM)
        throw new Error(`${file}: trained on obs v${j.obsVersion} (${j.obsDim}), env is v${OBS_VERSION} (${OBS_DIM})`);
    const params = dec(j.params);
    const adam = j.adam && {t: j.adam.t, m: dec(j.adam.m), v: dec(j.adam.v)};
    return {arch: j.arch, meta: j.meta, params, adam};
}

// A ready-to-act Model on the main thread.
export function loadModel(file)
{
    const c = loadCheckpoint(file);
    const m = new Model(c.arch);
    if (m.size != c.params.length) throw new Error('checkpoint size mismatch');
    return {model: m.bind(c.params), ...c};
}
