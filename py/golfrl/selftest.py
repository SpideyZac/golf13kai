"""Checks that what Python trains is what the browser plays.

1. models/agent.json round-trips through the PyTorch model byte for byte.
2. On real observations from the Rust env, the PyTorch actor and critic
   match rl/policy.mjs's (the browser's network) to float32 rounding.
3. log_prob matches the JS logProb, so PPO optimises the policy the browser
   samples from.
4. A trainer checkpoint loads in the JS tools (rl/checkpoint.mjs).

usage: python -m golfrl.selftest   (needs node on PATH)
"""
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import sim
from .model import Model, load_js, log_prob, sample, save_js

JS = r"""
import { loadModel } from './rl/checkpoint.mjs';
import { logProb, HEAD } from './rl/policy.mjs';
let s = ''; process.stdin.on('data', d => s += d); process.stdin.on('end', ()=>
{
    const {file, obs, acts} = JSON.parse(s);
    const {model} = loadModel(file);
    const out = [], val = [], lp = [];
    obs.forEach((o, i)=>
    {
        const x = Float32Array.from(o);
        const a = model.actor.forward(x, 1);
        out.push(Array.from(a));
        val.push(model.critic.forward(x, 1)[0]);
        lp.push(logProb(a, 0, acts[i]));
    });
    process.stdout.write(JSON.stringify({out, val, lp}));
});
"""


def run_node(payload):
    r = subprocess.run(['node', '--input-type=module', '-e', JS], input=json.dumps(payload), capture_output=True,
                       text=True, cwd=sim.ROOT, check=True)
    return json.loads(r.stdout)


def main():
    agent = sim.ROOT / 'models' / 'agent.json'
    model, _ = load_js(agent)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'rt.json'
        save_js(p, model, {})
        a = json.loads(agent.read_text())['params']
        b = json.loads(p.read_text())['params']
        assert base64.b64decode(a) == base64.b64decode(b), 'checkpoint round trip changed the parameters'
    print('ok  models/agent.json round-trips through PyTorch unchanged')

    # observations from real play: tee shots, approaches, putts
    env = sim.VecEnv(64, seed=3, start_prob=.6)
    obs = [env.reset_train().copy()]
    with torch.no_grad():
        for _ in range(3):
            out, _ = model(torch.from_numpy(env.obs))
            c, s, k = sample(out)
            k = k.double().numpy()
            env.step(c.numpy(), s.numpy(), k[:, 0], k[:, 1], k[:, 2])
            obs.append(env.obs.copy())
    env.close()
    X = torch.from_numpy(np.concatenate(obs))
    with torch.no_grad():
        out, val = model(X)
        club, spin, cont = sample(out)
        lp = log_prob(out, club, spin, cont)
    acts = [{'club': int(c), 'spin': int(s), 'aim': float(k[0]), 'dist': float(k[1]), 'impact': float(k[2])}
            for c, s, k in zip(club, spin, cont)]
    js = run_node({'file': str(agent), 'obs': X.tolist(), 'acts': acts})
    d_out = np.abs(np.array(js['out']) - out.numpy()).max()
    d_val = np.abs(np.array(js['val']) - val.numpy()).max()
    d_lp = np.abs(np.array(js['lp']) - lp.numpy()).max()
    print(f'ok  actor/critic vs rl/policy.mjs on {len(X)} observations: max |diff| {d_out:.2e} / {d_val:.2e}')
    assert d_out < 1e-4 and d_val < 1e-4, 'the PyTorch network does not match the browser network'
    print(f'ok  log_prob vs rl/policy.mjs logProb: max |diff| {d_lp:.2e}')
    assert d_lp < 1e-3

    # a fresh (trainer-initialised) model saved by Python, read by the JS tools
    fresh = Model().init_like_js(7)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'fresh.json'
        save_js(p, fresh, {'iter': 0})
        js = run_node({'file': str(p), 'obs': X[:4].tolist(), 'acts': acts[:4]})
        with torch.no_grad():
            o, v = fresh(X[:4])
        assert np.abs(np.array(js['out']) - o.numpy()).max() < 1e-4
    print('ok  a trainer checkpoint loads and runs in rl/checkpoint.mjs')


if __name__ == '__main__':
    sys.exit(main())
