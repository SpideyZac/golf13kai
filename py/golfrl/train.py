"""PPO trainer: the Rust env on every CPU core, the networks on the GPU.

The algorithm is rl/train.mjs's (PPO, GAE, clipped surrogate, separate actor
and critic, Adam, global grad-norm clip, linear LR decay, KL early stop), with
vectorised envs in place of worker threads: `--envs` envs each play `--steps`
strokes per iteration, and an unfinished episode is bootstrapped from the
critic (it carries on into the next iteration). See docs/TRAINING.md.

usage: python -m golfrl.train --name r5 [--iters 400] [--envs 512 --steps 32]
       [--lr 3e-4] [--device cuda] [--resume runs/r5/last.pt] [--init models/agent.json]

Writes runs/<name>/: log.csv (the JS trainer's columns, so `npm run plot`
works), config.json, best.json and last.json (JS format, for the browser and
rl/eval.mjs) and last.pt (full state, for --resume).
"""
import argparse
import collections
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import sim
from .evaluate import play, summarise
from .model import DEFAULT_HIDDEN, Model, entropies, load_js, log_prob, sample, save_js

LOG_COLS = ['iter', 'steps', 'episodes', 'toPar', 'penalties', 'pickups', 'pl', 'vl', 'ent', 'kl', 'clipfrac',
            'gradNorm', 'lr', 'sps', 'evalClassic', 'evalRemix']


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='run')
    ap.add_argument('--iters', type=int, default=400)
    ap.add_argument('--envs', type=int, default=512, help='parallel envs')
    ap.add_argument('--steps', type=int, default=32, help='strokes per env per iteration')
    ap.add_argument('--threads', type=int, default=0, help='env threads (0: every core)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lr-final', type=float, default=.1, help='fraction of lr at the last iteration')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--mb', type=int, default=4, help='minibatches per epoch')
    ap.add_argument('--clip', type=float, default=.2)
    ap.add_argument('--ent-cat', type=float, default=.01)
    ap.add_argument('--ent-cont', type=float, default=0.)
    ap.add_argument('--vf', type=float, default=.5)
    ap.add_argument('--max-grad', type=float, default=.5)
    ap.add_argument('--target-kl', type=float, default=.03)
    ap.add_argument('--gamma', type=float, default=1.)
    ap.add_argument('--lambda', dest='lam', type=float, default=.95)
    ap.add_argument('--classic-prob', type=float, default=0.)
    ap.add_argument('--start-prob', type=float, default=.3, help='exploring starts')
    ap.add_argument('--hidden', default=','.join(map(str, DEFAULT_HIDDEN)))
    ap.add_argument('--resume', help='runs/<name>/last.pt')
    ap.add_argument('--init', help='start from a JS-format checkpoint (e.g. models/agent.json)')
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--eval-rounds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--device', default='auto', help='auto, cuda, cuda:1, cpu, ...')
    return ap.parse_args(argv)


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main(argv=None):
    a = parse(argv)
    dev = pick_device(a.device)
    torch.manual_seed(a.seed)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run = sim.ROOT / 'runs' / a.name
    run.mkdir(parents=True, exist_ok=True)

    state = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    hidden = state['hidden'] if state else [int(x) for x in a.hidden.split(',')]
    if a.init and not state:
        model, _ = load_js(a.init)
        hidden = model.hidden
    else:
        model = Model(hidden).init_like_js(a.seed)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-8)
    start_iter, total_steps, best_eval = 0, 0, math.inf
    recent = collections.deque(maxlen=2000)
    if state:
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        start_iter, total_steps, best_eval = state['iter'], state['total_steps'], state['best_eval']
        recent.extend(state.get('recent', []))
        torch.set_rng_state(state['torch_rng'])

    n, T = a.envs, a.steps
    env = sim.VecEnv(n, threads=a.threads, seed=a.seed * 7919 + start_iter, classic_prob=a.classic_prob,
                     start_prob=a.start_prob)
    (run / 'config.json').write_text(json.dumps({'args': vars(a), 'arch': model.arch, 'device': str(dev),
                                                 'trainer': 'python'}, indent=2))
    log_path = run / 'log.csv'
    if not (state and log_path.exists()):
        log_path.write_text(','.join(LOG_COLS) + '\n')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'run {a.name}: {n} envs x {T} steps = {n * T} strokes/iter on {dev}, {n_params} params,'
          f' env threads {a.threads or os.cpu_count()}')

    D = sim.OBS_DIM
    buf_obs = torch.zeros((T, n, D), device=dev)
    buf_club = torch.zeros((T, n), dtype=torch.long, device=dev)
    buf_spin = torch.zeros((T, n), dtype=torch.long, device=dev)
    buf_cont = torch.zeros((T, n, 2), device=dev)
    buf_logp = torch.zeros((T, n), device=dev)
    buf_val = torch.zeros((T, n), device=dev)
    buf_rew = torch.zeros((T, n), device=dev)
    buf_done = torch.zeros((T, n), device=dev)
    obs = torch.from_numpy(env.reset_train()).to(dev)
    meta = lambda it, **k: {'iter': it, 'totalSteps': total_steps, 'bestEval': best_eval, 'trainer': 'python', **k}

    for it in range(start_iter, a.iters):
        t0 = time.perf_counter()
        frac = it / max(1, a.iters - 1)
        lr = a.lr * (1 - frac * (1 - a.lr_final))
        for g in opt.param_groups:
            g['lr'] = lr

        # ---- rollout ----
        model.eval()
        with torch.no_grad():
            for t in range(T):
                out, v = model(obs)
                club, spin, cont = sample(out)
                buf_obs[t] = obs
                buf_club[t], buf_spin[t], buf_cont[t] = club, spin, cont
                buf_logp[t] = log_prob(out, club, spin, cont)
                buf_val[t] = v
                c = cont.double().cpu().numpy()
                o, info = env.step(club.cpu().numpy(), spin.cpu().numpy(), c[:, 0], c[:, 1])
                buf_rew[t] = torch.from_numpy(info[:, sim.REWARD]).to(dev)
                buf_done[t] = torch.from_numpy(info[:, sim.DONE]).to(dev)
                for i in np.flatnonzero(info[:, sim.DONE]):
                    r = info[i]
                    recent.append((r[sim.STROKES] - r[sim.PAR], r[sim.PENALTIES], not r[sim.HOLED]))
                obs = torch.from_numpy(o).to(dev)
            next_v = model.critic(obs).squeeze(-1)
            # GAE; a finished episode's next value is 0 (it ended)
            adv = torch.zeros((T, n), device=dev)
            gae = torch.zeros(n, device=dev)
            for t in reversed(range(T)):
                nv = next_v if t == T - 1 else buf_val[t + 1]
                live = 1 - buf_done[t]
                delta = buf_rew[t] + a.gamma * nv * live - buf_val[t]
                gae = delta + a.gamma * a.lam * live * gae
                adv[t] = gae
            ret = adv + buf_val
        total_steps += n * T
        t_roll = time.perf_counter() - t0

        # ---- update ----
        model.train()
        B = n * T
        f_obs, f_club, f_spin = buf_obs.view(B, D), buf_club.view(B), buf_spin.view(B)
        f_cont, f_logp, f_adv, f_ret = buf_cont.view(B, 2), buf_logp.view(B), adv.view(B), ret.view(B)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std(unbiased=False) + 1e-8)
        st = collections.Counter()
        kl = gn = 0.
        for ep in range(a.epochs):
            perm = torch.randperm(B, device=dev)
            kl_sum, kl_n = torch.zeros((), device=dev), 0
            for mb in range(a.mb):
                idx = perm[B * mb // a.mb:B * (mb + 1) // a.mb]
                out, v = model(f_obs[idx])
                logp = log_prob(out, f_club[idx], f_spin[idx], f_cont[idx])
                ratio = (logp - f_logp[idx]).exp()
                ad = f_adv[idx]
                pl = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad)
                h_cat, h_gauss, ls = entropies(out)
                e = v - f_ret[idx]
                loss = (pl.mean() - a.ent_cat * h_cat.mean() - a.ent_cont * ls.sum(-1).mean()
                        + a.vf * .5 * (e * e).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad))
                opt.step()
                with torch.no_grad():
                    k = (ratio - 1) - ratio.log()
                    kl_sum += k.sum()
                    kl_n += len(idx)
                    st['pl'] += float(pl.sum())
                    st['vl'] += float((e * e).sum())
                    st['ent'] += float((h_cat + h_gauss).sum())
                    st['clip'] += float(((ratio - 1).abs() > a.clip).sum())
                    st['B'] += len(idx)
            kl = float(kl_sum) / kl_n
            if kl > a.target_kl * 1.5:
                break
        dt = time.perf_counter() - t0

        # ---- log, eval, checkpoints ----
        rc = np.array(recent) if recent else np.zeros((1, 3))
        to_par, pen, pick = rc[:, 0].mean(), rc[:, 1].mean(), rc[:, 2].mean()
        ev_c = ev_r = ''
        if a.eval_every and ((it + 1) % a.eval_every == 0 or it == a.iters - 1):
            c = summarise(play(model, sim.eval_set('classic', a.eval_rounds), dev, threads=a.threads))
            r = summarise(play(model, sim.eval_set('remix', a.eval_rounds), dev, threads=a.threads))
            ev_c, ev_r = f"{c['toPar']:.2f}", f"{r['toPar']:.2f}"
            print(f"  eval classic {ev_c} (best {c['best']}, worst {c['worst']}, pen {c['penalties']:.1f})"
                  f"  remix {ev_r} (pen {r['penalties']:.1f}, pickups {r['pickups']:.2f})")
            score = (c['toPar'] + r['toPar']) / 2
            if score < best_eval:
                best_eval = score
                save_js(run / 'best.json', model, meta(it + 1, evalClassic=c['toPar'], evalRemix=r['toPar']))
                print(f'  new best {score:.2f} -> {run / "best.json"}')
            save_js(run / 'last.json', model, meta(it + 1))
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'hidden': hidden, 'iter': it + 1,
                        'total_steps': total_steps, 'best_eval': best_eval, 'recent': list(recent),
                        'torch_rng': torch.get_rng_state(), 'args': vars(a)}, run / 'last.pt')
        Bn = max(1, st['B'])
        row = [it + 1, total_steps, len(recent), f'{to_par:.3f}', f'{pen:.3f}', f'{pick:.3f}', f"{st['pl'] / Bn:.4f}",
               f"{st['vl'] / Bn:.4f}", f"{st['ent'] / Bn:.3f}", f'{kl:.4f}', f"{st['clip'] / Bn:.3f}", f'{gn:.3f}',
               f'{lr:.2e}', int(n * T / dt), ev_c, ev_r]
        with log_path.open('a', newline='') as fh:
            csv.writer(fh).writerow(row)
        print(f"it {it + 1} steps {total_steps} toPar/hole {to_par:.3f} pen {pen:.2f} pick {pick:.3f}"
              f" vl {st['vl'] / Bn:.3f} ent {st['ent'] / Bn:.2f} kl {kl:.4f} clip {st['clip'] / Bn:.3f} gn {gn:.2f}"
              f" {int(n * T / dt)} sps (rollout {t_roll:.1f}s, update {dt - t_roll:.1f}s)", flush=True)
    env.close()


if __name__ == '__main__':
    sys.exit(main())
