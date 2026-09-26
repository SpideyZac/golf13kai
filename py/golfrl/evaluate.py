"""Evaluate a checkpoint on full 18-hole rounds it never trained on (the
Python twin of rl/eval.mjs; the same courses, winds and rules).

usage: python -m golfrl.evaluate [models/agent.json] [--set classic|remix|both]
       [--rounds 8] [--from 4] [--stochastic | --escape] [--card] [--shots]
"""
import argparse
import sys

import numpy as np
import torch

from . import sim
from .model import load_js, sample

SCORE = ['albatross', 'eagle', 'birdie', 'par', 'bogey', 'double', 'triple+']


@torch.no_grad()
def play(model, specs, device='cpu', deterministic=True, escape=False, shots=False, threads=0, rules=None):
    """Play every spec (round, seed, remix, hole, rng_seed) to the end. Returns
    one dict per spec: round, hole, strokes, par, penalties, holed, holedFrom
    (yards of the stroke that went in, or None), solved (solver's shots
    played), solvedHoled (of them, holed) and the shots, if asked. rules: the
    env rules (sim.DEFAULT_RULES keys), e.g. a checkpoint's meta['env']."""
    n = len(specs)
    env = sim.VecEnv(n, threads=threads, **sim.rules_kwargs(rules))
    for i, (_, seed_, remix, hole, rng) in enumerate(specs):
        env.reset_spec(i, seed_, remix, hole, rng)
    done = np.zeros(n, bool)
    stuck = np.zeros(n, bool)
    res = [{'round': s[0], 'hole': s[3], 'shots': [], 'solved': 0, 'solvedHoled': 0} for s in specs]
    model.eval()
    while not done.all():
        obs = torch.from_numpy(env.obs).to(device)
        out, _ = model(obs)
        det = None
        if deterministic:
            det = torch.from_numpy(~stuck if escape else np.ones(n, bool)).to(device)
        club, spin, cont = sample(out, det)
        cont = cont.double().cpu().numpy()
        _, info = env.step(club.cpu().numpy(), spin.cpu().numpy(), cont[:, 0], cont[:, 1], cont[:, 2], active=~done,
                           auto_reset=False)
        for i in np.flatnonzero(~done):
            r = info[i]
            if r[sim.SOLVED]:
                res[i]['solved'] += 1
                res[i]['solvedHoled'] += r[sim.HOLED] > 0 and r[sim.DONE] > 0
            if shots:
                res[i]['shots'].append({
                    'club': sim.CLUB_NAMES[int(r[sim.CLUB])], 'spin': int(r[sim.SPIN]), 'lie': sim.LIES[int(r[sim.LIE])],
                    'from': r[sim.FROM], 'want': r[sim.WANT], 'result': sim.RESULTS[int(r[sim.RESULT])],
                    'to': r[sim.TO], 'tree': bool(r[sim.TREE]), 'solved': bool(r[sim.SOLVED]),
                    'impact': r[sim.IMPACT]})
            stuck[i] = bool(r[sim.STUCK])
            if r[sim.DONE]:
                done[i] = True
                res[i].update(strokes=int(r[sim.STROKES]), par=int(r[sim.PAR]), penalties=int(r[sim.PENALTIES]),
                              holed=bool(r[sim.HOLED]), holedFrom=float(r[sim.FROM]) if r[sim.HOLED] else None)
    env.close()
    return res


def summarise(results, long_yd=30.):
    """pool.mjs summarise: per-round means, plus hole-outs from long_yd+ yards."""
    rounds = {}
    for e in results:
        r = rounds.setdefault(e['round'], {'strokes': 0, 'par': 0, 'pen': 0, 'pickups': 0, 'long': 0, 'solved': 0,
                                           'solvedHoled': 0})
        r['strokes'] += e['strokes']
        r['par'] += e['par']
        r['pen'] += e['penalties']
        r['pickups'] += not e['holed']
        r['long'] += (e.get('holedFrom') or 0) >= long_yd
        r['solved'] += e.get('solved', 0)
        r['solvedHoled'] += e.get('solvedHoled', 0)
    rs = list(rounds.values())
    to_par = [r['strokes'] - r['par'] for r in rs]
    return {'rounds': len(rs), 'strokes': np.mean([r['strokes'] for r in rs]), 'toPar': float(np.mean(to_par)),
            'penalties': float(np.mean([r['pen'] for r in rs])), 'pickups': float(np.mean([r['pickups'] for r in rs])),
            'best': min(to_par), 'worst': max(to_par), 'longHoleOuts': float(np.mean([r['long'] for r in rs])),
            'solved': float(np.mean([r['solved'] for r in rs])),
            'solvedHoled': float(np.mean([r['solvedHoled'] for r in rs]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint', nargs='?', default=str(sim.ROOT / 'models' / 'agent.json'))
    ap.add_argument('--set', default='both', choices=['classic', 'remix', 'both'])
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--from', dest='start', type=int, default=4, help='skip the rounds training selected best.json on')
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--escape', action='store_true', help='sample instead of the mode after a stuck shot')
    ap.add_argument('--card', action='store_true', help='hole-by-hole card of the first round')
    ap.add_argument('--shots', action='store_true', help='every shot of the first round')
    # the rules default to the ones the checkpoint was trained under (meta env)
    ap.add_argument('--no-solver', dest='solver', action='store_const', const=False, help='play with the solver off')
    ap.add_argument('--exact-solve', dest='exact_solve', action='store_const', const=True,
                    help="the solver's shots skip the swing noise")
    ap.add_argument('--noisy-solve', dest='exact_solve', action='store_const', const=False,
                    help="the solver's shots get the swing noise")
    ap.add_argument('--impact-noise', type=float)
    ap.add_argument('--aim-noise', type=float)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=0)
    a = ap.parse_args(argv)

    model, meta = load_js(a.checkpoint)
    model.to(a.device)
    rules = {**sim.DEFAULT_RULES, **meta.get('env', {})}
    for k, v in (('solver', a.solver), ('exactSolve', a.exact_solve), ('impactNoise', a.impact_noise),
                 ('aimNoise', a.aim_noise)):
        if v is not None:
            rules[k] = v
    print(f"{a.checkpoint}  (iter {meta.get('iter')}, {meta.get('totalSteps')} steps)  rules {rules}")
    for name in (['classic', 'remix'] if a.set == 'both' else [a.set]):
        res = play(model, sim.eval_set(name, a.rounds, a.start), a.device, not a.stochastic, a.escape,
                   shots=a.shots, threads=a.threads, rules=rules)
        s = summarise(res)
        dist = np.zeros(len(SCORE))
        for e in res:
            dist[min(max(e['strokes'] - e['par'] + 3, 0), 6)] += 1
        print(f"{name}: {s['rounds']} rounds, {s['toPar']:+.2f} to par (mean {s['strokes']:.1f}, best {s['best']},"
              f" worst {s['worst']}), {s['penalties']:.1f} penalties, {s['pickups']:.2f} pickups,"
              f" {s['longHoleOuts']:.2f} hole-outs from 30+ yd per round")
        print(f"  solver's shot {s['solved']:.1f} times a round, holed {s['solvedHoled']:.1f}")
        print('  ' + '  '.join(f'{n} {d / len(res) * 100:.1f}%' for n, d in zip(SCORE, dist)))
        if a.card or a.shots:
            for e in (e for e in res if e['round'] == a.start):
                pen = f" ({e['penalties']} pen)" if e['penalties'] else ''
                print(f"  hole {e['hole'] + 1:2d} par {e['par']}: {e['strokes']}{pen}")
                for sh in e['shots'] if a.shots else []:
                    spin = ['back', '    ', 'top '][sh['spin'] + 1]
                    shape = f"  impact {sh['impact']:+.2f}" if abs(sh['impact']) >= .02 else ''
                    print(f"      {sh['club']:<3} {spin} from {sh['lie']:<7} {sh['from']:6.1f}yd  asked {sh['want']:6.1f}"
                          f" -> {sh['result']:<7} {sh['to']:6.1f}yd left{'  TREE' if sh['tree'] else ''}"
                          f"{'  SOLVE' if sh['solved'] else ''}{shape}")


if __name__ == '__main__':
    sys.exit(main())
