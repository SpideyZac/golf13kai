"""Upgrade a v3 checkpoint (before the solver and the impact action) to the current obs version.

The new observation inputs get zero weights and the actor a SOLVE logit with
zero weights and bias --solve-logit, and an impact head that swings straight
(mean 0, narrow std), so every old output is unchanged. At the
default -20 the agent never picks SOLVE and plays exactly as before, which is
how models/agent.json keeps working in the browser. To fine-tune into the
solver, pass a bias that lets it try SOLVE, or use `train.py --init`, which
upgrades on the fly with --init-solve-logit.

usage: python -m golfrl.upgrade ../models/agent.json [-o out.json] [--solve-logit -20]
"""
import argparse
import json
import sys
from pathlib import Path

from .model import load_js, save_js
from .sim import OBS_VERSION


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoint')
    ap.add_argument('-o', '--out', help='default: overwrite the input')
    ap.add_argument('--solve-logit', type=float, default=-20.)
    a = ap.parse_args(argv)
    j = json.loads(Path(a.checkpoint).read_text())
    model, meta = load_js(a.checkpoint, upgrade_solve_logit=a.solve_logit)
    out = Path(a.out or a.checkpoint)
    save_js(out, model, {**meta, 'upgradedFrom': j['obsVersion'], 'solveLogit': a.solve_logit})
    print(f"{a.checkpoint} (obs v{j['obsVersion']}) -> {out} (obs v{OBS_VERSION}, SOLVE bias {a.solve_logit:g})")


if __name__ == '__main__':
    sys.exit(main())
