"""Optional penalty shaping for the trainer (train.py), all off by default.

The env's reward is -1 per stroke and -1 per penalty, so the return is minus
the score. These terms add extra, trainer-side penalties on top of it:

- --sand-penalty X: every shot that comes to rest in a bunker costs X more
  (a failed splash that stays in the sand counts again).
- --tree-penalty X: every shot that strikes a tree costs X more.
- --target-score T --target-penalty X: a hole played from the tee that
  finishes worse than T to par (T = -2: anything worse than an eagle) costs X
  more, once, at the end. Exploring starts drop the ball mid-hole, so their
  score to par means nothing and they are skipped.

The env and its rules are untouched (so the Rust/JS parity is too), and the
evals, the log's toPar and best.json selection all use the true score, so the
shaping can only change what the agent tries, never how it is judged.
"""
from dataclasses import dataclass

import numpy as np

from . import sim

SAND = sim.LIES.index('SAND')


@dataclass
class ShapingCfg:
    sand: float = 0.
    tree: float = 0.
    target_score: float = -2.
    target: float = 0.

    @property
    def on(self):
        return bool(self.sand or self.tree or self.target)

    @classmethod
    def from_args(cls, a):
        return cls(a.sand_penalty, a.tree_penalty, a.target_score, a.target_penalty)

    def describe(self):
        parts = []
        if self.sand:
            parts.append(f'sand -{self.sand:g}')
        if self.tree:
            parts.append(f'tree -{self.tree:g}')
        if self.target:
            parts.append(f'worse than {self.target_score:+g} on a hole -{self.target:g}')
        return ', '.join(parts) or 'off'


def add_args(ap):
    g = ap.add_argument_group('penalty shaping (py/golfrl/shaping.py; 0 = off)')
    g.add_argument('--sand-penalty', type=float, default=0., help='extra penalty per shot that ends in a bunker')
    g.add_argument('--tree-penalty', type=float, default=0., help='extra penalty per shot that hits a tree')
    g.add_argument('--target-score', type=float, default=-2., help='hole score to par to reach (tee starts)')
    g.add_argument('--target-penalty', type=float, default=0., help='extra penalty for a hole worse than --target-score')


def terms(c, info):
    """Per env, for the stroke just played (the (n, INFO_W) info rows):
    (in sand, hit a tree, finished a tee hole worse than the target) flags."""
    sand = info[:, sim.REST] == SAND
    tree = info[:, sim.TREE] > 0
    miss = (info[:, sim.DONE] > 0) & (info[:, sim.TEE] > 0) & (info[:, sim.STROKES] - info[:, sim.PAR] > c.target_score)
    return sand, tree, miss


def penalty(c, info):
    """The extra (negative) reward of every env for the stroke just played."""
    if not c.on:
        return np.zeros(len(info))
    sand, tree, miss = terms(c, info)
    return -(c.sand * sand + c.tree * tree + c.target * miss)
