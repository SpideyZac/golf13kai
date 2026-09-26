"""Hero-shot reward shaping (`train.py --hero`), off by default.

The env's reward is -1 per stroke and -1 per penalty, so the return is minus
the score and the agent learns to minimise expected strokes: safe, percentage
golf. Hero mode adds a bounded bonus on top of it in the trainer. It rewards
holing out from far, finishing long shots close to the pin, and beating
birdie. The env itself is not touched, so the Rust/JS parity, the evals and
`best.json` selection all still use the true score.

Three terms, each multiplied by `--hero` (0 = off, 1 = the defaults below):

1. Hole-out from distance. A shot that drops from `from` yards earns
   `hole * ramp(from)`, where ramp goes linearly from 0 at `--hero-min` yards
   to 1 at `--hero-full` yards. Tap-ins earn nothing, and a holed 150-yard
   approach earns `hole` (0.6 of a stroke by default).
2. Near miss. A long shot (ramp > 0) that stops on dry ground within
   `--hero-radius` yards of the cup earns `near * ramp(from) * (1 - to/radius)`.
   Long hole-outs are rare, so this denser term is what teaches the agent to
   attack the pin and read the wind, not just to find the green.
3. Birdie is par. At the end of a hole played from the tee, every stroke
   under birdie earns `under` (0.5) more, and every stroke over birdie (par or
   worse) costs `over` (0.25) more. `under > over` makes the agent mildly
   risk-seeking around birdie (an eagle chance is worth a little risk), while
   `over > 0` makes a blow-up hurt more than it does without shaping, so the
   aggression stays targeted. Exploring starts drop the ball mid-hole, so their
   score to par means nothing and this term skips them.

Every bonus is under a stroke, so a gamble pays only when it holes often
enough: a hero line that costs an extra 0.3 strokes on average must hole
about half the time to earn back 0.6. That is the "aggressive, not reckless"
balance. Raise `--hero` to push harder, and watch the eval scores (the true
score) to see when it starts to cost strokes.
"""
from dataclasses import dataclass

import numpy as np

from . import sim


@dataclass
class HeroCfg:
    scale: float = 0.       # --hero: 0 = off
    hole: float = .6        # max bonus for a hole-out, in strokes
    near: float = .15       # max bonus for a long shot that stops next to the cup
    radius: float = 4.      # yards: the near-miss zone
    min_yd: float = 10.     # ramp start: holes from closer earn nothing
    full_yd: float = 150.   # ramp end: full bonus from here on
    under: float = .5       # per stroke better than birdie (tee starts)
    over: float = .25       # per stroke worse than birdie (tee starts)

    @property
    def on(self):
        return self.scale > 0

    @classmethod
    def from_args(cls, a):
        return cls(a.hero, a.hero_hole, a.hero_near, a.hero_radius, a.hero_min, a.hero_full, a.hero_under,
                   a.hero_over)


def add_args(ap):
    g = ap.add_argument_group('hero shaping (docs/TRAINING.md)')
    d = HeroCfg()
    g.add_argument('--hero', type=float, default=d.scale, help='hero-shot shaping scale (0: off, 1: defaults)')
    g.add_argument('--hero-hole', type=float, default=d.hole)
    g.add_argument('--hero-near', type=float, default=d.near)
    g.add_argument('--hero-radius', type=float, default=d.radius)
    g.add_argument('--hero-min', type=float, default=d.min_yd)
    g.add_argument('--hero-full', type=float, default=d.full_yd)
    g.add_argument('--hero-under', type=float, default=d.under)
    g.add_argument('--hero-over', type=float, default=d.over)


def ramp(c, d):
    return np.clip((d - c.min_yd) / max(c.full_yd - c.min_yd, 1e-9), 0., 1.)


def bonus(c, info):
    """The shaping bonus of every env for the stroke just played, from the
    (n, INFO_W) info rows gs_vec_step wrote. Zeros when hero mode is off."""
    n = len(info)
    if not c.on:
        return np.zeros(n)
    frm, to = info[:, sim.FROM], info[:, sim.TO]
    res = info[:, sim.RESULT]
    holed = res == sim.RESULTS.index('holed')
    stopped = res == sim.RESULTS.index('stopped')
    rp = ramp(c, frm)
    b = np.where(holed, c.hole * rp, 0.)
    b += np.where(stopped & (to < c.radius), c.near * rp * np.clip(1 - to / c.radius, 0., 1.), 0.)
    end = (info[:, sim.DONE] > 0) & (info[:, sim.TEE] > 0)
    vs_birdie = info[:, sim.STROKES] - info[:, sim.PAR] + 1   # < 0: eagle or better
    b += np.where(end, np.where(vs_birdie < 0, -c.under * vs_birdie, -c.over * vs_birdie), 0.)
    return c.scale * b


def is_long_holeout(info, yards=30.):
    """Holed on this stroke from at least `yards` (the log's hero counter)."""
    return (info[:, sim.RESULT] == sim.RESULTS.index('holed')) & (info[:, sim.FROM] >= yards)
