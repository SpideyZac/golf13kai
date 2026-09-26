"""The agent in PyTorch: the same networks and action distribution as
rl/policy.mjs, so that a checkpoint trained here plays in the browser.

Actor output (HEAD = 21): club logits [0:12] (the 11 clubs, then CLUB_SOLVE,
the solver's shot), spin logits [12:15], Gaussian means of (aim, dist,
impact) [15:18], their log stds [18:21] clamped to [LS_MIN, LS_MAX]. The
critic is a separate MLP. Hidden layers are tanh.

A v3 checkpoint (before the solver and the impact) upgrades to v5 losslessly:
zero weights for the new observation inputs, a SOLVE logit with zero weights
and a chosen bias, and an impact head with zero weights (mean 0: a straight
swing) and a narrow std (upgrade_v3, and python -m golfrl.upgrade).

Checkpoints use the JS format (rl/checkpoint.mjs): the parameters as ONE flat
float32 array in the JS layout - per layer W as [in][out] then b, actor first,
then critic - base64 encoded.
"""
import base64
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sim import OBS_DIM, OBS_VERSION, N_CLUB_ACTIONS as N_CLUBS, CLUB_SOLVE, N_SPIN

N_CONT = 3      # the continuous action: aim, dist, impact
HEAD = N_CLUBS + N_SPIN + 2 * N_CONT
C0, S0, MU, LS = 0, N_CLUBS, N_CLUBS + N_SPIN, N_CLUBS + N_SPIN + N_CONT
LS_MIN, LS_MAX = -6., 1.
LS_INIT = -.7   # initial log std of (aim, dist, impact)
# the upgraded impact head's log std: raw std .37, so impact std ~.018, which
# keeps most of the old agent's swings inside the game's perfect-strike snap
UPGRADE_IMPACT_LS = -1.
V_BIAS = -4.    # critic output offset: returns are about -4 strokes a hole
LOG2PI = math.log(2 * math.pi)
DEFAULT_HIDDEN = [256, 128]


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i + 2 < len(sizes):
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Model(nn.Module):
    def __init__(self, hidden=DEFAULT_HIDDEN, obs_dim=OBS_DIM):
        super().__init__()
        self.hidden = list(hidden)
        self.obs_dim = obs_dim
        self.actor = mlp([obs_dim, *hidden, HEAD])
        self.critic = mlp([obs_dim, *hidden, 1])

    @property
    def arch(self):
        return {'hidden': self.hidden}

    def linears(self):
        return [m for m in self.actor if isinstance(m, nn.Linear)] + [m for m in self.critic if isinstance(m, nn.Linear)]

    @torch.no_grad()
    def init_like_js(self, seed=1):
        """MLP.init + Model.init: N(0, gain/sqrt(fan_in)) weights, zero biases;
        output gains .01 (actor) and 1 (critic); log std and value offsets."""
        g = torch.Generator().manual_seed(seed)
        for net, out_gain in ((self.actor, .01), (self.critic, 1.)):
            lin = [m for m in net if isinstance(m, nn.Linear)]
            for i, m in enumerate(lin):
                gain = out_gain if i == len(lin) - 1 else 1.
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * gain / math.sqrt(m.in_features))
                m.bias.zero_()
        self.actor[-1].bias[LS:LS + N_CONT] = LS_INIT
        self.critic[-1].bias[0] = V_BIAS
        return self

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    # ---- JS checkpoint layout ----

    def flat_params(self):
        """The JS flat parameter array (float32, numpy)."""
        parts = []
        for m in self.linears():
            parts.append(m.weight.detach().float().t().contiguous().cpu().numpy().ravel())
            parts.append(m.bias.detach().float().cpu().numpy())
        return np.concatenate(parts).astype('<f4')

    @torch.no_grad()
    def load_flat(self, flat):
        flat = np.asarray(flat, np.float32)
        o = 0
        for m in self.linears():
            n, k = m.in_features, m.out_features
            m.weight.copy_(torch.from_numpy(flat[o:o + n * k].reshape(n, k).T.copy()))
            o += n * k
            m.bias.copy_(torch.from_numpy(flat[o:o + k].copy()))
            o += k
        if o != flat.size:
            raise ValueError(f'checkpoint has {flat.size} params, the model {o}')
        return self


def save_js(path, model, meta):
    """Write a checkpoint rl/checkpoint.mjs (and the browser) can load."""
    params = base64.b64encode(model.flat_params().tobytes()).decode('ascii')
    j = {'format': 1, 'obsVersion': OBS_VERSION, 'obsDim': OBS_DIM, 'arch': model.arch, 'meta': meta, 'params': params}
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(j))
    tmp.replace(path)


# v3 (before the solver): 6 fewer observation floats (the solver block is the
# last one), no SOLVE logit and no impact head
V3_OBS_DIM, V3_HEAD = OBS_DIM - 6, HEAD - 3


def upgrade_v3(flat, hidden, solve_logit):
    """A v3 flat parameter array as v5: the new inputs get zero weights in
    both first layers, and the actor gets a SOLVE logit at CLUB_SOLVE with zero
    weights and bias `solve_logit`, and an impact mean (zero weights and bias:
    a straight swing) and log std (bias UPGRADE_IMPACT_LS). Every old output
    is unchanged, so with SOLVE at -20 the upgraded agent's mode plays as
    before (its sampled swings add a small impact spread); a higher bias makes
    it try the solver's shot from the start."""
    flat = np.asarray(flat, np.float32)
    out, o = [], 0
    for net_out in (V3_HEAD, 1):
        sizes = [V3_OBS_DIM, *hidden, net_out]
        for li in range(len(sizes) - 1):
            n, k = sizes[li], sizes[li + 1]
            W = flat[o:o + n * k].reshape(n, k)
            o += n * k
            b = flat[o:o + k]
            o += k
            if li == 0:
                W = np.concatenate([W, np.zeros((OBS_DIM - V3_OBS_DIM, k), np.float32)])
            if net_out == V3_HEAD and li == len(sizes) - 2:
                # SOLVE, then the impact's mean after (aim, dist), then its log std
                for at, bias in ((CLUB_SOLVE, solve_logit), (MU + 2, 0.), (LS + 2, UPGRADE_IMPACT_LS)):
                    W = np.insert(W, at, 0., axis=1)
                    b = np.insert(b, at, bias)
            out += [W.ravel(), b]
    if o != flat.size:
        raise ValueError(f'v3 checkpoint has {flat.size} params, expected {o}')
    return np.concatenate(out).astype(np.float32)


def load_js(path, upgrade_solve_logit=None):
    """(model, meta) from a JS checkpoint. With upgrade_solve_logit set, a v3
    checkpoint is upgraded to v4 (see upgrade_v3) instead of refused."""
    j = json.loads(Path(path).read_text())
    flat = np.frombuffer(base64.b64decode(j['params']), '<f4')
    if j['obsVersion'] == 3 and j['obsDim'] == V3_OBS_DIM and upgrade_solve_logit is not None:
        flat = upgrade_v3(flat, j['arch']['hidden'], upgrade_solve_logit)
    elif j['obsVersion'] != OBS_VERSION or j['obsDim'] != OBS_DIM:
        raise ValueError(f"{path}: trained on obs v{j['obsVersion']} ({j['obsDim']}), env is v{OBS_VERSION} ({OBS_DIM})")
    model = Model(j['arch']['hidden'])
    model.load_flat(flat)
    return model, j.get('meta', {})


# ---- the hybrid action distribution (rl/policy.mjs) ----

def split(out):
    ls = out[:, LS:LS + N_CONT].clamp(LS_MIN, LS_MAX)
    return out[:, C0:C0 + N_CLUBS], out[:, S0:S0 + N_SPIN], out[:, MU:MU + N_CONT], ls


def sample(out, deterministic=None):
    """Actions (club, spin, cont[:, (aim, dist, impact)]) for a batch of actor rows.
    deterministic: None (sample all), True (all modes), or a bool mask of the
    rows to take the mode for."""
    club_l, spin_l, mu, ls = split(out)
    club = torch.distributions.Categorical(logits=club_l).sample()
    spin = torch.distributions.Categorical(logits=spin_l).sample()
    cont = mu + ls.exp() * torch.randn_like(mu)
    if deterministic is not None:
        det = deterministic if torch.is_tensor(deterministic) else torch.full_like(club, bool(deterministic), dtype=torch.bool)
        club = torch.where(det, club_l.argmax(-1), club)
        spin = torch.where(det, spin_l.argmax(-1), spin)
        cont = torch.where(det[:, None], mu, cont)
    return club, spin, cont


def log_prob(out, club, spin, cont):
    club_l, spin_l, mu, ls = split(out)
    lp = torch.log_softmax(club_l, -1).gather(-1, club[:, None]).squeeze(-1)
    lp = lp + torch.log_softmax(spin_l, -1).gather(-1, spin[:, None]).squeeze(-1)
    z = (cont - mu) / ls.exp()
    return lp + (-.5 * z * z - ls - .5 * LOG2PI).sum(-1)


def entropies(out):
    """(categorical entropy, Gaussian entropy, log stds) per row."""
    club_l, spin_l, _, ls = split(out)
    h = 0
    for logits in (club_l, spin_l):
        lp = torch.log_softmax(logits, -1)
        h = h - (lp.exp() * lp).sum(-1)
    return h, (ls + .5 + .5 * LOG2PI).sum(-1), ls
