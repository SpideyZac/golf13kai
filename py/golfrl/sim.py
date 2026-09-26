"""The Rust environment (golfsim/src/ffi.rs) through ctypes.

Build it first: cargo build --release --manifest-path golfsim/Cargo.toml
(or set GOLFSIM_LIB to the library's path).
"""
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# info columns (ffi.rs I_*)
REWARD, DONE, RESULT, STROKES, PAR, PENALTIES, HOLED, STUCK = range(8)
CLUB, SPIN, LIE, FROM, WANT, POWER, TO, TREE = range(8, 16)
SOLVED = 16  # the shot played was the solver's (CLUB_SOLVE with a solution)
IMPACT = 17  # the meter impact swung for (+ early, - late), before the swing noise
RESULTS = ['holed', 'stopped', 'water', 'ob']
LIES = ['ROUGH', 'FAIRWAY', 'GREEN', 'TEE', 'SAND', 'WATER', 'OB']
CLUB_NAMES = ['1W', '3W', '5W', '3i', '5i', '7i', '9i', '13i', 'PW', 'SW', 'PT']
N_CLUBS, N_SPIN = 11, 3
CLUB_SOLVE = N_CLUBS            # the club head's extra choice: the solver's shot
N_CLUB_ACTIONS = N_CLUBS + 1
CLASSIC_SEED = 1113


def _lib_path():
    if os.environ.get('GOLFSIM_LIB'):
        return Path(os.environ['GOLFSIM_LIB'])
    name = {'win32': 'golfsim.dll', 'darwin': 'libgolfsim.dylib'}.get(sys.platform, 'libgolfsim.so')
    return ROOT / 'golfsim' / 'target' / 'release' / name


def _load():
    path = _lib_path()
    if not path.exists():
        raise FileNotFoundError(f'{path} not found: cargo build --release --manifest-path golfsim/Cargo.toml')
    lib = ctypes.CDLL(str(path))
    P, F, D, U, I, U8 = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint8
    lib.gs_obs_version.restype = U
    lib.gs_obs_dim.restype = U
    lib.gs_info_width.restype = U
    lib.gs_vec_new.restype = P
    lib.gs_vec_new.argtypes = [U, U, D, D, I, U, D, D, U8, U8]
    lib.gs_vec_free.argtypes = [P]
    lib.gs_vec_reset_train.argtypes = [P, F]
    lib.gs_vec_reset_spec.argtypes = [P, U, D, U8, U, U, U8, D, D, F]
    lib.gs_vec_step.argtypes = [P, F, F, F, F, F, F, U8, F, F]
    return lib


_lib = _load()
OBS_VERSION = _lib.gs_obs_version()
OBS_DIM = _lib.gs_obs_dim()
INFO_W = _lib.gs_info_width()
if INFO_W <= IMPACT:
    raise ImportError(f'{_lib_path()} is out of date (info width {INFO_W}): npm run build:sim')


def _ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class VecEnv:
    """n golf envs stepped in parallel (one stroke each per step).

    obs is an (n, OBS_DIM) float32 array and info an (n, INFO_W) float64 array,
    both owned here and overwritten in place by every call.
    """

    def __init__(self, n, threads=0, impact_noise=.04, aim_noise=.015, max_over=5, seed=1,
                 classic_prob=0., start_prob=.3, solver=True, exact_solve=False):
        self.n = n
        self._h = _lib.gs_vec_new(n, threads, impact_noise, aim_noise, max_over, seed & 0xFFFFFFFF,
                                  classic_prob, start_prob, int(bool(solver)), int(bool(exact_solve)))
        self.obs = np.zeros((n, OBS_DIM), np.float32)
        self.info = np.zeros((n, INFO_W), np.float64)

    def close(self):
        if self._h:
            _lib.gs_vec_free(self._h)
            self._h = None

    def __del__(self):
        self.close()

    def reset_train(self):
        """Every env to a fresh training episode (courses.mjs trainEpisode)."""
        _lib.gs_vec_reset_train(self._h, _ptr(self.obs))
        return self.obs

    def reset_spec(self, i, seed, remix, hole, rng_seed, start=None):
        """Env i to one hole; start = (u, v) for an exploring start."""
        u, v = start if start else (0., 0.)
        row = self.obs[i]
        _lib.gs_vec_reset_spec(self._h, i, float(seed), int(remix), hole, rng_seed, start is not None, u, v, _ptr(row))

    def step(self, club, spin, aim, dist, impact=None, active=None, auto_reset=True):
        """One stroke in every (active) env; impact (raw, like aim and dist)
        defaults to straight swings. With auto_reset, a finished env's obs row
        is already its next episode's first observation."""
        club = np.ascontiguousarray(club, np.int32)
        spin = np.ascontiguousarray(spin, np.int32)
        aim = np.ascontiguousarray(aim, np.float64)
        dist = np.ascontiguousarray(dist, np.float64)
        impact = np.zeros(self.n) if impact is None else np.ascontiguousarray(impact, np.float64)
        act = None if active is None else _ptr(np.ascontiguousarray(active, np.uint8))
        _lib.gs_vec_step(self._h, _ptr(club), _ptr(spin), _ptr(aim), _ptr(dist), _ptr(impact), act, int(auto_reset),
                         _ptr(self.obs), _ptr(self.info))
        return self.obs, self.info


# The env rules a checkpoint was trained under, as train.py records them in its
# meta ('env'); evaluate.py and the browser play by the same rules.
DEFAULT_RULES = {'impactNoise': .04, 'aimNoise': .015, 'solver': True, 'exactSolve': False}


def rules_kwargs(rules):
    """VecEnv keyword arguments for a rules dict (missing keys: defaults)."""
    r = {**DEFAULT_RULES, **(rules or {})}
    return {'impact_noise': r['impactNoise'], 'aim_noise': r['aimNoise'], 'solver': r['solver'],
            'exact_solve': r['exactSolve']}


def eval_set(name, rounds, start=0):
    """courses.mjs evalSet: [(round, seed, remix, hole, rng_seed)] for 18-hole
    rounds start..start+rounds of 'classic' or 'remix'."""
    specs = []
    for r in range(start, start + rounds):
        for h in range(18):
            rng = 1000 * (r + 1) + h
            specs.append((r, CLASSIC_SEED, False, h, rng) if name == 'classic' else (r, r + 1, True, h, rng))
    return specs
