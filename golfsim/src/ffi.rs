//! C ABI for the Python trainer (py/golf/sim.py loads it with ctypes): a batch
//! of envs stepped in parallel on a rayon pool. Every array is caller-owned
//! and row-major; `obs` is n x OBS_DIM float32 and `info` is n x INFO_W
//! float64 (columns I_*).
//!
//! Training: gs_vec_reset_train, then gs_vec_step with auto_reset = 1. A
//! finished env reports its episode in `info` and its `obs` row already holds
//! the first observation of a fresh training episode (courses.mjs
//! trainEpisode, drawn from the env's own RNG).
//! Evaluation: gs_vec_reset_spec each env, then gs_vec_step with auto_reset
//! = 0 and `active` masking out the envs that are done.

use crate::env::*;
use crate::jsmath::Mulberry32;
use crate::obs::{is_stuck, Action, N_CLUB_ACTIONS, OBS_DIM, OBS_VERSION};
use rayon::prelude::*;

pub const INFO_W: usize = 20;
pub const I_REWARD: usize = 0;
pub const I_DONE: usize = 1;
pub const I_RESULT: usize = 2; // RES_*: holed, stopped, water, ob
pub const I_STROKES: usize = 3;
pub const I_PAR: usize = 4;
pub const I_PENALTIES: usize = 5;
pub const I_HOLED: usize = 6; // episode ended in the cup (else picked up)
pub const I_STUCK: usize = 7; // the escape rule applies to the NEXT shot
pub const I_CLUB: usize = 8; // the shot just played, as env.log records it
pub const I_SPIN: usize = 9;
pub const I_LIE: usize = 10;
pub const I_FROM: usize = 11;
pub const I_WANT: usize = 12;
pub const I_POWER: usize = 13;
pub const I_TO: usize = 14;
pub const I_TREE: usize = 15;
pub const I_SOLVED: usize = 16; // the shot played was the solver's (CLUB_SOLVE with a solution)
pub const I_IMPACT: usize = 17; // the impact swung for (before the swing noise)
pub const I_REST: usize = 18; // SURF_* the ball came to rest on (after any penalty drop)
pub const I_TEE: usize = 19; // the episode started on the tee (not an exploring start)

struct Slot {
    env: GolfEnv,
    rng: Mulberry32,
}

pub struct VecEnv {
    slots: Vec<Slot>,
    pool: rayon::ThreadPool,
    classic_prob: f64,
    start_prob: f64,
}

impl Slot {
    fn reset_train(&mut self, classic_prob: f64, start_prob: f64, obs: &mut [f32]) {
        let rng = &mut self.rng;
        let spec = train_episode(&mut || rng.next(), classic_prob, start_prob);
        self.env.reset(spec);
        self.env.observe(obs);
    }

    fn step(&mut self, a: &Action, auto_reset: bool, cp: f64, sp: f64, obs: &mut [f32], info: &mut [f64]) {
        let r = self.env.step(a);
        let e = &self.env;
        let l = *e.log.last().unwrap();
        info[I_REWARD] = r.reward;
        info[I_DONE] = r.done as u8 as f64;
        info[I_RESULT] = r.result as f64;
        info[I_STROKES] = e.strokes as f64;
        info[I_PAR] = e.g.hole.par as f64;
        info[I_PENALTIES] = e.penalties as f64;
        info[I_HOLED] = (r.result == RES_HOLED) as u8 as f64;
        info[I_STUCK] = 0.0;
        info[I_CLUB] = l.club as f64;
        info[I_SPIN] = l.spin;
        info[I_LIE] = l.lie as f64;
        info[I_FROM] = l.from;
        info[I_WANT] = l.want;
        info[I_POWER] = l.power;
        info[I_TO] = l.to;
        info[I_TREE] = l.tree as u8 as f64;
        info[I_SOLVED] = l.solved as u8 as f64;
        info[I_IMPACT] = l.impact;
        info[I_REST] = e.g.ball_ground().s as f64;
        info[I_TEE] = e.spec.start.is_none() as u8 as f64;
        if !r.done {
            info[I_STUCK] = is_stuck(&e.prev, l.club == crate::game::CLUB_PUTTER) as u8 as f64;
            self.env.observe(obs);
        } else if auto_reset {
            self.reset_train(cp, sp, obs);
        }
    }
}

#[no_mangle]
pub extern "C" fn gs_obs_version() -> u32 {
    OBS_VERSION
}

#[no_mangle]
pub extern "C" fn gs_obs_dim() -> u32 {
    OBS_DIM as u32
}

#[no_mangle]
pub extern "C" fn gs_info_width() -> u32 {
    INFO_W as u32
}

/// A batch of n envs on `threads` worker threads (0 = one per core). `seed`
/// seeds each env's training-episode RNG. `solver` = 0 turns the solver off
/// (its obs block is zeros and CLUB_SOLVE plays the game's club), and
/// `exact_solve` = 1 plays the solver's shots without swing noise.
#[no_mangle]
pub extern "C" fn gs_vec_new(
    n: u32,
    threads: u32,
    impact_noise: f64,
    aim_noise: f64,
    max_over: i32,
    seed: u32,
    classic_prob: f64,
    start_prob: f64,
    solver: u8,
    exact_solve: u8,
) -> *mut VecEnv {
    let cfg = EnvCfg {
        impact_noise,
        aim_noise,
        max_over,
        solver: solver != 0,
        exact_solve: exact_solve != 0,
    };
    let slots = (0..n)
        .map(|i| Slot {
            env: GolfEnv::new(cfg),
            rng: Mulberry32::new(
                seed.wrapping_mul(0x9E3779B1)
                    .wrapping_add(i.wrapping_mul(0x85EBCA77))
                    .wrapping_add(1),
            ),
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads as usize)
        .build()
        .expect("thread pool");
    Box::into_raw(Box::new(VecEnv {
        slots,
        pool,
        classic_prob,
        start_prob,
    }))
}

/// # Safety
/// `h` must come from gs_vec_new and not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn gs_vec_free(h: *mut VecEnv) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

/// Resets every env to a fresh training episode; writes n x OBS_DIM obs.
///
/// # Safety
/// `h` from gs_vec_new; `obs` holds n * OBS_DIM floats.
#[no_mangle]
pub unsafe extern "C" fn gs_vec_reset_train(h: *mut VecEnv, obs: *mut f32) {
    let v = &mut *h;
    let n = v.slots.len();
    let obs = std::slice::from_raw_parts_mut(obs, n * OBS_DIM);
    let (cp, sp) = (v.classic_prob, v.start_prob);
    v.pool.install(|| {
        v.slots
            .par_iter_mut()
            .zip(obs.par_chunks_mut(OBS_DIM))
            .for_each(|(s, o)| s.reset_train(cp, sp, o));
    });
}

/// Resets env i to one hole (has_start: use the exploring start u, v) and
/// writes its OBS_DIM obs.
///
/// # Safety
/// `h` from gs_vec_new, i < n; `obs` holds OBS_DIM floats.
#[no_mangle]
pub unsafe extern "C" fn gs_vec_reset_spec(
    h: *mut VecEnv,
    i: u32,
    seed: f64,
    remix: u8,
    hole: u32,
    rng_seed: u32,
    has_start: u8,
    u: f64,
    v: f64,
    obs: *mut f32,
) {
    let ve = &mut *h;
    let s = &mut ve.slots[i as usize];
    let spec = Spec {
        seed,
        remix: remix != 0,
        hole: (hole as usize).min(17),
        rng_seed,
        start: if has_start != 0 { Some((u, v)) } else { None },
    };
    s.env.reset(spec);
    s.env.observe(std::slice::from_raw_parts_mut(obs, OBS_DIM));
}

/// One stroke in every active env (`active` may be null: all). Actions are
/// n-long arrays (club, spin, aim, dist, impact). Writes each active env's obs row (the next state, or with
/// auto_reset the first state of a new episode when it finished) and info row.
///
/// # Safety
/// `h` from gs_vec_new; every array sized for n envs as described above.
#[no_mangle]
pub unsafe extern "C" fn gs_vec_step(
    h: *mut VecEnv,
    club: *const i32,
    spin: *const i32,
    aim: *const f64,
    dist: *const f64,
    impact: *const f64,
    active: *const u8,
    auto_reset: u8,
    obs: *mut f32,
    info: *mut f64,
) {
    let v = &mut *h;
    let n = v.slots.len();
    let club = std::slice::from_raw_parts(club, n);
    let spin = std::slice::from_raw_parts(spin, n);
    let aim = std::slice::from_raw_parts(aim, n);
    let dist = std::slice::from_raw_parts(dist, n);
    let impact = std::slice::from_raw_parts(impact, n);
    let active = if active.is_null() {
        None
    } else {
        Some(std::slice::from_raw_parts(active, n))
    };
    let obs = std::slice::from_raw_parts_mut(obs, n * OBS_DIM);
    let info = std::slice::from_raw_parts_mut(info, n * INFO_W);
    let (cp, sp, ar) = (v.classic_prob, v.start_prob, auto_reset != 0);
    v.pool.install(|| {
        v.slots
            .par_iter_mut()
            .zip(obs.par_chunks_mut(OBS_DIM))
            .zip(info.par_chunks_mut(INFO_W))
            .enumerate()
            .for_each(|(i, ((s, o), inf))| {
                if active.is_some_and(|a| a[i] == 0) || s.env.done {
                    return;
                }
                let a = Action {
                    club: club[i].clamp(0, N_CLUB_ACTIONS as i32 - 1) as usize,
                    spin: spin[i].clamp(0, 2) as usize,
                    // a NaN would poison the ball (and never finish a drop)
                    aim: if aim[i].is_finite() { aim[i] } else { 0.0 },
                    dist: if dist[i].is_finite() { dist[i] } else { 0.0 },
                    impact: if impact[i].is_finite() { impact[i] } else { 0.0 },
                };
                s.step(&a, ar, cp, sp, o, inf);
            });
    });
}
