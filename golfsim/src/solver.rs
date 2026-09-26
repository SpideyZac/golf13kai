//! The solver: a port of rl/solver.mjs, and it must agree with it bit for bit
//! (its results are in the observation). When the pin is in range, it steers
//! trial swings through the real physics until one holes, then replays that
//! swing across the swing-meter noise. Spec: docs/SPEC.md (Solver).

use crate::game::*;
use crate::jsmath::{self as jm, hypot2, js_max, js_min};

/// trial shots per candidate while steering
pub const SOLVE_ITERS: usize = 8;
/// ball-flight frames per trial: the env's own limit
const SIM_FRAMES: i32 = 60 * 30;
/// impact errors under this snap to a perfect strike (launchBall)
const PERFECT: f64 = 0.02;
/// the sand wedge: the chip candidate when the game hands over the putter off the green
const SW: usize = 9;
/// the fallback searches this many clubs either side of the game's
const FALLBACK_CLUBS: usize = 2;
/// the fallback's shaped swings: the impact swung for (+ early, - late)
const CURVES: [f64; 4] = [-0.12, -0.06, 0.06, 0.12];

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = 2.0 * std::f64::consts::PI;

/// the impact launchBall actually plays
#[inline]
fn snap(e: f64) -> f64 {
    if e.abs() < PERFECT {
        0.0
    } else {
        e
    }
}

/// Solver.solve's result. `spin` is the launch spin (-1 back, 0 none, 1 top),
/// `impact` the meter impact swung for.
#[derive(Clone, Copy, Debug, Default)]
pub struct Solution {
    pub tried: bool,
    pub found: bool,
    pub club: usize,
    pub spin: f64,
    pub impact: f64,
    pub yaw: f64,
    pub power: f64,
    pub lm: f64,
    pub want: f64,
    pub p_hole: f64,
    pub p_hazard: f64,
    pub leave: f64,
}

struct Trial {
    ev: u8,
    x: f64,
    z: f64,
}

/// One trial swing from where the ball lies, as GolfEnv::step plays it.
#[allow(clippy::too_many_arguments)]
fn trial(g: &mut Game, club: usize, spin: f64, yaw: f64, power: f64, lm: f64, impact: f64, aim_err: f64, pin_out: bool) -> Trial {
    let s = g.save_flight();
    g.pin_out = pin_out;
    g.tree_hit = false;
    g.launch_ball(club, power, impact, spin, yaw + aim_err, lm);
    let mut t = 0;
    while t < SIM_FRAMES && g.ball_event == 0 {
        g.ball_update();
        t += 1;
    }
    let r = Trial {
        ev: if g.ball_event != 0 { g.ball_event } else { EV_STOPPED },
        x: g.ball.x,
        z: g.ball.z,
    };
    g.restore_flight(&s);
    r
}

/// quarter centres of a uniform +-n, or a single 0 sample when n <= 0
fn quarters(n: f64) -> Vec<(f64, f64)> {
    if n > 0.0 {
        vec![(-n * 0.75, 0.25), (-n * 0.25, 0.25), (n * 0.25, 0.25), (n * 0.75, 0.25)]
    } else {
        vec![(0.0, 1.0)]
    }
}

/// noiseSamples: (impact error, aim error, weight)
fn noise_samples(impact_noise: f64, aim_noise: f64, putt: bool) -> Vec<(f64, f64, f64)> {
    let imp = quarters(impact_noise);
    let aim = if putt { vec![(0.0, 1.0)] } else { quarters(aim_noise) };
    let mut out = Vec::with_capacity(imp.len() * aim.len());
    for &(i, wi) in &imp {
        for &(a, wa) in &aim {
            out.push((i, a, wi * wa));
        }
    }
    out
}

/// Solver.candidates: (club, spin, impact) swings worth solving from here.
fn candidates(auto: usize, lie: u8, d: f64) -> Vec<(usize, f64, f64)> {
    let mut c = Vec::with_capacity(3);
    if d < PUTT_MAX && (lie == SURF_GREEN || lie == SURF_FAIRWAY || lie == SURF_TEE) {
        c.push((CLUB_PUTTER, 0.0, 0.0));
    }
    let full = if auto == CLUB_PUTTER {
        if lie == SURF_GREEN {
            None
        } else {
            Some(SW)
        }
    } else {
        Some(auto)
    };
    if let Some(f) = full {
        c.push((f, 0.0, 0.0));
        c.push((f, -1.0, 0.0));
    }
    c
}

/// Solver.fallback: when no candidate holes, the clubs up to FALLBACK_CLUBS
/// either side of the game's, longest first: straight with no spin, backspin
/// and topspin, then hooked and sliced (CURVES) with no spin and backspin.
fn fallback(auto: usize, tried: &[(usize, f64, f64)]) -> Vec<(usize, f64, f64)> {
    let base = if auto == CLUB_PUTTER { SW } else { auto };
    let (lo, hi) = (base.saturating_sub(FALLBACK_CLUBS), (base + FALLBACK_CLUBS).min(SW));
    let mut c = Vec::new();
    let mut add = |k: usize, spin: f64, imp: f64| {
        if !tried.iter().any(|&(tc, ts, ti)| tc == k && ts == spin && ti == imp) {
            c.push((k, spin, imp));
        }
    };
    for k in lo..=hi {
        for spin in [0.0, -1.0, 1.0] {
            add(k, spin, 0.0);
        }
    }
    for k in lo..=hi {
        for spin in [0.0, -1.0] {
            for imp in CURVES {
                add(k, spin, imp);
            }
        }
    }
    c
}

/// Solver.steer: (yaw, power, want) of a swing of (club, spin, impact) that holes.
#[allow(clippy::too_many_arguments)]
fn steer(g: &mut Game, club: usize, spin: f64, impact: f64, d: f64, pdir: f64, pin_out: bool) -> Option<(f64, f64, f64)> {
    let (bx, bz) = (g.ball.x, g.ball.z);
    let putt = club == CLUB_PUTTER;
    let lm = g.lie_mul(club);
    let max = if putt { PUTT_MAX } else { CLUBS[club].1 * lm };
    let lo = if putt { 0.005 } else { 0.02 };
    let (mut yaw, mut want) = (pdir, d);
    for _ in 0..SOLVE_ITERS {
        let power = js_min(js_max(want / max, lo), 1.0);
        let r = trial(g, club, spin, yaw, power, lm, impact, 0.0, pin_out);
        if r.ev == EV_HOLED {
            return Some((yaw, power, want));
        }
        let dx = r.x - bx;
        let dz = r.z - bz;
        let got = hypot2(dx, dz);
        if got < 0.5 || (power == 1.0 && got < d) {
            return None;
        }
        let mut turn = pdir - jm::atan2(dx, dz);
        if turn > PI {
            turn -= TAU;
        } else if turn < -PI {
            turn += TAU;
        }
        yaw += turn;
        want *= d / got;
    }
    None
}

/// Solver.solve: the best holing swing from the current lie, if any.
pub fn solve(g: &mut Game, impact_noise: f64, aim_noise: f64) -> Solution {
    let (bx, bz) = (g.ball.x, g.ball.z);
    let (px, pz) = (g.hole.pin.x, g.hole.pin.z);
    let d = hypot2(px - bx, pz - bz);
    let pdir = jm::atan2(px - bx, pz - bz);
    let auto = g.auto_club();
    let reach = if auto == CLUB_PUTTER {
        PUTT_MAX
    } else {
        CLUBS[auto].1 * g.lie_mul(auto)
    };
    // in range: the test that makes the game aim its default shot at the pin
    if !(d < reach + 20.0) {
        return Solution::default();
    }
    let lie = g.ground_at(bx, bz).s;
    let pin_out = d < 15.0 && lie == SURF_GREEN;
    let mut best = Solution {
        tried: true,
        ..Default::default()
    };
    let cands = candidates(auto, lie, d);
    let fb = fallback(auto, &cands);
    // every candidate is solved and the best odds win; the fallback stops at
    // its first solution
    for (pass, (club, spin, impact)) in cands.iter().map(|&c| (0, c)).chain(fb.into_iter().map(|c| (1, c))) {
        if pass == 1 && best.found {
            break;
        }
        let putt = club == CLUB_PUTTER;
        let lm = g.lie_mul(club);
        let Some((hy, hp, hw)) = steer(g, club, spin, impact, d, pdir, pin_out) else { continue };
        let (mut p_hole, mut p_hazard, mut leave) = (0.0, 0.0, 0.0);
        // samples that snap to the same strike play the same shot: one trial each
        let mut seen: Vec<(f64, f64, u8, f64, f64)> = Vec::new();
        for (imp, aim, w) in noise_samples(impact_noise, aim_noise, putt) {
            // the swing that plays exactly as the solution drops
            if snap(impact + imp) == snap(impact) && aim == 0.0 {
                p_hole += w;
                continue;
            }
            let e = snap(impact + imp);
            let r = match seen.iter().find(|q| q.0 == e && q.1 == aim) {
                Some(q) => Trial { ev: q.2, x: q.3, z: q.4 },
                None => {
                    let r = trial(g, club, spin, hy, hp, lm, impact + imp, aim, pin_out);
                    seen.push((e, aim, r.ev, r.x, r.z));
                    r
                }
            };
            if r.ev == EV_HOLED {
                p_hole += w;
            } else {
                if r.ev == EV_WATER || r.ev == EV_OB {
                    p_hazard += w;
                }
                leave += w * hypot2(px - r.x, pz - r.z);
            }
        }
        if !best.found || p_hole > best.p_hole || (p_hole == best.p_hole && leave < best.leave) {
            best = Solution {
                tried: true,
                found: true,
                club,
                spin,
                impact,
                yaw: hy,
                power: hp,
                lm,
                want: hw,
                p_hole,
                p_hazard,
                leave,
            };
        }
    }
    best
}
