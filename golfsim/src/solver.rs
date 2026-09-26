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

const PI: f64 = std::f64::consts::PI;
const TAU: f64 = 2.0 * std::f64::consts::PI;

/// Solver.solve's result. `spin` is the launch spin (-1 back, 0 none).
#[derive(Clone, Copy, Debug, Default)]
pub struct Solution {
    pub tried: bool,
    pub found: bool,
    pub club: usize,
    pub spin: f64,
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

/// noiseSamples: (impact, aim error, weight)
fn noise_samples(impact_noise: f64, aim_noise: f64, putt: bool) -> Vec<(f64, f64, f64)> {
    let mut imp = Vec::with_capacity(3);
    if impact_noise <= PERFECT {
        imp.push((0.0, 1.0));
    } else {
        let pp = PERFECT / impact_noise;
        let e = PERFECT + (impact_noise - PERFECT) * 0.5;
        imp.push((0.0, pp));
        imp.push((-e, (1.0 - pp) / 2.0));
        imp.push((e, (1.0 - pp) / 2.0));
    }
    let aim: Vec<(f64, f64)> = if putt || aim_noise <= 0.0 {
        vec![(0.0, 1.0)]
    } else {
        vec![
            (-aim_noise * 0.75, 0.25),
            (-aim_noise * 0.25, 0.25),
            (aim_noise * 0.25, 0.25),
            (aim_noise * 0.75, 0.25),
        ]
    };
    let mut out = Vec::with_capacity(imp.len() * aim.len());
    for &(i, wi) in &imp {
        for &(a, wa) in &aim {
            out.push((i, a, wi * wa));
        }
    }
    out
}

/// Solver.candidates: (club, spin) pairs worth solving from here.
fn candidates(auto: usize, lie: u8, d: f64) -> Vec<(usize, f64)> {
    let mut c = Vec::with_capacity(3);
    if d < PUTT_MAX && (lie == SURF_GREEN || lie == SURF_FAIRWAY || lie == SURF_TEE) {
        c.push((CLUB_PUTTER, 0.0));
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
        c.push((f, 0.0));
        c.push((f, -1.0));
    }
    c
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
    for (club, spin) in candidates(auto, lie, d) {
        let putt = club == CLUB_PUTTER;
        let lm = g.lie_mul(club);
        let max = if putt { PUTT_MAX } else { CLUBS[club].1 * lm };
        let lo = if putt { 0.005 } else { 0.02 };
        let (mut yaw, mut want) = (pdir, d);
        let mut hit: Option<(f64, f64, f64)> = None;
        for _ in 0..SOLVE_ITERS {
            let power = js_min(js_max(want / max, lo), 1.0);
            let r = trial(g, club, spin, yaw, power, lm, 0.0, 0.0, pin_out);
            if r.ev == EV_HOLED {
                hit = Some((yaw, power, want));
                break;
            }
            let dx = r.x - bx;
            let dz = r.z - bz;
            let got = hypot2(dx, dz);
            if got < 0.5 || (power == 1.0 && got < d) {
                break;
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
        let Some((hy, hp, hw)) = hit else { continue };
        let (mut p_hole, mut p_hazard, mut leave) = (0.0, 0.0, 0.0);
        for (imp, aim, w) in noise_samples(impact_noise, aim_noise, putt) {
            if imp == 0.0 && aim == 0.0 {
                p_hole += w;
                continue;
            }
            let r = trial(g, club, spin, hy, hp, lm, imp, aim, pin_out);
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
