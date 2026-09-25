//! The observation and action decoding: a port of rl/obs.mjs (spec:
//! docs/SPEC.md). The browser agent still runs rl/obs.mjs against the real
//! game, so the two must agree exactly - rl/test/parity.test.mjs checks every
//! float. Change both (and OBS_VERSION) together.

use crate::game::*;
use crate::jsmath::{self as jm, hypot2, js_max, js_min};

pub const OBS_VERSION: u32 = 3;
pub const N_CLUBS: usize = 11;
pub const N_SPIN: usize = 3;

pub const AIM_SCALE: f64 = 0.35;
pub const AIM_CLIP: f64 = 3.0;
pub const DIST_SCALE: f64 = 0.35;
pub const DIST_LO: f64 = -4.0;
pub const DIST_HI: f64 = 2.0;

const GA_FWD: [f64; 12] = [-0.1, 0.1, 0.25, 0.4, 0.55, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5];
const GA_LAT: [f64; 9] = [-0.4, -0.25, -0.12, -0.05, 0.0, 0.05, 0.12, 0.25, 0.4];
const GB_LAT: [f64; 7] = [-50.0, -30.0, -15.0, 0.0, 15.0, 30.0, 50.0];
const N_GB_FWD: usize = 15;
const N_LOOK: usize = 10;
const N_RAYS: usize = 13;
const RAY_LEN: f64 = 80.0;
const RAY_STEPS: [f64; 7] = [4.0, 8.0, 14.0, 22.0, 32.0, 45.0, 60.0];
const CH: usize = 6;

pub const OBS_DIM: usize = 7
    + 2
    + 2
    + 3
    + 3
    + 2
    + 1
    + 3
    + 11
    + 3
    + 3
    + 2 * N_RAYS
    + 6
    + 8
    + 5
    + 2 * N_LOOK
    + CH * (GA_FWD.len() * GA_LAT.len() + N_GB_FWD * GB_LAT.len());

pub const ESCAPE_YD: f64 = 10.0;

#[inline]
pub fn clip(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// The last shot, as the observation reports it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Prev {
    pub moved: f64,
    pub tree: bool,
    pub hazard: bool,
}

/// isStuck: a full swing that moved under ESCAPE_YD or found a hazard.
pub fn is_stuck(prev: &Prev, last_was_putt: bool) -> bool {
    !last_was_putt && (prev.hazard || prev.moved < ESCAPE_YD)
}

/// The game's default shot (enterAim / aimDefault).
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub club: usize,
    pub dir: f64,
    pub dist: f64,
}

/// A raw action: club 0-10, spin 0-2 (back/none/top), aim, dist.
#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub club: usize,
    pub spin: usize,
    pub aim: f64,
    pub dist: f64,
}

/// A decoded action: the swing the game plays.
#[derive(Clone, Copy, Debug)]
pub struct Shot {
    pub club: usize,
    pub spin: f64,
    pub yaw: f64,
    pub power: f64,
    pub lm: f64,
    pub want: f64,
}

/// An observation frame: +fwd along the angle a, +lat to its right.
#[derive(Clone, Copy)]
struct Frame {
    bx: f64,
    bz: f64,
    sn: f64,
    cs: f64,
}

impl Frame {
    fn new(b: &Ball, a: f64) -> Frame {
        Frame {
            bx: b.x,
            bz: b.z,
            sn: jm::sin(a),
            cs: jm::cos(a),
        }
    }
    #[inline]
    fn to_w(self, f: f64, l: f64) -> (f64, f64) {
        (self.bx + self.sn * f + self.cs * l, self.bz + self.cs * f - self.sn * l)
    }
    #[inline]
    fn to_p(self, x: f64, z: f64) -> (f64, f64) {
        let dx = x - self.bx;
        let dz = z - self.bz;
        (dx * self.sn + dz * self.cs, dx * self.cs - dz * self.sn)
    }
    #[inline]
    fn vec(self, x: f64, z: f64) -> (f64, f64) {
        (x * self.sn + z * self.cs, x * self.cs - z * self.sn)
    }
}

pub fn pin_dist(g: &Game) -> f64 {
    hypot2(g.hole.pin.x - g.ball.x, g.hole.pin.z - g.ball.z)
}

pub fn pin_dir(g: &Game) -> f64 {
    jm::atan2(g.hole.pin.x - g.ball.x, g.hole.pin.z - g.ball.z)
}

/// Observer.reference(): the game's club, aimed at the pin when it can
/// plausibly reach it, else at a lay-up one carry down the centreline.
pub fn reference(g: &mut Game) -> Reference {
    let club = g.auto_club();
    let d = pin_dist(g);
    let reach = if club == CLUB_PUTTER {
        PUTT_MAX
    } else {
        CLUBS[club].1 * g.lie_mul(club)
    };
    if d < reach + 20.0 {
        return Reference {
            club,
            dir: pin_dir(g),
            dist: d,
        };
    }
    let (bx, bz) = (g.ball.x, g.ball.z);
    g.dist_to_path_mut(bx, bz);
    let t = g.path_point_at(js_min(g.last_along + reach * 0.95, g.hole.len));
    Reference {
        club,
        dir: jm::atan2(t.x - bx, t.z - bz),
        dist: js_max(1.0, hypot2(t.x - bx, t.z - bz)),
    }
}

/// Observer.decode: a raw action into the shot, relative to the reference.
pub fn decode(g: &Game, r: &Reference, a: &Action) -> Shot {
    let yaw = r.dir + AIM_SCALE * clip(a.aim, -AIM_CLIP, AIM_CLIP);
    let want = js_max(0.3, r.dist * jm::exp(DIST_SCALE * clip(a.dist, DIST_LO, DIST_HI)));
    let lm = g.lie_mul(a.club);
    let power = if a.club == CLUB_PUTTER {
        clip(want / PUTT_MAX, 0.005, 1.0)
    } else {
        clip(want / (CLUBS[a.club].1 * lm), 0.02, 1.0)
    };
    Shot {
        club: a.club,
        spin: if a.club == CLUB_PUTTER {
            0.0
        } else {
            a.spin as f64 - 1.0
        },
        yaw,
        power,
        lm,
        want,
    }
}

/// treeAt: 1 inside a canopy the ball can hit
fn tree_at(g: &Game, x: f64, z: f64) -> f64 {
    for &n in g.trees.at(x, z) {
        let t = &g.hole.near[n as usize];
        if (x - t.x) * (x - t.x) + (z - t.z) * (z - t.z) < t.s * t.s {
            return 1.0;
        }
    }
    0.0
}

/// The game's putt preview plus the closest pass to the cup:
/// (stop x, stop z, closest distance, lateral miss there)
fn putt_preview(g: &mut Game, dist: f64, dir: f64) -> (f64, f64, f64, f64) {
    let (px, pz) = (g.hole.pin.x, g.hole.pin.z);
    let sn = jm::sin(dir);
    let cs = jm::cos(dir);
    let mut r = Ball {
        x: g.ball.x,
        y: g.ball.y,
        z: g.ball.z,
        ..Default::default()
    };
    Game::putt_vel(&mut r, dist, dir);
    let mut best = 1e9;
    let mut lat = 0.0;
    g.roll_rest = false;
    let mut i = 0;
    while i < 600 && !g.roll_rest {
        g.roll_step(&mut r);
        let cd = hypot2(r.x - px, r.z - pz);
        if cd < best {
            best = cd;
            lat = (r.x - px) * cs - (r.z - pz) * sn;
        }
        i += 1;
    }
    (r.x, r.z, best, lat)
}

struct Put<'a> {
    o: &'a mut [f32],
    i: usize,
}

impl Put<'_> {
    #[inline]
    fn put(&mut self, v: f64) {
        self.o[self.i] = v as f32;
        self.i += 1;
    }
    #[inline]
    fn flag(&mut self, b: bool) {
        self.put(if b { 1.0 } else { 0.0 });
    }
}

fn rays(g: &Game, dir: f64, hb: f64, p: &mut Put) {
    let b = g.ball;
    let near: Vec<&Tree> = g
        .hole
        .near
        .iter()
        .filter(|t| hypot2(t.x - b.x, t.z - b.z) < RAY_LEN + t.s)
        .collect();
    for k in 0..N_RAYS {
        let off = (k as f64 - 6.0) * std::f64::consts::PI / 18.0;
        let a = dir + off;
        let ux = jm::sin(a);
        let uz = jm::cos(a);
        let mut hit = RAY_LEN;
        for t in &near {
            let dx = t.x - b.x;
            let dz = t.z - b.z;
            let al = dx * ux + dz * uz;
            let pp = (dx * uz - dz * ux).abs();
            if al > 0.0 && pp < t.s {
                hit = js_min(hit, js_max(0.0, al - (t.s * t.s - pp * pp).sqrt()));
            }
        }
        let mut rise = -1.0;
        for r in RAY_STEPS {
            rise = js_max(rise, jm::tanh((g.height_at(b.x + ux * r, b.z + uz * r) - hb) / r * 2.0));
        }
        p.put(1.0 - hit / RAY_LEN);
        p.put(rise);
    }
}

/// Observer.observe: writes OBS_DIM floats into `o` and returns the reference
/// shot that decode() is relative to.
pub fn observe(g: &mut Game, strokes: i32, prev: &Prev, max_over: i32, o: &mut [f32]) -> Reference {
    assert_eq!(o.len(), OBS_DIM);
    let d = pin_dist(g);
    let pdir = pin_dir(g);
    let r = reference(g);
    let dir = r.dir;
    let auto = r.club;
    let b = g.ball;
    let fa = Frame::new(&b, dir);
    let fp = Frame::new(&b, pdir);
    let gr = g.ground_at(b.x, b.z);
    let hb = gr.h;
    let h = &g.hole;
    let (par, len, hills, wind_a, wind_s, pin) = (h.par, h.len, h.hills, h.wind_a, h.wind_s, h.pin);
    let mut p = Put { o, i: 0 };

    for s in 0..7u8 {
        p.flag(gr.s == s);
    }
    p.put(g.lie_mul(0));
    p.put(g.lie_mul(8));
    let wr = wind_a - dir;
    p.put(wind_s * jm::cos(wr) / 8.0);
    p.put(wind_s * jm::sin(wr) / 8.0);
    for q in [3, 4, 5] {
        p.flag(par == q);
    }
    p.put(strokes as f64 / 10.0);
    p.put((par + max_over - strokes) as f64 / 10.0);
    p.put(hills);
    let (gx, gz) = g.slope_at(b.x, b.z, 0.5);
    let (sx, sz) = fa.vec(gx, gz);
    p.put(sx * 5.0);
    p.put(sz * 5.0);
    p.put(jm::tanh((g.height_at(pin.x, pin.z) - hb) / 10.0));
    p.put(d / 300.0);
    p.put(jm::log1p(d) / 6.0);
    p.flag(d < 45.0);
    for c in 0..N_CLUBS {
        p.flag(c == auto);
    }
    let dp = g.dist_to_path_mut(b.x, b.z);
    let along = g.last_along;
    p.put(dp / 60.0);
    p.put(along / len);
    p.put((len - along) / 300.0);
    p.put(jm::log1p(prev.moved) / 6.0);
    p.flag(prev.tree);
    p.flag(prev.hazard);
    rays(g, dir, hb, &mut p);
    let (pf, pl) = fa.to_p(pin.x, pin.z);
    p.put(pf / 300.0);
    p.put(pl / 300.0);
    p.put(jm::cos(pdir - dir));
    p.put(jm::sin(pdir - dir));
    p.put(r.dist / 300.0);
    p.put(jm::log(r.dist / js_max(d, 1.0)));

    for k in [1.0, PUTT_OVER] {
        if d < 45.0 {
            let (tx, tz, best, lat) = putt_preview(g, d * k, pdir);
            let (f, l) = fp.to_p(tx, tz);
            let dm = js_max(d, 1.0);
            p.put(jm::tanh((f - d) / dm));
            p.put(jm::tanh(l / dm * 5.0));
            p.put(jm::tanh(lat / dm * 5.0));
            p.flag(best < HOLE_R);
        } else {
            for _ in 0..4 {
                p.put(0.0);
            }
        }
    }
    if auto != CLUB_PUTTER {
        let lm = g.lie_mul(auto);
        let power = js_min(1.0, r.dist / (CLUBS[auto].1 * lm));
        let (q, hit) = g.predict_landing(auto, dir, 0.0, lm, power);
        let (f, l) = fa.to_p(q.x, q.z);
        p.put(jm::tanh((f - r.dist) / 50.0));
        p.put(jm::tanh(l / 30.0));
        p.flag(hit);
        let s = g.surface_at(q.x, q.z);
        p.flag(s >= SURF_WATER);
        p.flag(s == SURF_BUNKER);
    } else {
        for _ in 0..5 {
            p.put(0.0);
        }
    }
    for k in 0..N_LOOK {
        let a = 30.0 + 30.0 * k as f64;
        let q = g.path_point_at(js_min(along + a, len));
        let (f, l) = fa.to_p(q.x, q.z);
        p.put(f / 300.0);
        p.put(l / 300.0);
    }

    let cell = |p: &mut Put, x: f64, z: f64, hs: f64| {
        let s = g.surface_at(x, z);
        p.flag(s >= SURF_WATER);
        p.flag(s == SURF_BUNKER);
        p.flag(s == SURF_GREEN);
        p.flag(s == SURF_FAIRWAY || s == SURF_TEE);
        p.put(tree_at(g, x, z));
        p.put(jm::tanh((g.height_at(x, z) - hb) / hs));
    };
    let sc = js_max(d, 4.0);
    for f in GA_FWD {
        for l in GA_LAT {
            let (x, z) = fp.to_w(f * sc, l * sc);
            cell(&mut p, x, z, js_max(0.5, sc * 0.05));
        }
    }
    for k in 0..N_GB_FWD {
        let f = 20.0 + 20.0 * k as f64;
        for l in GB_LAT {
            let (x, z) = fa.to_w(f, l);
            cell(&mut p, x, z, 15.0);
        }
    }
    assert_eq!(p.i, OBS_DIM);
    r
}
