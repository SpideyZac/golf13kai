//! Sunshine Golf Classic's course and ball physics: a line-by-line port of
//! Golf13K/game/course.js and Golf13K/game/golfSim.js (plus the engineMath.js
//! helpers they use). The JS globals (`hole`, `ball`, `ballAir`, `shotDir`,
//! ...) are fields of `Game`, one per env instance.
//!
//! Every expression keeps the JS operand order, every RandomGenerator draw
//! happens in the JS order, and the math goes through `jsmath`, so a hole and
//! every shot on it come out BIT-IDENTICAL to the browser game's (checked by
//! rl/test/parity.test.mjs). If the submodule changes, re-port and re-run it.
//!
//! Two things are faster than the JS without changing any result:
//! - the value noise's lattice hashes (one `sin` each) are cached per hole;
//! - the ball's tree test looks up a grid of the trees that could touch it,
//!   and tests those in the JS list order, so the first hit is the same tree.
//!
//! Left out because nothing the agent reads depends on them: colours, sounds,
//! the drawn trail and roll, the swing meter (the env calls launchBall with an
//! impact error directly), non-colliding props (far trees, wildflowers - drawn
//! LAST from the hole's RandomGenerator, so skipping them shifts no other
//! draw), and predictLanding's putter branch (the observer never asks for it).

use crate::jsmath::{self as jm, clamp01, hypot2, hypot3, js_max, js_min, lerp, smooth_step};
use std::cell::Cell;

pub const SURF_ROUGH: u8 = 0;
pub const SURF_FAIRWAY: u8 = 1;
pub const SURF_GREEN: u8 = 2;
pub const SURF_TEE: u8 = 3;
pub const SURF_BUNKER: u8 = 4;
pub const SURF_WATER: u8 = 5;
pub const SURF_OB: u8 = 6;
pub const SURF_NAMES: [&str; 7] = ["ROUGH", "FAIRWAY", "GREEN", "TEE", "SAND", "WATER", "OB"];

pub const EV_HOLED: u8 = 1;
pub const EV_STOPPED: u8 = 2;
pub const EV_WATER: u8 = SURF_WATER;
pub const EV_OB: u8 = SURF_OB;

/// [par, lenScale, fairwayW, dogleg, bunkers, water, treeDen, hills]
pub const CLASSIC_HOLES: [[f64; 8]; 18] = [
    [4.0, 0.90, 50.0, 0.0, 1.0, 0.0, 0.5, 0.3],
    [3.0, 0.80, 0.0, 0.0, 1.0, 0.0, 0.6, 0.5],
    [5.0, 0.72, 40.0, 0.5, 1.0, 1.0, 0.7, 0.4],
    [4.0, 1.00, 36.0, 2.0, 2.0, 0.0, 0.9, 0.5],
    [3.0, 1.30, 0.0, 0.0, 2.0, 0.0, 0.6, 0.5],
    [4.0, 1.00, 36.0, 1.0, 2.0, 0.0, 1.0, 0.5],
    [5.0, 0.90, 40.0, -0.7, 2.0, 0.8, 0.8, 0.5],
    [3.0, 0.90, 0.0, 0.0, 1.0, 1.0, 0.4, 0.4],
    [4.0, 1.00, 36.0, 1.5, 3.0, 0.8, 0.0, 0.3],
    [4.0, 0.95, 36.0, -0.9, 3.0, 0.5, 1.0, 0.7],
    [5.0, 1.00, 36.0, 2.0, 2.0, 0.7, 1.0, 0.6],
    [4.0, 0.6, 30.0, 0.9, 1.0, 1.0, 0.7, 0.3],
    [4.0, 1.05, 30.0, 0.5, 2.0, 0.95, 1.2, 1.1],
    [4.0, 1.10, 28.0, -1.0, 1.0, 0.0, 1.2, 1.3],
    [5.0, 1.05, 30.0, 2.2, 2.0, 0.9, 0.3, 0.8],
    [3.0, 1.10, 0.0, 0.0, 4.0, 0.5, 0.5, 1.0],
    [4.0, 1.10, 24.0, 1.0, 2.0, 0.0, 0.7, 0.5],
    [5.0, 1.10, 30.0, 2.0, 2.0, 0.9, 1.0, 1.2],
];

/// [name, carry yards, loft degrees]
pub const CLUBS: [(&str, f64, f64); 11] = [
    ("1W", 235.0, 11.0),
    ("3W", 215.0, 14.0),
    ("5W", 198.0, 17.0),
    ("3i", 183.0, 19.0),
    ("5i", 168.0, 22.0),
    ("7i", 150.0, 26.0),
    ("9i", 131.0, 31.0),
    ("13i", 120.0, 41.0),
    ("PW", 98.0, 38.0),
    ("SW", 78.0, 48.0),
    ("PT", 0.0, 0.0),
];
pub const CLUB_PUTTER: usize = 10;
pub const PUTT_MAX: f64 = 40.0;
pub const PUTT_OVER: f64 = 1.3;
const PUTT_PUSH: f64 = 0.75;
pub const GRAV: f64 = 11.0;
const PROP_SCALE: f64 = 3.0;
pub const HOLE_R: f64 = 0.059 * PROP_SCALE;
const POLE_H: f64 = 2.3;
const POST_R: f64 = 0.15;
const LIP_R: f64 = 0.4;
const LIP_D: f64 = 0.06;
pub const CUP_SPEED: f64 = 8.0;
const AIR_HOLE: f64 = HOLE_R * 1.8;
const DT: f64 = 1.0 / 60.0;
const WIND_V: f64 = 1.4;
const DRAG_K: f64 = 0.004;
const LIFT_K: f64 = 0.0015;
const CARRY_K: f64 = 0.4;
const LIFT_SPIN: f64 = 0.001;
const SPIN_LOFT: f64 = 8.0;
/// per surface: [bounce restitution, bounce keep, roll friction, power mult]
pub const SURF_PHYS: [[f64; 4]; 7] = [
    [0.26, 0.4, 10.0, 0.8],
    [0.32, 0.52, 6.0, 1.0],
    [0.26, 0.45, 3.0, 1.0],
    [0.32, 0.52, 6.0, 1.0],
    [0.05, 0.20, 30.0, 0.65],
    [0.0, 0.0, 99.0, 1.0],
    [0.25, 0.45, 14.0, 1.0],
];
const TRUNK_R: f64 = 0.2;
const TREE_COOL: i32 = 8;
const GREEN_BUMP: f64 = 0.5;
const GREEN_FLAT: f64 = 0.7;
const MAXWIND: f64 = 7.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct Pt {
    pub x: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PathPt {
    pub x: f64,
    pub z: f64,
    pub along: f64,
}

/// A bunker, or a lake (with its surface height `h`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ellipse {
    pub x: f64,
    pub z: f64,
    pub rx: f64,
    pub rz: f64,
    pub h: f64,
}

/// A prop the ball can hit (the even kinds). `y` is the canopy centre.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tree {
    pub x: f64,
    pub z: f64,
    pub s: f64,
    pub k: u8,
    pub y: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Hole {
    pub par: i32,
    pub len: f64,
    pub fw: f64,
    pub hills: f64,
    pub index: usize,
    pub tree_den: f64,
    pub path: Vec<PathPt>,
    pub bunkers: Vec<Ellipse>,
    pub waters: Vec<Ellipse>,
    /// hole.near: every prop the ball can hit, in the JS order
    pub near: Vec<Tree>,
    /// hole.trees.length after the colliding kinds (flowers are not generated)
    pub n_props: usize,
    pub gr: f64,
    pub green: Pt,
    pub green_h: f64,
    pub tee_h: f64,
    pub pin: Pt,
    pub wind_a: f64,
    pub wind_s: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
}

/// Every global a shot writes (launchBall, shotBegin, ballUpdate, flyStep,
/// rollStep): the solver's trial shots save and restore it (rl/solver.mjs).
#[derive(Clone, Copy, Debug)]
pub struct Flight {
    ball: Ball,
    ball_air: bool,
    ball_rolling: bool,
    bounces: i32,
    pin_hit: bool,
    pin_out: bool,
    ball_spin: f64,
    ball_curve: f64,
    shot_dir: f64,
    shot_sc: (f64, f64),
    tree_hit: bool,
    tree_cool: i32,
    ball_event: u8,
    shot_start: Pt,
    ball_safe: Pt,
    roll_rest: bool,
}

/// groundAt's {h, s}
#[derive(Clone, Copy, Debug)]
pub struct Ground {
    pub h: f64,
    pub s: u8,
}

/// Uniform grid of the colliding props, TREE_CELL yards a side. Each prop is
/// listed in every cell within ceil(s/TREE_CELL) of its own, which covers every
/// point closer than s to it; cells hold prop indices in ascending order.
#[derive(Clone, Default)]
pub struct TreeGrid {
    x0: i64,
    z0: i64,
    w: i64,
    h: i64,
    start: Vec<u32>,
    items: Vec<u32>,
}

pub const TREE_CELL: f64 = 8.0;

impl TreeGrid {
    fn build(trees: &[Tree]) -> TreeGrid {
        if trees.is_empty() {
            return TreeGrid::default();
        }
        let span = |t: &Tree| {
            let r = (t.s / TREE_CELL).ceil() as i64;
            let cx = (t.x / TREE_CELL).floor() as i64;
            let cz = (t.z / TREE_CELL).floor() as i64;
            (cx - r, cx + r, cz - r, cz + r)
        };
        let (mut x0, mut x1, mut z0, mut z1) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
        for t in trees {
            let (a, b, c, d) = span(t);
            x0 = x0.min(a);
            x1 = x1.max(b);
            z0 = z0.min(c);
            z1 = z1.max(d);
        }
        let (w, h) = (x1 - x0 + 1, z1 - z0 + 1);
        let mut count = vec![0u32; (w * h + 1) as usize];
        for t in trees {
            let (a, b, c, d) = span(t);
            for i in a..=b {
                for j in c..=d {
                    count[((i - x0) * h + (j - z0)) as usize + 1] += 1;
                }
            }
        }
        for k in 1..count.len() {
            count[k] += count[k - 1];
        }
        let mut fill = count.clone();
        let mut items = vec![0u32; *count.last().unwrap() as usize];
        for (n, t) in trees.iter().enumerate() {
            let (a, b, c, d) = span(t);
            for i in a..=b {
                for j in c..=d {
                    let cell = ((i - x0) * h + (j - z0)) as usize;
                    items[fill[cell] as usize] = n as u32;
                    fill[cell] += 1;
                }
            }
        }
        TreeGrid {
            x0,
            z0,
            w,
            h,
            start: count,
            items,
        }
    }

    /// Prop indices listed in cell (i, j), ascending.
    #[inline]
    pub fn cell(&self, i: i64, j: i64) -> &[u32] {
        let (ci, cj) = (i - self.x0, j - self.z0);
        if ci < 0 || cj < 0 || ci >= self.w || cj >= self.h {
            return &[];
        }
        let c = (ci * self.h + cj) as usize;
        &self.items[self.start[c] as usize..self.start[c + 1] as usize]
    }

    #[inline]
    pub fn at(&self, x: f64, z: f64) -> &[u32] {
        self.cell((x / TREE_CELL).floor() as i64, (z / TREE_CELL).floor() as i64)
    }
}

const LATTICE_SLOTS: usize = 1 << 13;

pub struct Game {
    pub hole: Hole,
    pub noise_seed: f64,
    pub remix_mode: bool,
    pub ball: Ball,
    pub ball_air: bool,
    pub ball_rolling: bool,
    pub bounces: i32,
    pub pin_hit: bool,
    pub pin_out: bool,
    pub ball_spin: f64,
    pub ball_curve: f64,
    pub shot_dir: f64,
    pub tree_hit: bool,
    pub tree_cool: i32,
    pub ball_event: u8,
    pub shot_start: Pt,
    pub ball_safe: Pt,
    pub roll_rest: bool,
    /// the distance along the path from the last distToPath call
    pub last_along: f64,
    /// Math.random: the env seeds it per episode (wind, swing noise)
    pub rng: jm::Mulberry32,
    pub trees: TreeGrid,
    // caches of values the JS recomputes: sin/cos of the wind and shot
    // directions, and the lattice hashes of the value noise
    wind_sc: (f64, f64),
    shot_sc: (f64, f64),
    lattice_gen: u32,
    lattice: Vec<Cell<(u64, f64)>>,
    cands: Vec<u32>,
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Game {
    pub fn new() -> Game {
        Game {
            hole: Hole::default(),
            noise_seed: 0.0,
            remix_mode: false,
            ball: Ball::default(),
            ball_air: false,
            ball_rolling: false,
            bounces: 0,
            pin_hit: false,
            pin_out: false,
            ball_spin: 0.0,
            ball_curve: 0.0,
            shot_dir: 0.0,
            tree_hit: false,
            tree_cool: 0,
            ball_event: 0,
            shot_start: Pt::default(),
            ball_safe: Pt::default(),
            roll_rest: false,
            last_along: 0.0,
            rng: jm::Mulberry32::new(1),
            trees: TreeGrid::default(),
            wind_sc: (0.0, 1.0),
            shot_sc: (0.0, 1.0),
            lattice_gen: 0,
            lattice: (0..LATTICE_SLOTS).map(|_| Cell::new((u64::MAX, 0.0))).collect(),
            cands: Vec::with_capacity(64),
        }
    }

    pub fn save_flight(&self) -> Flight {
        Flight {
            ball: self.ball,
            ball_air: self.ball_air,
            ball_rolling: self.ball_rolling,
            bounces: self.bounces,
            pin_hit: self.pin_hit,
            pin_out: self.pin_out,
            ball_spin: self.ball_spin,
            ball_curve: self.ball_curve,
            shot_dir: self.shot_dir,
            shot_sc: self.shot_sc,
            tree_hit: self.tree_hit,
            tree_cool: self.tree_cool,
            ball_event: self.ball_event,
            shot_start: self.shot_start,
            ball_safe: self.ball_safe,
            roll_rest: self.roll_rest,
        }
    }

    pub fn restore_flight(&mut self, f: &Flight) {
        self.ball = f.ball;
        self.ball_air = f.ball_air;
        self.ball_rolling = f.ball_rolling;
        self.bounces = f.bounces;
        self.pin_hit = f.pin_hit;
        self.pin_out = f.pin_out;
        self.ball_spin = f.ball_spin;
        self.ball_curve = f.ball_curve;
        self.shot_dir = f.shot_dir;
        self.shot_sc = f.shot_sc;
        self.tree_hit = f.tree_hit;
        self.tree_cool = f.tree_cool;
        self.ball_event = f.ball_event;
        self.shot_start = f.shot_start;
        self.ball_safe = f.ball_safe;
        self.roll_rest = f.roll_rest;
    }

    ///////////////////////////////////////////////////////////////////////////
    // seeded value noise

    #[inline]
    fn hash_n_raw(&self, x: f64, z: f64) -> f64 {
        let s = jm::sin(x * 127.1 + z * 311.7 + self.noise_seed * 74.7) * 43758.5453;
        s - s.floor()
    }

    /// hashN at integer lattice points, through a direct-mapped cache that is
    /// invalidated whenever noiseSeed changes.
    #[inline]
    fn hash_n(&self, x: f64, z: f64) -> f64 {
        if x.abs() >= 1048576.0 || z.abs() >= 1048576.0 {
            return self.hash_n_raw(x, z);
        }
        let (xi, zi) = (x as i64 as u64 & 0x1F_FFFF, z as i64 as u64 & 0x1F_FFFF);
        let key = (self.lattice_gen as u64) << 42 | xi << 21 | zi;
        let slot = ((xi.wrapping_mul(0x9E3779B1) ^ zi.wrapping_mul(0x85EBCA77)) as usize) & (LATTICE_SLOTS - 1);
        let c = &self.lattice[slot];
        let (k, v) = c.get();
        if k == key {
            return v;
        }
        let v = self.hash_n_raw(x, z);
        c.set((key, v));
        v
    }

    pub fn noise2(&self, x: f64, z: f64) -> f64 {
        let xi = x.floor();
        let zi = z.floor();
        let u = smooth_step(x - xi);
        let v = smooth_step(z - zi);
        lerp(
            lerp(self.hash_n(xi, zi), self.hash_n(xi + 1.0, zi), u),
            lerp(self.hash_n(xi, zi + 1.0), self.hash_n(xi + 1.0, zi + 1.0), u),
            v,
        )
    }

    fn set_noise_seed(&mut self, s: f64) {
        self.noise_seed = s;
        // generations 1..2^21: a key's top bit stays clear, so no live key
        // equals the empty marker
        self.lattice_gen += 1;
        if self.lattice_gen >= 1 << 21 {
            self.lattice_gen = 1;
            for c in &self.lattice {
                c.set((u64::MAX, 0.0));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // hole queries

    /// distToPath: (distance to the centreline, lastAlong)
    pub fn dist_to_path(&self, x: f64, z: f64) -> (f64, f64) {
        let mut best = 1e9;
        let mut along = self.last_along;
        let p = &self.hole.path;
        let mut i = p.len() - 1;
        while i > 0 {
            i -= 1;
            let (a, b) = (p[i], p[i + 1]);
            let bx = b.x - a.x;
            let bz = b.z - a.z;
            let len2 = bx * bx + bz * bz;
            let t = clamp01(((x - a.x) * bx + (z - a.z) * bz) / len2);
            let dx = x - (a.x + bx * t);
            let dz = z - (a.z + bz * t);
            let d = hypot2(dx, dz);
            if d < best {
                best = d;
                along = a.along + len2.sqrt() * t;
            }
        }
        (best, along)
    }

    /// distToPath with its side effect on lastAlong
    pub fn dist_to_path_mut(&mut self, x: f64, z: f64) -> f64 {
        let (d, a) = self.dist_to_path(x, z);
        self.last_along = a;
        d
    }

    pub fn path_point_at(&self, d: f64) -> Pt {
        let p = &self.hole.path;
        let mut i = p.len() - 1;
        while i > 0 {
            i -= 1;
            if d > p[i].along || i == 0 {
                let (a, b) = (p[i], p[i + 1]);
                let t = clamp01((d - a.along) / (b.along - a.along));
                return Pt {
                    x: lerp(a.x, b.x, t),
                    z: lerp(a.z, b.z, t),
                };
            }
        }
        unreachable!("a path has at least two points")
    }

    #[inline]
    fn ellipse_dist(x: f64, z: f64, e: &Ellipse) -> f64 {
        hypot2((x - e.x) / e.rx, (z - e.z) / e.rz)
    }

    pub fn height_raw(&self, x: f64, z: f64) -> f64 {
        (self.noise2(x * 0.017 + 9.0, z * 0.017) - 0.5) * 20.0 * self.hole.hills
            + (self.noise2(x * 0.04, z * 0.04 + 7.0) - 0.5) * 6.0 * self.hole.hills
    }

    pub fn height_at(&self, x: f64, z: f64) -> f64 {
        let hole = &self.hole;
        let mut h = self.height_raw(x, z);
        let far = clamp01((self.dist_to_path(x, z).0 - 110.0) / 250.0);
        h += far * far * (18.0 + 50.0 * self.noise2(x * 0.004 + 3.0, z * 0.004));
        const WATER_OFFSET: f64 = 1.0;
        for w in &hole.waters {
            let nd = Self::ellipse_dist(x, z, w);
            if nd < 2.0 {
                h = lerp(h, w.h + WATER_OFFSET, smooth_step(2.0 - nd));
            }
        }
        let dg = hypot2(x - hole.green.x, z - hole.green.z);
        let k = smooth_step(1.0 - dg / (hole.gr * 2.4));
        h = lerp(h, hole.green_h, k * GREEN_FLAT) + k * GREEN_BUMP;
        const BANK_R: f64 = 1.2;
        if k != 0.0 {
            for w in &hole.waters {
                let nd = Self::ellipse_dist(x, z, w);
                if nd < BANK_R {
                    h = lerp(
                        h,
                        w.h + WATER_OFFSET,
                        smooth_step((BANK_R - nd) / (BANK_R - 1.0)) * smooth_step((dg - hole.gr) / 3.0 + 1.0),
                    );
                }
            }
        }
        let kt = smooth_step(1.0 - hypot2(x, z) / 14.0);
        h = lerp(h, hole.tee_h, kt);
        for b in &hole.bunkers {
            h -= 2.0 * (1.0 - k) * smooth_step(1.0 - Self::ellipse_dist(x, z, b));
        }
        h
    }

    pub fn slope_at(&self, x: f64, z: f64, e: f64) -> (f64, f64) {
        (
            (self.height_at(x + e, z) - self.height_at(x - e, z)) / (2.0 * e),
            (self.height_at(x, z + e) - self.height_at(x, z - e)) / (2.0 * e),
        )
    }

    /// surfaceAt, and the lake's height when it is water (lastWater.h)
    fn surface_water(&self, x: f64, z: f64) -> (u8, f64) {
        let hole = &self.hole;
        let (dp, along) = self.dist_to_path(x, z);
        if dp > 62.0 + (self.noise2(x * 0.05, z * 0.05) - 0.5) * 18.0 {
            return (SURF_OB, 0.0);
        }
        let dg = hypot2(x - hole.green.x, z - hole.green.z);
        if dg < hole.gr {
            return (SURF_GREEN, 0.0);
        }
        if x.abs() < 5.0 && z.abs() < 5.0 {
            return (SURF_TEE, 0.0);
        }
        for b in &hole.bunkers {
            if Self::ellipse_dist(x, z, b) < 1.0 {
                return (SURF_BUNKER, 0.0);
            }
        }
        for w in &hole.waters {
            if Self::ellipse_dist(x, z, w) < 1.0 {
                return (SURF_WATER, w.h);
            }
        }
        if hole.fw != 0.0 && along > 8.0 && along < hole.len - 2.0 {
            let fw_here = hole.fw * (1.0 + (self.noise2(along * 0.014, hole.index as f64 * 7.0 + 3.0) - 0.5) * 0.9);
            if dp < fw_here * 0.5 + (self.noise2(x * 0.09 + 5.0, z * 0.09) - 0.5) * 7.0 {
                return (SURF_FAIRWAY, 0.0);
            }
        }
        (SURF_ROUGH, 0.0)
    }

    #[inline]
    pub fn surface_at(&self, x: f64, z: f64) -> u8 {
        self.surface_water(x, z).0
    }

    pub fn ground_at(&self, x: f64, z: f64) -> Ground {
        let (s, wh) = self.surface_water(x, z);
        Ground {
            h: if s == SURF_WATER { wh } else { self.height_at(x, z) },
            s,
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // generation

    /// genCourse: the classic rows, or the rows re-dealt under the seed
    pub fn gen_course(seed: f64, remix: bool) -> [[f64; 8]; 18] {
        let mut rows = CLASSIC_HOLES;
        if remix {
            let mut r = jm::RandomGenerator::new(seed);
            for i in (0..18).rev() {
                let j = r.int((i + 1) as f64) as usize;
                rows.swap(i, j);
            }
        }
        rows
    }

    pub fn gen_hole(&mut self, course_seed: f64, index: usize, row: &[f64; 8]) {
        let mut r = jm::RandomGenerator::new(course_seed * 137.0 + index as f64 * 7919.0 + 1.0);
        let [par, len_scale, fw, dog, bunker_n, water_c, tree_den, hills] = *row;
        let len = (if par == 3.0 {
            165.0
        } else if par == 4.0 {
            410.0
        } else {
            620.0
        }) * len_scale;
        self.set_noise_seed(r.float(1e3, 0.0));

        self.hole = Hole {
            par: par as i32,
            len,
            fw,
            hills,
            index,
            tree_den,
            ..Default::default()
        };

        let mut p = vec![PathPt {
            x: 0.0,
            z: 0.0,
            along: 0.0,
        }];
        let add_pt = |p: &mut Vec<PathPt>, x: f64, z: f64| {
            let l = *p.last().unwrap();
            p.push(PathPt {
                x,
                z,
                along: l.along + hypot2(x - l.x, z - l.z),
            });
        };
        if par == 3.0 || dog == 0.0 {
            add_pt(&mut p, 0.0, len);
        } else if dog == 2.0 {
            let a1 = r.float_sign(0.3, 0.4);
            let a2 = -a1 * r.float(0.8, 1.2);
            let z1 = len * 0.4;
            let z2 = len * 0.8;
            add_pt(&mut p, 0.0, z1);
            add_pt(&mut p, jm::sin(a1) * (z2 - z1), z1 + jm::cos(a1) * (z2 - z1));
            let q = p[2];
            add_pt(&mut p, q.x + jm::sin(a2) * (len - z2), q.z + jm::cos(a2) * (len - z2));
        } else {
            let bend_z = len * r.float(0.4, 0.6);
            let a = dog * r.float(0.4, 0.6);
            add_pt(&mut p, 0.0, bend_z);
            add_pt(
                &mut p,
                jm::sin(a) * (len - bend_z),
                bend_z + jm::cos(a) * (len - bend_z),
            );
        }
        self.hole.path = p;

        let end = *self.hole.path.last().unwrap();
        self.hole.gr = r.float(12.0, 20.0) - hills + if water_c == 1.0 { 3.0 } else { 0.0 };
        self.hole.green = Pt { x: end.x, z: end.z };
        self.hole.green_h = self.height_raw(end.x, end.z) + GREEN_BUMP;
        self.hole.tee_h = self.height_raw(0.0, 0.0) + 0.3;
        let pa = r.float(1e3, 0.0);
        let pd = r.float(0.0, self.hole.gr * 0.5);
        self.hole.pin = Pt {
            x: end.x + jm::sin(pa) * pd,
            z: end.z + jm::cos(pa) * pd,
        };

        // wind: Math.random, rolled fresh every play
        let wa = 0.0 + self.rng.next() * (std::f64::consts::PI * 2.0 - 0.0);
        let u = 0.0 + self.rng.next() * (1.0 - 0.0);
        self.hole.wind_a = wa;
        self.hole.wind_s = 1.0 + u * u * MAXWIND;
        self.wind_sc = (jm::sin(wa), jm::cos(wa));

        let gr = self.hole.gr;
        let green = self.hole.green;
        let mut side = r.sign();
        for _ in 0..bunker_n as i32 {
            let a = r.float(1e3, 0.0);
            let d = gr + r.float(2.0, 7.0);
            let gx = green.x + jm::sin(a) * d;
            let gz = green.z + jm::cos(a) * d;
            if index == 0 || r.bool(0.5) || par == 3.0 && water_c == 1.0 {
                let rx = r.float(6.0, 18.0);
                let rz = r.float(6.0, 18.0);
                self.hole.bunkers.push(Ellipse {
                    x: gx,
                    z: gz,
                    rx,
                    rz,
                    h: 0.0,
                });
            } else {
                side *= -1.0;
                let q = self.path_point_at(len * r.float(0.4, 0.9));
                let x = q.x + side * (fw * 0.5 + r.float(-2.0, 4.0));
                let rx = r.float(6.0, 18.0);
                let rz = r.float(6.0, 18.0);
                self.hole.bunkers.push(Ellipse {
                    x,
                    z: q.z,
                    rx,
                    rz,
                    h: 0.0,
                });
            }
        }

        let is_hard_tree_hole = index == 12;
        let is_island_hole = water_c == 1.0;
        let is_river_hole = index == 9;
        if is_river_hole {
            let q = self.path_point_at(len * 0.5);
            self.hole.waters.push(Ellipse {
                x: q.x,
                z: q.z,
                rx: 60.0,
                rz: 9.0,
                h: 0.0,
            });
        }
        if is_island_hole {
            let rx = r.float(40.0, 60.0);
            let rz = r.float(40.0, 60.0);
            self.hole.waters.push(Ellipse {
                x: end.x,
                z: end.z - len * 0.1,
                rx,
                rz,
                h: 0.0,
            });
        } else if r.bool(water_c) {
            let n = 1 + r.bool(0.3) as i32;
            for _ in 0..n {
                let q = self.path_point_at(len * r.float(0.4, 1.0));
                side *= -1.0;
                let x = q.x + side * (fw * 0.5 + r.float(9.0, 22.0));
                let z = q.z + r.float_sign(15.0, 0.0);
                let rx = r.float(14.0, 26.0);
                let rz = r.float(16.0, 34.0);
                self.hole.waters.push(Ellipse { x, z, rx, rz, h: 0.0 });
            }
        }
        for i in 0..self.hole.waters.len() {
            let w = self.hole.waters[i];
            self.hole.waters[i].h = js_min(self.height_raw(w.x, w.z), self.hole.green_h) - 2.0;
        }

        // props. `near` gets the colliding kinds (0 tree, 2 bush) in order;
        // `n` counts every prop, as hole.trees.length does.
        let mut near: Vec<Tree> = Vec::with_capacity(2048);
        let mut n = 0usize;
        let tree_count = js_min(400.0, jm::to_int32(tree_den * len * 0.5) as f64) as usize;
        let tree_scale = if index != 4 || self.remix_mode { 1.0 } else { 2.0 };
        let mut i = tree_count;
        while i > 0 && n < tree_count {
            i -= 1;
            let q = self.path_point_at(r.float(-9.0, len * 1.1));
            let sg = r.sign();
            let off = (if fw * 0.5 != 0.0 { fw * 0.5 } else { 12.0 }) + r.float(4.0, 70.0);
            let x = q.x + sg * off;
            let z = q.z + r.float_sign(9.0, 0.0);
            if self.surface_at(x, z) == SURF_ROUGH {
                near.push(Tree {
                    x,
                    z,
                    s: tree_scale * r.float(2.0, 6.0),
                    k: 0,
                    y: 0.0,
                });
                n += 1;
            }
        }
        let (mut x0, mut x1) = (0.0f64, 0.0f64);
        for q in &self.hole.path {
            x0 = js_min(x0, q.x);
            x1 = js_max(x1, q.x);
        }
        for _ in 0..4000 {
            let x = r.float(x0 - 450.0, x1 + 450.0);
            let z = r.float(-450.0, len + 450.0);
            let dp = self.dist_to_path(x, z).0;
            if dp > 75.0 && self.noise2(x * 0.02 + 5.0, z * 0.02) > 0.5 {
                let s = tree_scale * r.float(2.0, 8.0);
                if dp <= 180.0 {
                    near.push(Tree { x, z, s, k: 0, y: 0.0 });
                }
                n += 1;
            }
        }
        for _ in 0..99 {
            let q = self.path_point_at(r.float(0.0, len));
            let x = q.x + r.float_sign(15.0, 200.0);
            let z = q.z + r.float_sign(60.0, 0.0);
            let s = self.surface_at(x, z);
            if s == SURF_ROUGH || s == SURF_OB {
                near.push(Tree {
                    x,
                    z,
                    s: r.float(1.0, 2.0),
                    k: 2,
                    y: 0.0,
                });
                n += 1;
            }
        }
        // (wildflowers: never collide, and nothing draws from R after them)
        if is_hard_tree_hole {
            let q = self.path_point_at(len * 0.28);
            near.push(Tree {
                x: q.x,
                z: q.z,
                s: 5.0,
                k: 0,
                y: 0.0,
            });
            n += 1;
        }
        for t in near.iter_mut() {
            t.y = self.height_at(t.x, t.z) + t.s * if t.k > 1 { 1.2 } else { 3.4 };
        }
        self.trees = TreeGrid::build(&near);
        self.hole.near = near;
        self.hole.n_props = n;
    }

    ///////////////////////////////////////////////////////////////////////////
    // shot launch

    pub fn ball_ground(&self) -> Ground {
        self.ground_at(self.ball.x, self.ball.z)
    }

    pub fn ball_to_pin(&self) -> f64 {
        hypot2(self.ball.x - self.hole.pin.x, self.ball.z - self.hole.pin.z)
    }

    /// launchVel: sets b's velocity
    fn launch_vel(b: &mut Ball, club: usize, dir: f64, lie_mul: f64, power: f64, spin: f64) {
        let c = CLUBS[club];
        let la = (c.2 + 7.0 - spin * SPIN_LOFT) * std::f64::consts::PI / 180.0;
        let q = lie_mul * power;
        let v = (c.1 * q * (1.0 + CARRY_K * q) * GRAV / jm::sin(2.0 * la)).sqrt();
        let cl = jm::cos(la);
        b.vx = jm::sin(dir) * cl * v;
        b.vy = jm::sin(la) * v;
        b.vz = jm::cos(dir) * cl * v;
    }

    fn shot_begin(&mut self, air: bool, dir: f64) {
        let p = Pt {
            x: self.ball.x,
            z: self.ball.z,
        };
        self.shot_start = p;
        self.ball_safe = p;
        self.ball_air = air;
        self.ball_rolling = !air;
        self.bounces = 0;
        self.pin_hit = false;
        self.tree_cool = 0;
        self.ball_event = 0;
        self.shot_dir = dir;
        self.shot_sc = (jm::sin(dir), jm::cos(dir));
    }

    /// launchBall: the one launch, putter included
    pub fn launch_ball(
        &mut self,
        club: usize,
        mut power: f64,
        impact: f64,
        spin: f64,
        mut dir: f64,
        lie_mul: f64,
    ) -> f64 {
        let putt = club == CLUB_PUTTER;
        let err = if impact.abs() < 0.02 { 0.0 } else { impact };
        power *= 1.0 - js_max(err, -err) * 0.35;
        dir += err * if putt { PUTT_PUSH } else { 0.05 };
        self.ball_curve = err * 22.0;
        self.ball_spin = if putt { 0.0 } else { spin };
        self.shot_begin(!putt, dir);
        if putt {
            Self::putt_vel(&mut self.ball, power * PUTT_MAX, dir);
        } else {
            Self::launch_vel(&mut self.ball, club, dir, lie_mul, power, spin);
            self.ball.y = self.ball_ground().h + 0.1;
        }
        err
    }

    pub fn putt_vel(b: &mut Ball, dist: f64, dir: f64) {
        let v = (6.0 * dist).sqrt() * 1.02;
        b.vx = jm::sin(dir) * v;
        b.vz = jm::cos(dir) * v;
        b.vy = 0.0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // simulation

    /// pathDist: closest approach of the step x0,z0 -> b to (px, pz); also segT
    #[inline]
    fn path_dist(b: &Ball, x0: f64, z0: f64, px: f64, pz: f64) -> (f64, f64) {
        let dx = b.x - x0;
        let dz = b.z - z0;
        let l2 = dx * dx + dz * dz;
        let seg_t = if l2 != 0.0 {
            clamp01(((px - x0) * dx + (pz - z0) * dz) / l2)
        } else {
            0.0
        };
        (hypot2(px - x0 - dx * seg_t, pz - z0 - dz * seg_t), seg_t)
    }

    #[inline]
    fn cup_hit(&self, x0: f64, z0: f64, r: f64) -> bool {
        Self::path_dist(&self.ball, x0, z0, self.hole.pin.x, self.hole.pin.z).0 < r
    }

    /// flyStep for a scratch ball (the preview): no trees
    fn fly_step_free(&self, b: &mut Ball, curve: f64, wv: f64) {
        let (sw, cw) = self.wind_sc;
        let (ss, cs) = self.shot_sc;
        let ry = b.vy;
        let rx = b.vx - sw * wv;
        let rz = b.vz - cw * wv;
        let sp = hypot3(rx, ry, rz);
        let vh = {
            let h = hypot2(rx, rz);
            if h != 0.0 {
                h
            } else {
                1.0
            }
        };
        let d = DRAG_K * sp * DT;
        let l = (LIFT_K - self.ball_spin * LIFT_SPIN) * sp * DT;
        b.vx += cs * curve * DT - rx * d - ry * rx / vh * l;
        b.vz -= ss * curve * DT + rz * d + ry * rz / vh * l;
        b.vy += vh * l - ry * d - GRAV * DT;
        b.x += b.vx * DT;
        b.y += b.vy * DT;
        b.z += b.vz * DT;
    }

    /// flyStep for the real ball: flight, then the tree strike
    fn fly_step_ball(&mut self) {
        let mut b = self.ball;
        self.fly_step_free(&mut b, self.ball_curve, self.hole.wind_s * WIND_V);
        self.ball = b;
        if self.tree_cool != 0 {
            self.tree_cool -= 1;
            return;
        }
        let x0 = b.x - b.vx * DT;
        let z0 = b.z - b.vz * DT;
        // every prop that could touch this step: those listed in the cells of
        // the step's bounding box, tested in hole.near order
        let i0 = (x0.min(b.x) / TREE_CELL).floor() as i64;
        let i1 = (x0.max(b.x) / TREE_CELL).floor() as i64;
        let j0 = (z0.min(b.z) / TREE_CELL).floor() as i64;
        let j1 = (z0.max(b.z) / TREE_CELL).floor() as i64;
        let mut cands = std::mem::take(&mut self.cands);
        cands.clear();
        if i1 - i0 <= 4 && j1 - j0 <= 4 {
            for i in i0..=i1 {
                for j in j0..=j1 {
                    cands.extend_from_slice(self.trees.cell(i, j));
                }
            }
            if i1 > i0 || j1 > j0 {
                cands.sort_unstable();
                cands.dedup();
            }
        } else {
            // a step longer than the grid expects (never in play): test all
            cands.extend(0..self.hole.near.len() as u32);
        }
        for &n in &cands {
            let t = self.hole.near[n as usize];
            let b = &mut self.ball;
            let dx = b.x - t.x;
            let dz = b.z - t.z;
            let dy = b.y - t.y;
            let r2 = dx * dx + dz * dz;
            let s2 = t.s * t.s;
            let canopy = r2 + dy * dy < s2;
            let (hit, seg_t) = if canopy {
                (dx * b.vx + dy * b.vy + dz * b.vz < 0.0, 0.0)
            } else if dy < 0.0 {
                let (pd, st) = Self::path_dist(b, x0, z0, t.x, t.z);
                (pd < t.s * TRUNK_R && (x0 - t.x) * b.vx + (z0 - t.z) * b.vz < 0.0, st)
            } else {
                (false, 0.0)
            };
            if hit {
                if !canopy {
                    b.x = lerp(x0, b.x, seg_t);
                    b.z = lerp(z0, b.z, seg_t);
                }
                let sp = hypot2(b.vx, b.vz);
                if sp > 2.0 {
                    self.tree_hit = true;
                }
                b.vx *= -0.1;
                b.vz *= -0.1;
                b.vy = js_min(b.vy, 0.0);
                self.tree_cool = TREE_COOL;
                break;
            }
        }
        self.cands = cands;
    }

    /// lieMul: what the lie costs club c
    pub fn lie_mul(&self, c: usize) -> f64 {
        let s = self.ball_ground().s;
        SURF_PHYS[s as usize][3] * if s == SURF_BUNKER && c < 8 { 0.5 } else { 1.0 }
    }

    /// rollStep: one step of roll for b; returns the speed before the step
    /// and sets rollRest
    pub fn roll_step(&mut self, b: &mut Ball) -> f64 {
        let p = SURF_PHYS[self.ground_at(b.x, b.z).s as usize];
        let (gx, gz) = self.slope_at(b.x, b.z, 0.6);
        b.vx -= gx * GRAV * DT;
        b.vz -= gz * GRAV * DT;
        let sp = hypot2(b.vx, b.vz);
        if sp > 0.0 {
            let k = js_max(0.0, sp - p[2] * DT) / sp;
            b.vx *= k;
            b.vz *= k;
        }
        let cx = b.x - self.hole.pin.x;
        let cz = b.z - self.hole.pin.z;
        let cd = hypot2(cx, cz);
        if cd < LIP_R && cd > 0.01 {
            let u = cd / LIP_R;
            let dh = 4.0 * LIP_D * u * (1.0 - u * u) / LIP_R;
            b.vx -= dh * cx / cd * GRAV * DT;
            b.vz -= dh * cz / cd * GRAV * DT;
        }
        b.x += b.vx * DT;
        b.z += b.vz * DT;
        b.y = self.ground_at(b.x, b.z).h;
        self.roll_rest = sp < 0.4 && hypot2(gx, gz) * GRAV < p[2];
        sp
    }

    fn hazard_end(&mut self, s: u8) -> bool {
        if s < SURF_WATER {
            return false;
        }
        self.ball_air = false;
        self.ball_rolling = false;
        self.ball_event = s;
        true
    }

    pub fn ball_update(&mut self) {
        if self.ball_air {
            let x0 = self.ball.x;
            let z0 = self.ball.z;
            self.fly_step_ball();
            let g = self.ball_ground();
            if g.s < SURF_WATER {
                self.ball_safe = Pt {
                    x: self.ball.x,
                    z: self.ball.z,
                };
            }
            if self.ball.y > g.h {
                let pin = self.hole.pin;
                if !self.pin_hit
                    && !self.pin_out
                    && self.ball.y < g.h + POLE_H
                    && self.cup_hit(x0, z0, POST_R)
                    && (x0 - pin.x) * self.ball.vx + (z0 - pin.z) * self.ball.vz < 0.0
                {
                    self.pin_hit = true;
                    self.ball.vx *= -0.6;
                    self.ball.vz *= -0.6;
                }
                return;
            }
            self.ball.y = g.h;
            if self.hazard_end(g.s) {
                return;
            }
            if self.cup_hit(x0, z0, AIR_HOLE) {
                self.ball_air = false;
                self.ball_event = EV_HOLED;
                return;
            }
            let p = SURF_PHYS[g.s as usize];
            if self.ball.vy >= 0.0 {
                let (gx, gz) = self.slope_at(self.ball.x, self.ball.z, 0.5);
                let b = &mut self.ball;
                let k = (b.vy - b.vx * gx - b.vz * gz) / (gx * gx + 1.0 + gz * gz) * (1.0 + p[0]);
                if k < 0.0 {
                    self.bounces += 1;
                    b.vx = (b.vx + gx * k) * p[1];
                    b.vy = (b.vy - k) * p[1];
                    b.vz = (b.vz + gz * k) * p[1];
                }
                return;
            }
            self.bounces += 1;
            let mut keep = p[1];
            if self.bounces == 1 {
                keep *= 1.0 + self.ball_spin * 0.5;
                if g.s == SURF_GREEN && self.ball_spin < 0.0 {
                    let (ss, cs) = self.shot_sc;
                    self.ball.vx += ss * self.ball_spin * 3.0 / keep;
                    self.ball.vz += cs * self.ball_spin * 3.0 / keep;
                }
                self.ball_spin *= 0.3;
            }
            let (gx, gz) = self.slope_at(self.ball.x, self.ball.z, 0.5);
            let b = &mut self.ball;
            let m = (b.vy - b.vx * gx - b.vz * gz) / (gx * gx + 1.0 + gz * gz) * (keep + p[0]);
            b.vx = b.vx * keep + gx * m;
            b.vy = b.vy * keep - m;
            b.vz = b.vz * keep + gz * m;
            if b.vy < 1.6 {
                b.vy = 0.0;
                self.ball_air = false;
                self.ball_rolling = true;
            }
        } else if self.ball_rolling {
            self.ball_safe = Pt {
                x: self.ball.x,
                z: self.ball.z,
            };
            let x0 = self.ball.x;
            let z0 = self.ball.z;
            let mut b = self.ball;
            let sp = self.roll_step(&mut b);
            self.ball = b;
            if self.hazard_end(self.ball_ground().s) {
                return;
            }
            let hit = self.cup_hit(x0, z0, HOLE_R);
            if hit && sp < CUP_SPEED {
                self.ball_rolling = false;
                self.ball_event = EV_HOLED;
            } else if hit {
                self.ball_air = true;
                self.ball_rolling = false;
                self.ball.vy = sp * 0.35;
                self.ball.vx *= 0.93;
                self.ball.vz *= 0.93;
            } else if self.roll_rest {
                self.ball.vx = 0.0;
                self.ball.vz = 0.0;
                self.ball_rolling = false;
                self.ball_event = EV_STOPPED;
            }
        }
    }

    /// autoClub: the club the game pre-selects
    pub fn auto_club(&self) -> usize {
        let d = self.ball_to_pin();
        let s = self.ball_ground().s;
        if s == SURF_GREEN || (d < 19.0 && s != SURF_BUNKER && s != SURF_ROUGH) {
            return CLUB_PUTTER;
        }
        for i in (0..CLUB_PUTTER).rev() {
            if CLUBS[i].1 * self.lie_mul(i) / 1.15 >= d || i == 0 {
                return i;
            }
        }
        0
    }

    /// predictLanding for a full swing: the still-air flight to the ground.
    /// Returns (landing ball, hit: whether the arc clipped a rising face).
    pub fn predict_landing(&mut self, club: usize, dir: f64, spin: f64, lie_mul: f64, power: f64) -> (Ball, bool) {
        let mut b = Ball {
            x: self.ball.x,
            y: self.height_at(self.ball.x, self.ball.z) + 0.1,
            z: self.ball.z,
            ..Default::default()
        };
        Self::launch_vel(&mut b, club, dir, lie_mul, power, spin);
        self.ball_spin = spin;
        let mut hit = false;
        let mut d0 = 0.1;
        for _ in 0..600 {
            let x0 = b.x;
            let z0 = b.z;
            self.fly_step_free(&mut b, 0.0, 0.0);
            let d1 = b.y - self.height_at(b.x, b.z);
            if d1 <= 0.0 {
                if b.vy < 0.0 {
                    let t = d0 / (d0 - d1);
                    b.x = lerp(x0, b.x, t);
                    b.z = lerp(z0, b.z, t);
                    b.y = self.height_at(b.x, b.z);
                    break;
                }
                hit = true;
                b.y -= d1;
            }
            d0 = js_max(d1, 0.0);
        }
        (b, hit)
    }
}
