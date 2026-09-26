//! GolfEnv: one episode = one hole, one step = one stroke. A port of
//! rl/env.mjs (rules) and rl/courses.mjs (which holes are played); spec in
//! docs/SPEC.md.

use crate::game::*;
use crate::jsmath::{self as jm, hypot2};
use crate::obs::{self, Action, Prev, Reference, Shot, SolveCfg, OBS_DIM};
use crate::solver::Solution;

pub const CLASSIC_SEED: f64 = 1113.0;
pub const TRAIN_SEED_MIN: f64 = 1000.0;
pub const TRAIN_SEED_MAX: f64 = 2e6;

#[derive(Clone, Copy, Debug)]
pub struct EnvCfg {
    /// uniform +- swing meter timing error (the scripted bot's)
    pub impact_noise: f64,
    /// uniform +- radians on full swings (the scripted bot's)
    pub aim_noise: f64,
    /// the game's mercy rule: pick up at par + max_over
    pub max_over: i32,
    /// run the solver (solver.rs) in every observation
    pub solver: bool,
}

impl Default for EnvCfg {
    fn default() -> Self {
        EnvCfg {
            impact_noise: 0.04,
            aim_noise: 0.015,
            max_over: 5,
            solver: true,
        }
    }
}

/// reset spec: course seed, remix or classic, hole 0-17, the seed of the
/// instance's Math.random (wind, swing noise), and an optional exploring
/// start {u, v} in [0,1)^2
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub seed: f64,
    pub remix: bool,
    pub hole: usize,
    pub rng_seed: u32,
    pub start: Option<(f64, f64)>,
}

pub const RES_HOLED: u8 = 0;
pub const RES_STOPPED: u8 = 1;
pub const RES_WATER: u8 = 2;
pub const RES_OB: u8 = 3;
pub const RESULT_NAMES: [&str; 4] = ["holed", "stopped", "water", "ob"];

/// one entry of env.log
#[derive(Clone, Copy, Debug)]
pub struct ShotLog {
    pub club: usize,
    pub spin: f64,
    pub lie: u8,
    pub from: f64,
    pub want: f64,
    pub power: f64,
    pub result: u8,
    pub to: f64,
    pub tree: bool,
    /// the solver's shot was played
    pub solved: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct StepResult {
    pub reward: f64,
    pub done: bool,
    pub result: u8,
    pub strokes: i32,
    pub par: i32,
}

pub struct GolfEnv {
    pub g: Game,
    pub cfg: EnvCfg,
    pub strokes: i32,
    pub penalties: i32,
    pub done: bool,
    pub prev: Prev,
    pub spec: Spec,
    pub log: Vec<ShotLog>,
    pub reference: Option<Reference>,
    /// the solver's result in the last observe()
    pub solution: Solution,
}

impl GolfEnv {
    pub fn new(cfg: EnvCfg) -> GolfEnv {
        GolfEnv {
            g: Game::new(),
            cfg,
            strokes: 0,
            penalties: 0,
            done: true,
            prev: Prev::default(),
            spec: Spec {
                seed: CLASSIC_SEED,
                remix: false,
                hole: 0,
                rng_seed: 1,
                start: None,
            },
            log: Vec::new(),
            reference: None,
            solution: Solution::default(),
        }
    }

    /// Deals the hole. Call observe() for the first observation.
    pub fn reset(&mut self, spec: Spec) {
        let g = &mut self.g;
        g.rng = jm::Mulberry32::new(spec.rng_seed);
        g.remix_mode = spec.remix;
        let rows = Game::gen_course(spec.seed, spec.remix);
        g.gen_hole(spec.seed, spec.hole, &rows[spec.hole]);
        g.ball.x = 0.0;
        g.ball.z = 0.0;
        g.ball.vx = 0.0;
        g.ball.vy = 0.0;
        g.ball.vz = 0.0;
        if let Some((u, v)) = spec.start {
            self.place_ball(u, v);
        }
        let g = &mut self.g;
        g.ball.y = g.ground_at(g.ball.x, g.ball.z).h;
        g.ball_event = 0;
        self.strokes = 0;
        self.penalties = 0;
        self.done = false;
        self.log.clear();
        self.prev = Prev::default();
        self.spec = spec;
        self.reference = None;
        self.solution = Solution::default();
    }

    /// Exploring start: u along the centreline, (2v-1)*60 yards across it,
    /// shrinking toward the centreline until the spot is not water or OB.
    fn place_ball(&mut self, u: f64, v: f64) {
        let g = &mut self.g;
        let len = g.hole.len;
        let along = u * len * 0.97;
        let p = g.path_point_at(along);
        let q = g.path_point_at(jm::js_min(along + 2.0, len));
        let tl = {
            let t = hypot2(q.x - p.x, q.z - p.z);
            if t != 0.0 {
                t
            } else {
                1.0
            }
        };
        let nx = (q.z - p.z) / tl;
        let nz = -(q.x - p.x) / tl;
        let mut lat = (2.0 * v - 1.0) * 60.0;
        loop {
            let x = p.x + nx * lat;
            let z = p.z + nz * lat;
            if g.surface_at(x, z) < SURF_WATER {
                g.ball.x = x;
                g.ball.z = z;
                return;
            }
            if lat.abs() < 1.0 {
                return;
            }
            lat *= 0.6;
        }
    }

    pub fn pin_dist(&self) -> f64 {
        obs::pin_dist(&self.g)
    }

    /// The observation of the current state, into `o` (OBS_DIM floats).
    pub fn observe(&mut self, o: &mut [f32]) {
        let c = self.cfg;
        let sc = SolveCfg {
            impact_noise: c.impact_noise,
            aim_noise: c.aim_noise,
            on: c.solver,
        };
        let (r, sol) = obs::observe(&mut self.g, self.strokes, &self.prev, c.max_over, &sc, o);
        self.reference = Some(r);
        self.solution = sol;
    }

    pub fn observe_vec(&mut self) -> Vec<f32> {
        let mut o = vec![0.0f32; OBS_DIM];
        self.observe(&mut o);
        o
    }

    pub fn decode(&mut self, a: &Action) -> Shot {
        let r = match self.reference {
            Some(r) => r,
            None => {
                let r = obs::reference(&mut self.g);
                self.reference = Some(r);
                r
            }
        };
        obs::decode(&self.g, &r, &self.solution, a)
    }

    /// One stroke. The next observation is observe()'s job (not when done).
    pub fn step(&mut self, a: &Action) -> StepResult {
        assert!(!self.done, "step after done");
        let s = self.decode(a);
        let cfg = self.cfg;
        let g = &mut self.g;
        let putt = s.club == CLUB_PUTTER;
        let lie = g.ball_ground().s;
        let d0 = obs::pin_dist(g);
        let impact = (g.rng.next() * 2.0 - 1.0) * cfg.impact_noise;
        let yaw = s.yaw
            + if putt {
                0.0
            } else {
                (g.rng.next() * 2.0 - 1.0) * cfg.aim_noise
            };
        g.pin_out = d0 < 15.0 && lie == SURF_GREEN;
        g.tree_hit = false;
        g.launch_ball(s.club, s.power, impact, s.spin, yaw, s.lm);
        self.strokes += 1;
        let mut reward = -1.0;
        let mut t = 0;
        while t < 60 * 30 && g.ball_event == 0 {
            g.ball_update();
            t += 1;
        }
        let ev = if g.ball_event != 0 { g.ball_event } else { EV_STOPPED };
        g.ball_event = 0;
        let result = match ev {
            EV_HOLED => RES_HOLED,
            EV_WATER => RES_WATER,
            EV_OB => RES_OB,
            _ => RES_STOPPED,
        };
        if ev == EV_WATER || ev == EV_OB {
            self.strokes += 1;
            self.penalties += 1;
            reward -= 1.0;
            self.hazard_drop();
        }
        let g = &mut self.g;
        g.ball.vx = 0.0;
        g.ball.vy = 0.0;
        g.ball.vz = 0.0;
        self.prev = Prev {
            moved: hypot2(g.ball.x - g.shot_start.x, g.ball.z - g.shot_start.z),
            tree: g.tree_hit,
            hazard: result == RES_WATER || result == RES_OB,
        };
        let to = obs::pin_dist(g);
        self.log.push(ShotLog {
            club: s.club,
            spin: s.spin,
            lie,
            from: d0,
            want: s.want,
            power: s.power,
            result,
            to,
            tree: g.tree_hit,
            solved: s.solved,
        });
        if ev == EV_HOLED || self.strokes >= self.g.hole.par + cfg.max_over {
            self.done = true;
        }
        self.reference = None;
        self.solution = Solution::default();
        StepResult {
            reward,
            done: self.done,
            result,
            strokes: self.strokes,
            par: self.g.hole.par,
        }
    }

    /// Penalty drop, from game.js updateFlight: walk back along the shot from
    /// the last safe point until the ball can stay.
    fn hazard_drop(&mut self) {
        let g = &mut self.g;
        let safe = g.ball_safe;
        let start = g.shot_start;
        let dx = start.x - safe.x;
        let dz = start.z - safe.z;
        let dl = {
            let t = hypot2(dx, dz);
            if t != 0.0 {
                t
            } else {
                1.0
            }
        };
        let mut d = 2.0;
        loop {
            let t = jm::js_min(d, dl);
            g.ball.x = safe.x + dx / dl * t;
            g.ball.z = safe.z + dz / dl * t;
            let gr = g.ground_at(g.ball.x, g.ball.z);
            g.ball.y = gr.h;
            if t == dl
                || gr.s < SURF_WATER && gr.s != SURF_GREEN && {
                    let (sx, sz) = g.slope_at(g.ball.x, g.ball.z, 0.5);
                    hypot2(sx, sz) * GRAV < SURF_PHYS[gr.s as usize][2]
                }
            {
                break;
            }
            d += 2.0;
        }
    }

    pub fn holed(&self) -> bool {
        self.log.last().is_some_and(|l| l.result == RES_HOLED)
    }
}

/// courses.mjs trainEpisode: a random hole of a random remix course (seed >=
/// TRAIN_SEED_MIN), or of the classic course with classic_prob; start_prob of
/// them from a random spot. `rand` is any [0,1) source.
pub fn train_episode(rand: &mut impl FnMut() -> f64, classic_prob: f64, start_prob: f64) -> Spec {
    let classic = rand() < classic_prob;
    let mut seed = CLASSIC_SEED;
    if !classic {
        loop {
            seed = TRAIN_SEED_MIN + (rand() * (TRAIN_SEED_MAX - TRAIN_SEED_MIN)).floor();
            if seed != CLASSIC_SEED {
                break;
            }
        }
    }
    let hole = (rand() * 18.0).floor() as usize;
    let rng_seed = 1 + (rand() * 2147483648.0).floor() as u32;
    let start = if rand() < start_prob {
        Some((rand(), rand()))
    } else {
        None
    };
    Spec {
        seed,
        remix: !classic,
        hole,
        rng_seed,
        start,
    }
}

/// courses.mjs evalSet: 18-hole rounds `from .. from + rounds` of the classic
/// course (wind seed 1000(r+1)+hole) or of remix seed r+1.
pub fn eval_set(classic: bool, rounds: usize, from: usize) -> Vec<(usize, Spec)> {
    let mut v = Vec::new();
    for r in from..from + rounds {
        for h in 0..18 {
            let rng_seed = (1000 * (r + 1) + h) as u32;
            let spec = if classic {
                Spec {
                    seed: CLASSIC_SEED,
                    remix: false,
                    hole: h,
                    rng_seed,
                    start: None,
                }
            } else {
                Spec {
                    seed: (r + 1) as f64,
                    remix: true,
                    hole: h,
                    rng_seed,
                    start: None,
                }
            };
            v.push((r, spec));
        }
    }
    v
}
