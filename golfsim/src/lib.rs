//! golfsim: Sunshine Golf Classic's physics (Golf13K) and the RL environment
//! in Rust, bit-identical to the JS game, with a C ABI for the Python trainer.
//!
//! - `jsmath`: JavaScript's numerics exactly as V8 computes them
//! - `game`:   course.js + golfSim.js
//! - `obs`:    rl/obs.mjs (observation, reference shot, action decoding)
//! - `env`:    rl/env.mjs + rl/courses.mjs (episodes and rules)
//! - `ffi`:    a batched, multi-threaded env for Python (py/golf/sim.py)

pub mod env;
pub mod ffi;
pub mod game;
pub mod jsmath;
pub mod obs;
