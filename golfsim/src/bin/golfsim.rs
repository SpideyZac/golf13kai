//! golfsim CLI.
//!
//!   golfsim serve        JSON lines on stdin/stdout, one env: the parity test
//!                        (rl/test/parity.test.mjs) drives it next to the JS env
//!   golfsim bench [N]    speed: N holes of the default shot, one thread, then
//!                        a batch on every core
//!
//! serve requests, one per line; every reply is one line:
//!   {"cmd":"reset","spec":{seed,remix,hole,rngSeed,start?:{u,v}},"detail":bool,
//!    "cfg"?:{impactNoise,aimNoise,exactSolve}}
//!       -> {obs, hole:{...}} (detail adds every prop: near [[x,z,s,y]...]);
//!       cfg (optional) replaces the env's settings from here on
//!   {"cmd":"step","action":{club,spin,aim,dist,impact?}}
//!       -> {reward, done, result, strokes, penalties, ball, prev, shot, obs|null}
//!   {"cmd":"math","x":[..],"y":[..]} -> each jsmath function over x (and y)
use golfsim::env::*;
use golfsim::game::*;
use golfsim::jsmath as jm;
use golfsim::obs::{Action, OBS_DIM};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::time::Instant;

fn obs_json(o: &[f32]) -> Value {
    Value::Array(o.iter().map(|&v| json!(v as f64)).collect())
}

fn hole_json(g: &Game, detail: bool) -> Value {
    let h = &g.hole;
    let mut j = json!({
        "par": h.par, "len": h.len, "gr": h.gr, "noiseSeed": g.noise_seed,
        "pin": [h.pin.x, h.pin.z], "green": [h.green.x, h.green.z],
        "greenH": h.green_h, "teeH": h.tee_h, "wind": [h.wind_a, h.wind_s],
        "path": h.path.iter().map(|p| json!([p.x, p.z, p.along])).collect::<Vec<_>>(),
        "bunkers": h.bunkers.iter().map(|b| json!([b.x, b.z, b.rx, b.rz])).collect::<Vec<_>>(),
        "waters": h.waters.iter().map(|w| json!([w.x, w.z, w.rx, w.rz, w.h])).collect::<Vec<_>>(),
        "nNear": h.near.len(),
        "ball": [g.ball.x, g.ball.y, g.ball.z],
    });
    if detail {
        j["near"] = h.near.iter().map(|t| json!([t.x, t.z, t.s, t.y])).collect();
    }
    j
}

fn spec_from(j: &Value) -> Spec {
    let start = j
        .get("start")
        .filter(|s| !s.is_null())
        .map(|s| (s["u"].as_f64().unwrap(), s["v"].as_f64().unwrap()));
    Spec {
        seed: j["seed"].as_f64().unwrap(),
        remix: j["remix"].as_bool().unwrap_or(false),
        hole: j["hole"].as_u64().unwrap() as usize,
        rng_seed: j["rngSeed"].as_u64().unwrap() as u32,
        start,
    }
}

fn serve() {
    let mut env = GolfEnv::new(EnvCfg::default());
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = serde_json::from_str(&line).expect("bad request");
        let reply = match req["cmd"].as_str().unwrap() {
            "reset" => {
                if let Some(c) = req.get("cfg").filter(|c| !c.is_null()) {
                    let d = EnvCfg::default();
                    env.cfg = EnvCfg {
                        impact_noise: c["impactNoise"].as_f64().unwrap_or(d.impact_noise),
                        aim_noise: c["aimNoise"].as_f64().unwrap_or(d.aim_noise),
                        exact_solve: c["exactSolve"].as_bool().unwrap_or(d.exact_solve),
                        ..d
                    };
                }
                env.reset(spec_from(&req["spec"]));
                let o = env.observe_vec();
                json!({"obs": obs_json(&o), "hole": hole_json(&env.g, req["detail"].as_bool().unwrap_or(false))})
            }
            "step" => {
                let a = &req["action"];
                let act = Action {
                    club: a["club"].as_u64().unwrap() as usize,
                    spin: a["spin"].as_u64().unwrap() as usize,
                    aim: a["aim"].as_f64().unwrap(),
                    dist: a["dist"].as_f64().unwrap(),
                    impact: a["impact"].as_f64().unwrap_or(0.0),
                };
                let r = env.step(&act);
                let l = *env.log.last().unwrap();
                let obs = if r.done {
                    Value::Null
                } else {
                    obs_json(&env.observe_vec())
                };
                let b = env.g.ball;
                json!({
                    "reward": r.reward, "done": r.done, "result": RESULT_NAMES[r.result as usize],
                    "strokes": env.strokes, "penalties": env.penalties,
                    "ball": [b.x, b.y, b.z],
                    "prev": [env.prev.moved, env.prev.tree as u8, env.prev.hazard as u8],
                    "shot": {"club": CLUBS[l.club].0, "spin": l.spin, "lie": SURF_NAMES[l.lie as usize],
                             "from": l.from, "want": l.want, "power": l.power, "to": l.to, "tree": l.tree as u8,
                             "solved": l.solved as u8, "impact": l.impact},
                    "obs": obs,
                })
            }
            "math" => {
                let x: Vec<f64> = req["x"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let y: Vec<f64> = req["y"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect();
                let map = |f: &dyn Fn(f64) -> f64| x.iter().map(|&v| f(v)).collect::<Vec<_>>();
                let map2 = |f: &dyn Fn(f64, f64) -> f64| x.iter().zip(&y).map(|(&a, &b)| f(a, b)).collect::<Vec<_>>();
                json!({
                    "sin": map(&jm::sin), "cos": map(&jm::cos), "tanh": map(&jm::tanh),
                    "exp": map(&jm::exp), "log": map(&|v| jm::log(v.abs())), "log1p": map(&|v| jm::log1p(v.abs())),
                    "atan2": map2(&jm::atan2), "hypot2": map2(&jm::hypot2),
                    "hypot3": map2(&|a, b| jm::hypot3(a, b, a - b)),
                })
            }
            c => panic!("unknown cmd {c}"),
        };
        serde_json::to_writer(&mut out, &reply).unwrap();
        out.write_all(b"\n").unwrap();
        out.flush().unwrap();
    }
}

fn bench(holes: usize) {
    let mut env = GolfEnv::new(EnvCfg::default());
    let mut o = vec![0.0f32; OBS_DIM];
    let (mut steps, mut t_reset, mut t_obs, mut t_step) = (0usize, 0.0, 0.0, 0.0);
    let mut strokes_over = 0i64;
    for i in 0..holes {
        let t = Instant::now();
        env.reset(Spec {
            seed: 1000.0 + i as f64,
            remix: true,
            hole: i % 18,
            rng_seed: 1 + i as u32,
            start: None,
        });
        t_reset += t.elapsed().as_secs_f64();
        loop {
            let t = Instant::now();
            env.observe(&mut o);
            t_obs += t.elapsed().as_secs_f64();
            let club = env.reference.unwrap().club;
            let t = Instant::now();
            let r = env.step(&Action {
                club,
                spin: 1,
                aim: 0.0,
                dist: 0.0,
                impact: 0.0,
            });
            t_step += t.elapsed().as_secs_f64();
            steps += 1;
            if r.done {
                strokes_over += (env.strokes - env.g.hole.par) as i64;
                break;
            }
        }
    }
    println!(
        "{holes} holes, {steps} strokes (default shot, {:+.2}/hole): reset {:.2} ms, observe {:.3} ms, stroke {:.3} ms",
        strokes_over as f64 / holes as f64,
        t_reset * 1e3 / holes as f64,
        t_obs * 1e3 / steps as f64,
        t_step * 1e3 / steps as f64
    );
    // the batched path, as Python drives it
    let n = 256u32;
    let h = golfsim::ffi::gs_vec_new(n, 0, 0.04, 0.015, 5, 7, 0.0, 0.3, 1, 0);
    let mut obs = vec![0.0f32; n as usize * OBS_DIM];
    let mut info = vec![0.0f64; n as usize * golfsim::ffi::INFO_W];
    let club = vec![0i32; n as usize];
    let spin = vec![1i32; n as usize];
    let zero = vec![0.0f64; n as usize];
    unsafe {
        golfsim::ffi::gs_vec_reset_train(h, obs.as_mut_ptr());
        let t = Instant::now();
        let iters = 20;
        for _ in 0..iters {
            golfsim::ffi::gs_vec_step(
                h,
                club.as_ptr(),
                spin.as_ptr(),
                zero.as_ptr(),
                zero.as_ptr(),
                zero.as_ptr(),
                std::ptr::null(),
                1,
                obs.as_mut_ptr(),
                info.as_mut_ptr(),
            );
        }
        let dt = t.elapsed().as_secs_f64();
        println!(
            "batch of {n} on {} threads: {:.0} strokes/s (driver off every lie, auto-reset)",
            rayon::current_num_threads(),
            (n as usize * iters) as f64 / dt
        );
        golfsim::ffi::gs_vec_free(h);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("serve") => serve(),
        Some("bench") => bench(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200)),
        _ => {
            eprintln!("usage: golfsim serve | golfsim bench [holes]");
            std::process::exit(2);
        }
    }
}
