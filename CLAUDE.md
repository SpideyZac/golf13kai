# CLAUDE.md

A reinforcement learning agent for **Sunshine Golf Classic**, Frank Force's JS13K 2026 golf game, which lives in the `Golf13K/` git submodule. Training runs on a **bit-exact Rust port** of the game's physics (`golfsim/`) and **PyTorch** (`py/golfrl/`, GPU when present). The browser agent still runs the JS observation and network against the real game, and loads the JS-format checkpoints Python writes. The shipped agent (`models/agent.json`) scores about −10 a round on held-out courses, against −3 to −1 for the game's scripted bot.

## Layout

- `Golf13K/`: the upstream game (submodule, **read-only**; never edit or commit inside it).
- `golfsim/`: Rust crate (cdylib + rlib + `golfsim-cli`).
  - `src/jsmath.rs`: V8's numerics (fdlibm `sin`/`cos`/`log`/`log1p`/`tanh`, V8 `hypot`, LittleJS `RandomGenerator`, `mulberry32`).
  - `src/game.rs`: `course.js` + `golfSim.js`, line by line. Globals are `Game` fields.
  - `src/obs.rs`, `src/env.rs`, `src/solver.rs`: `rl/obs.mjs`, `rl/env.mjs` + `rl/courses.mjs`, and `rl/solver.mjs`.
  - `src/ffi.rs`: the batched, rayon-parallel C ABI Python uses (`info` columns `I_*`).
  - `src/bin/golfsim.rs`: `golfsim-cli serve` (JSON lines, for the parity test) and `bench`.
- `py/golfrl/`: `sim.py` (ctypes `VecEnv`), `model.py` (the JS model in torch + JS checkpoint I/O + the v3→v4 upgrade), `train.py` (PPO), `evaluate.py`, `selftest.py`, `upgrade.py`.
- `rl/sim/loader.mjs`: concatenates `engineMath.js` + `course.js` + `golfSim.js` into an isolated closure per instance (the JS reference env).
- `rl/sim/api.mjs`: `API_BODY`, the ONE list of game bindings (getters/setters), shared by the Node loader and the browser.
- `rl/obs.mjs`: `Observer`, which holds the observation, the reference (the game's default shot) and action decoding. Pure, no Node imports, because the browser runs it. **Spec: `docs/SPEC.md`.**
- `rl/solver.mjs`: the holing-shot solver `observe()` runs. It plays trial shots through the game's `launchBall`/`ballUpdate` and restores every flight global. `CLUB_SOLVE` (club 11) plays its shot.
- `rl/env.mjs`: `GolfEnv`, where one episode is one hole and one step is one stroke. Also rules (penalty drop, pin, mercy) and exploring starts.
- `rl/nn.mjs`, `rl/policy.mjs`: the MLP and the hybrid distribution the browser acts with (and the original Node trainer's backprop).
- `rl/train.mjs`, `rl/worker.mjs`, `rl/pool.mjs`, `rl/eval.mjs`, `rl/baseline.mjs`: the original pure-Node trainer and tools. They still work, about 18× slower.
- `rl/test/`: `grad.test.mjs` (finite differences), `env.test.mjs` (contract), `parity.test.mjs` (Rust vs JS, `===`).
- `rl/web/`: `index.html` (the game's script list plus `ai.mjs`), `ai.mjs` (replaces the dev bot's global `botSwing`), `serve.mjs`, and `build.mjs` (everything, model included, inlined into one `build/index.html`).
- `docs/`: `SPEC.md` (env contract), `TRAINING.md` (how to train, GPU notes, results log), `DESIGN.md` (why, history, findings).

## Commands

```sh
npm run build:sim                                   # cargo build --release in golfsim/ (needed by Python and the parity test)
npm test                                            # gradient checks + env contract + Rust/JS parity (~10 s)
PARITY_HOLES=1500 npm test                          # the thorough parity run (~2 min)
golfsim/target/release/golfsim-cli bench            # Rust speed
cd py && python -m golfrl.selftest                  # torch net == browser net, checkpoint round trip
cd py && python -m golfrl.train --name r5 --iters 400 --lr 2.5e-4   # ~12 min on 16 CPU cores; GPU auto
cd py && python -m golfrl.train --name solve1 --init ../models/agent.json --iters 300 --lr 1.5e-4   # fine-tune into the solver (v3 inits upgrade)
cd py && python -m golfrl.upgrade ../models/agent.json   # v3 checkpoint -> v4, SOLVE never picked (plays as before)
cd py && python -m golfrl.evaluate ../runs/r5/best.json --rounds 20 --escape
npm run eval -- runs/r5/best.json --rounds 20 --escape   # the JS eval reads Python checkpoints too
npm run baseline -- --rounds 20                     # default-shot + scripted bot references
npm run plot -- runs/r5/log.csv docs/img/r5.svg
npm run web                                         # http://localhost:8013/rl/web/?auto=1
npm run build:web                                   # build/index.html: one file, open from disk, auto-plays
```

## Rules of thumb

- **The Rust port must stay bit-exact.** Any change to `golfsim/src/{game,obs,env,solver,jsmath}.rs`, or a submodule bump, is checked with `npm run build:sim && npm test` (and `PARITY_HOLES=1500` before shipping). Keep JS operand order and RandomGenerator draw order, route every `Math.*` through `jsmath`, and never use Rust's `f64::sin` and friends in the sim. Only change game rules in the JS (upstream) first, then port.
- **Changing `observe()` or `decode()` semantics** means changing `rl/obs.mjs` AND `golfsim/src/obs.rs` (the solver, `rl/solver.mjs` AND `golfsim/src/solver.rs`, is part of both), bumping `OBS_VERSION` in both, and updating `docs/SPEC.md`. Old checkpoints then refuse to load, and that is intended. A shipped model must be retrained after such a change.
- **The model format is the JS one.** `py/golfrl/model.py` must mirror `rl/policy.mjs` (head layout, log-std clamp, init). `python -m golfrl.selftest` checks it against node.
- **Keep the train/eval split honest.** Training uses remix seeds ≥ 1000; eval uses classic 1113 plus remix seeds 1…R. The trainer selects `best.json` on rounds 0–3, so final reports use `--from 4` (the default in both evals).
- **After editing `nn.mjs` or `policy.mjs` math, run `npm test`.** It holds finite-difference checks.
- **The solver runs on the live game in the browser.** It must restore every global a shot writes (`save`/`restore` in `rl/solver.mjs`, `Game::save_flight` in Rust), and those need setters in `rl/sim/api.mjs`. If the submodule adds flight state, add it there too.
- **Rule mirrors.** `GolfEnv.hazardDrop` and the pin rule mirror `Golf13K/game/game.js`, and `Observer.reference()` mirrors `aimDefault`. Re-check them (in JS and Rust) if the submodule is bumped.
- **Performance.** The envs saturate every core (rayon), so don't run evals alongside training unless you accept the slowdown. The Python trainer's time splits about evenly between play and update on CPU; a GPU takes most of the update.
- **No doctype on the web pages.** The engine sizes its canvas from `body.clientHeight`, which is the viewport only in quirks mode. With `<!doctype html>` the canvas is 0px tall and the screen is black while the game keeps playing (`build.mjs` refuses to emit one). Headless Chrome screenshots work (`--use-angle=swiftshader`), and the console log (`AI …`, `SHOT …`, `RESULT …`) shows every shot.
- **Commits** use Conventional Commits (`feat(sim): …`, `feat(py): …`, `docs: …`, `fix(env): …`). `runs/` and `golfsim/target/` are git-ignored, so copy a checkpoint to `models/` to ship it.
