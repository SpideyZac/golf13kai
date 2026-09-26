# golf13kai

A reinforcement learning agent that plays [Sunshine Golf Classic](https://github.com/KilledByAPixel/Golf13K) (Frank Force, JS13K 2026). It learns to pick a club, a spin, an aim line and a distance, and it holes out on courses it has never seen.

- **The game's physics, in Rust, bit for bit.** `golfsim/` ports the game's `course.js` and `golfSim.js` (and the observation) to Rust, down to V8's own `sin`, `log` and `hypot`. A parity test plays the same holes in the real JS game and in Rust and requires every ball position and every observation float to be identical, and it is: 1500 holes, 6145 strokes. About 25× faster than the JS per stroke, and it runs on every core.
- **PyTorch training, GPU ready.** `py/golfrl` is PPO on the GPU (or CPU), stepping 512 Rust envs in parallel through a C ABI. A 400-iteration run (6.5M strokes) takes 12 minutes on a 16-core CPU with no GPU, against 2.3 hours for the original pure-Node trainer, and matches its agent.
- **Generalises.** It trains on endless procedural remix courses and is tested on the classic course and remix seeds it never saw.
- **Plays in the browser.** Checkpoints are saved in the JS format, so the browser agent loads them unchanged: `npm run build:web` makes `build/index.html`, one file that plays the real game from disk.

## Results

<!-- results table: kept in sync with docs/TRAINING.md -->
Strokes to par per 18-hole round (par 73), averaged over 20 rounds per column. Swing-meter noise is on, and every course is held out from training:

| Player | Classic course | Unseen remix courses |
|---|---|---|
| The game's default shot every time (its club, aim and target) | +7.3 | +12.8 |
| The game's scripted dev bot (`Golf13K/game/tools/sim.mjs`) | −2.9 | −1.1 |
| **RL agent** (`models/agent.json`, deterministic + escape rule) | **−9.9** | **−9.4** |

The agent makes birdie on about 58% of holes and bogey or worse on about 4%. It trained for 6M strokes (about 2.3 hours on 16 CPU cores, with the original Node trainer; the Python trainer reproduces it in 12 minutes on the same CPU, see py1 in docs/TRAINING.md).

![learning curve](docs/img/r4-learning-curve.svg)

It also found something in the game's balance: it plays the **driver for nearly every full shot**, down to 20-yard chips, dialing distance with the meter and holding greens with backspin. The game lets any club hit any distance under its maximum, and a club only picks a launch angle, so the driver with backspin works as a universal club.

## Setup

```sh
git clone --recursive <this repo>        # or: git submodule update --init
npm run build:sim                        # the Rust env (needs Rust: https://rustup.rs)
pip install -r py/requirements.txt       # PyTorch + numpy; the CUDA build of torch for a GPU
npm test                                 # includes the Rust-vs-JS parity test

cd py
python -m golfrl.selftest                # PyTorch network == the browser's network
python -m golfrl.train --name myrun      # uses the GPU if there is one
python -m golfrl.evaluate ../runs/myrun/best.json --rounds 20 --escape
cd ..
cp runs/myrun/best.json models/agent.json
npm run build:web                        # build/index.html: the agent playing the real game, one file
npm run web                              # or watch it at http://localhost:8013/rl/web/?auto=1
```

Node ≥ 20, Rust (stable) and Python ≥ 3.9 with PyTorch ≥ 2.1. See [docs/TRAINING.md](docs/TRAINING.md) for GPU notes and the flags.

## How it works

One episode is one hole, and one step is one stroke. The reward is −1 per stroke (plus 1 per penalty), so the return is minus the score.

The agent observes what a player sees, in a frame pointed at the pin:

- the lie, the wind, the club the game suggests, and the slope
- terrain grids of hazards, sand, green, fairway, trees and height
- line-of-fire rays that show trees and walls
- the game's own aim previews, including the putt line with its break

When the pin is in range, a solver plays trial shots through the real physics until it finds the swing that holes, wind included, and works out how often that swing drops given the swing-meter error. The agent sees those odds and can choose `SOLVE` to play the solver's swing, so its own job becomes strategy: when to go for it, and where to leave the ball so the next one is makeable.

For everything else it acts residually to the obvious shot, and it chooses the swing-meter impact too, so it can hook or slice (the solver curves shots around trees the same way). `aim = 0, dist = 0` means "at the pin, exactly that far", and it learns the wind, lay-ups, doglegs and break on top of that. Every swing goes through the game's `launchBall` with the same timing error the game's bot is given.

See [docs/SPEC.md](docs/SPEC.md) for the environment contract, [docs/DESIGN.md](docs/DESIGN.md) for the reasoning, and [docs/TRAINING.md](docs/TRAINING.md) for how to train and the results log.

## Layout

```
Golf13K/        the game (submodule, untouched)
golfsim/        the game's physics + the env in Rust (bit-exact), C ABI, parity CLI
py/golfrl/      PyTorch: PPO trainer, evaluation, the ctypes env, checkpoint I/O
rl/sim/         loads the game's physics headless; the shared accessor list
rl/obs.mjs      observation + action decoding (Node and browser)
rl/solver.mjs   the holing-shot solver (Node and browser; golfsim/src/solver.rs is its twin)
rl/env.mjs      GolfEnv (episode = hole), the reference the Rust port is tested against
rl/nn.mjs       MLP, Adam
rl/policy.mjs   actor/critic, hybrid action distribution, PPO gradient
rl/train.mjs    the original pure-Node PPO trainer (rl/worker.mjs, rl/pool.mjs)
rl/eval.mjs     held-out evaluation; rl/baseline.mjs reference bots
rl/test/        gradient checks, env contract, Rust parity
rl/web/         the agent at the controls of the browser game
models/         the shipped agent
docs/           spec, design, training
```
