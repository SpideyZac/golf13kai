# golf13kai

A reinforcement learning agent that plays [Sunshine Golf Classic](https://github.com/KilledByAPixel/Golf13K) (Frank Force, JS13K 2026). It learns to pick a club, a spin, an aim line and a distance, and it holes out on courses it has never seen.

- **Pure Node.js, zero dependencies.** PPO, the MLP and backprop are hand-written, and the gradients are checked by finite differences in `npm test`.
- **The game's own physics.** The game's `golfSim.js` and `course.js` load straight from the `Golf13K` submodule, so the agent plays exactly what the browser runs.
- **Generalises.** It trains on endless procedural remix courses and is tested on the classic course and remix seeds it never saw.
- **Plays in the browser.** `npm run web`, then open `http://localhost:8013/rl/web/?auto=1` and watch it play the real game.

## Results

<!-- results table: kept in sync with docs/TRAINING.md -->
Strokes to par per 18-hole round (par 73), averaged over 20 rounds per column. Swing-meter noise is on, and every course is held out from training:

| Player | Classic course | Unseen remix courses |
|---|---|---|
| The game's default shot every time (its club, aim and target) | +7.3 | +12.8 |
| The game's scripted dev bot (`Golf13K/game/tools/sim.mjs`) | −2.9 | −1.1 |
| **RL agent** (`models/agent.json`, deterministic + escape rule) | **−9.9** | **−9.4** |

The agent makes birdie on about 58% of holes and bogey or worse on about 4%. It trained for 6M strokes (about 2.3 hours on 16 CPU cores).

![learning curve](docs/img/r4-learning-curve.svg)

It also found something in the game's balance: it plays the **driver for nearly every full shot**, down to 20-yard chips, dialing distance with the meter and holding greens with backspin. The game lets any club hit any distance under its maximum, and a club only picks a launch angle, so the driver with backspin works as a universal club.

## Setup

```sh
git clone --recursive <this repo>        # or: git submodule update --init
npm test
npm run baseline
npm run train -- --name myrun            # uses every core; ~18s per iteration on 16
npm run eval -- runs/myrun/best.json --card
npm run web                              # watch models/agent.json play
```

This needs Node ≥ 20. Nothing to install.

## How it works

One episode is one hole, and one step is one stroke. The reward is −1 per stroke (plus 1 per penalty), so the return is minus the score.

The agent observes what a player sees, in a frame pointed at the pin:

- the lie, the wind, the club the game suggests, and the slope
- terrain grids of hazards, sand, green, fairway, trees and height
- line-of-fire rays that show trees and walls
- the game's own aim previews, including the putt line with its break

It acts residually to the obvious shot. `aim = 0, dist = 0` means "at the pin, exactly that far", and it learns the wind, lay-ups, doglegs and break on top of that. Every swing goes through the game's `launchBall` with the same timing error the game's bot is given.

See [docs/SPEC.md](docs/SPEC.md) for the environment contract, [docs/DESIGN.md](docs/DESIGN.md) for the reasoning, and [docs/TRAINING.md](docs/TRAINING.md) for how to train and the results log.

## Layout

```
Golf13K/        the game (submodule, untouched)
rl/sim/         loads the game's physics headless; the shared accessor list
rl/obs.mjs      observation + action decoding (Node and browser)
rl/env.mjs      GolfEnv (episode = hole)
rl/nn.mjs       MLP, Adam
rl/policy.mjs   actor/critic, hybrid action distribution, PPO gradient
rl/train.mjs    PPO trainer over worker threads (rl/worker.mjs, rl/pool.mjs)
rl/eval.mjs     held-out evaluation; rl/baseline.mjs reference bots
rl/web/         the agent at the controls of the browser game
models/         the shipped agent
docs/           spec, design, training
```
