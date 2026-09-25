# Training

## Quick start

```sh
npm test                                          # ~1s: backprop + env contract checks
npm run baseline                                  # reference scores (about 30s)
npm run train -- --name myrun --iters 300         # all cores; Ctrl-C is safe after an eval
npm run eval -- runs/myrun/best.json --card       # held-out classic + remix rounds
npm run web                                       # then open http://localhost:8013/rl/web/?auto=1
npm run build:web                                 # or: build/index.html, one self-contained file
```

Each run writes to `runs/<name>/`:

| File | Contents |
|---|---|
| `log.csv` | One row per iteration: training toPar per hole (a moving window of the last 2000 holes), penalties, pickups, PPO losses, entropy, KL, clip fraction, grad norm, lr, steps/s, and the eval columns when evaluated |
| `best.json` | The checkpoint with the best mean of classic and remix eval to par |
| `last.json` | Latest params plus Adam state. Resume with `--resume runs/<name>/last.json --iters <new total>`. |
| `config.json` | The exact arguments |

To ship a model to the browser page, copy it to `models/agent.json` (params only are needed; `last.json` also carries Adam state and is 3× larger).

## Hyperparameters (defaults in `rl/train.mjs`)

| Flag | Default | Notes |
|---|---|---|
| `--workers` | cores − 1 | Each worker owns one env and one rollout shard |
| `--steps` | 1024 | Env steps (strokes) per worker per iteration, about 15k per iteration on 16 cores |
| `--epochs` / `--mb` | 4 / 4 | 16 Adam steps per iteration. The KL early stop (`--target-kl` 0.03 × 1.5) cuts epochs short. |
| `--lr` | 3e-4 | Linear decay to `--lr-final` × lr (0.1) |
| `--clip` | 0.2 | PPO ratio clip |
| `--ent-cat` / `--ent-cont` | 0.01 / 0 | Entropy bonus on the club/spin heads, and on the Gaussian heads |
| `--vf` | 0.5 | Critic gradient scale (separate network) |
| `--gamma` / `--lambda` | 1 / 0.95 | Undiscounted: the return is minus the score |
| `--hidden` | 256,128 | Both actor and critic |
| `--start-prob` | 0.3 | Share of episodes that start from a random playable spot on the hole (exploring starts) instead of the tee |
| `--classic-prob` | 0 | Share of training episodes on the classic course. It is 0 so classic stays a held-out test. |
| `--eval-every` / `--eval-rounds` | 10 / 4 | Deterministic 18-hole rounds on each eval set |

## Reading the log

- **`toPar/hole`** is the training score per hole with a stochastic policy, so it runs worse than eval. Multiply by 18 for a round.
- **`ent`** is the summed entropy. It goes negative once the Gaussian stds shrink below 1, which is expected as putting sharpens.
- **`gn`** is the grad norm before clipping at 0.5. Values of about 1–10 are normal here, because the gradient is clipped every step.
- **Eval** uses the argmax club and spin and the mean aim and distance, with swing noise on. It is typically 1–2 strokes per round better than training.

## Results log

Newest first. Scores are strokes to par per 18-hole round. Rows before r4 are 4-round in-training evals (noisy, ±2–3 strokes), and r4 plus the baselines use 20 fresh rounds (`--from 4`).

| Run | Obs | Iters (steps) | Classic | Remix | Notes |
|---|---|---|---|---|---|
| _baseline_ default shot | — | — | +7.25 | +12.80 | Action (0,0) with the game's club, which is the game's own default aim and target |
| _baseline_ scripted | — | — | −2.85 | −1.05 | The game's dev bot, `Golf13K/game/tools/sim.mjs`. Its wind is unseeded. |
| **r4** | v3 | 400 (6.15M) | **−9.85** | **−9.40** | From scratch, lr 2.5e-4, 30% exploring starts, `--escape`. Without escape it scores −9.70 / −8.80, with 0.25 / 0.40 pickups per round. Shipped as `models/agent.json` (iter 390). |
| r3 | v2 | 70→103 | −7.0 | −5.75 | r2 resumed with exploring starts, lr 2e-4. Pickups remained on hairpins, where the pin-relative aim cannot follow the fairway. That motivated v3. |
| _(early)_ pin-aim naive | — | — | +27.3 | +41.5 | The game's club straight at the pin (rounds 0–3). Obsolete under v3. |
| r2 | v2 | 77 (1.19M) | −6.5 | −4.25 | Tee starts only. Deterministic triple+ on 4–7% of holes, stuck behind trees or walls. Stopped and resumed as r3. |
| r1 | v1 | 41 (630k) | −5.0 | −4.0 | Stopped early. Deterministic play could loop against a tree it could not see, which motivated obs v2. |

![r4 learning curve](img/r4-learning-curve.svg)

## Reproducing the shipped model

```sh
npm run train -- --name r4 --iters 400 --lr 2.5e-4
npm run eval -- runs/r4/best.json --rounds 20 --escape
npm run plot -- runs/r4/log.csv docs/img/r4-learning-curve.svg --scripted=-2.85,-1.05
cp runs/r4/best.json models/agent.json
```

Runs are not bit-reproducible, because worker scheduling orders the rollouts. Expect ±1 stroke.

## Known weaknesses and next steps

- **Tree lock-in.** A deterministic policy can still replay a shot into the same trunk until the mercy rule, about 0.2–0.4 times a round. The escape rule (`--escape`, always on in the browser) samples the policy after a stuck swing and mostly fixes it. A learned fix would be more exploring starts right behind trees, or an "attempts from this spot" feature.
- **Update size.** Mid-run KL reached 0.06–0.09 with a clip fraction of about 0.35, so the KL early stop cut most iterations to one epoch. A lower lr (1.5e-4) or `--target-kl 0.02` would likely train more smoothly.
- **Eval noise.** Four rounds per periodic eval swing ±2–3 strokes, and `best.json` is chosen on them. `--eval-rounds 8` costs about 30 s more per eval.
- **The driver.** The driver-for-everything strategy is legal in the game, but a designer might want the lofted clubs to matter more. See DESIGN.md.
