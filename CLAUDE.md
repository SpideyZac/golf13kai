# CLAUDE.md

A reinforcement learning agent for **Sunshine Golf Classic**, Frank Force's JS13K 2026 golf game, which lives in the `Golf13K/` git submodule. The agent is pure Node.js: no Python, no npm dependencies, CPU only.

## Layout

- `Golf13K/`: the upstream game (submodule, **read-only**; do not edit or commit inside it). The agent loads its physics source directly.
- `rl/sim/loader.mjs`: loads `engineMath.js` + `course.js` + `golfSim.js` into an isolated closure per instance, and exposes game bindings through getters/setters on `G`.
- `rl/env.mjs`: `GolfEnv`, where one episode is one hole and one step is one stroke. Holds the observation builder and action decoding. **Spec: `docs/SPEC.md`.**
- `rl/nn.mjs`: MLP (tanh, `[in][out]` weight layout, zero-input skipping), Adam and grad clipping, all over one flat `Float32Array`.
- `rl/policy.mjs`: `Model` (actor + critic), the hybrid action distribution (club and spin categorical, aim and distance Gaussian), and `ppoRowGrad`.
- `rl/worker.mjs` / `rl/pool.mjs`: worker threads that do rollouts and gradient shards over a `SharedArrayBuffer` of params.
- `rl/train.mjs`: the PPO loop, CSV log, checkpoints in `runs/<name>/{best,last}.json`.
- `rl/eval.mjs`, `rl/baseline.mjs`, `rl/courses.mjs`: held-out evaluation, the reference bots, and the train/eval course split.
- `docs/`: `SPEC.md` (env contract), `TRAINING.md` (how to train, hyperparameters, results log), `DESIGN.md` (why things are the way they are).

## Commands

```sh
npm test                                   # gradient checks + env contract tests
npm run train -- --name r2 --iters 400     # ~18s/iter on 16 cores; logs to runs/r2/log.csv
npm run eval -- runs/r2/best.json --card   # classic + held-out remix rounds
npm run baseline                           # naive + scripted-bot reference scores
npm run eval -- runs/r2/best.json --set classic --rounds 1 --shots   # every shot of a round
```

## Rules of thumb

- **Never reimplement game physics.** Always call into `G` (the loaded game). If the game changes upstream, bump the submodule and re-check the baselines.
- **Changing the observation** (anything in `observe()` or the grid constants) means bumping `OBS_VERSION` and updating `docs/SPEC.md`. Old checkpoints then fail to load, and that is intended.
- **Game globals** (`let` bindings such as `hole`, `ball`, `ballEvent`, `pinOut`) need accessors in the loader's `EPILOGUE` before the env can touch them.
- **Keep the train/eval split honest.** Training seeds are ≥ 1000 remix seeds, and eval uses classic seed 1113 plus remix seeds 1…R.
- **After editing `nn.mjs` or `policy.mjs` math, run `npm test`.** It holds finite-difference checks for the backprop and the PPO gradient.
- **Performance.** A game instance inside `node:vm` was 20× slower, which is why it is a closure. A step costs about 7 ms (physics plus observation). A 256-sample fwd+bwd of both nets is about 0.5 s.
- **Commits** use Conventional Commits (`feat(rl): …`, `docs: …`, `fix(env): …`). `runs/` is git-ignored except for checkpoints deliberately copied to `models/`.
