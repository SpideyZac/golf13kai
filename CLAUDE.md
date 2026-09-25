# CLAUDE.md

A reinforcement learning agent for **Sunshine Golf Classic**, Frank Force's JS13K 2026 golf game, which lives in the `Golf13K/` git submodule. The agent is pure Node.js: no Python, no npm dependencies, CPU only. The shipped agent (`models/agent.json`) scores about −10 a round on held-out courses, against −3 to −1 for the game's scripted bot.

## Layout

- `Golf13K/`: the upstream game (submodule, **read-only**; never edit or commit inside it). The agent loads its physics source directly.
- `rl/sim/loader.mjs`: concatenates `engineMath.js` + `course.js` + `golfSim.js` into an isolated closure per instance.
- `rl/sim/api.mjs`: `API_BODY`, the ONE list of game bindings (getters/setters), shared by the Node loader and the browser.
- `rl/obs.mjs`: `Observer`, which holds the observation, the reference (the game's default shot) and action decoding. Pure, no Node imports, because the browser runs it too. **Spec: `docs/SPEC.md`.**
- `rl/env.mjs`: `GolfEnv`, where one episode is one hole and one step is one stroke. Also rules (penalty drop, pin, mercy) and exploring starts.
- `rl/nn.mjs`: MLP (tanh, `[in][out]` weights, zero-input skipping), Adam and grad clipping over one flat `Float32Array`.
- `rl/policy.mjs`: `Model` (actor + critic), the hybrid distribution (club and spin categorical, aim and dist Gaussian), and `ppoRowGrad`.
- `rl/worker.mjs` / `rl/pool.mjs`: worker threads doing rollouts and gradient shards over a `SharedArrayBuffer` of params.
- `rl/train.mjs`: the PPO loop, writing `runs/<name>/{log.csv,best.json,last.json,config.json}`.
- `rl/eval.mjs`, `rl/baseline.mjs`, `rl/courses.mjs`, `rl/plot.mjs`: evaluation, reference bots, the train/eval split, and learning-curve SVGs.
- `rl/web/`: `index.html` (the game's script list plus `ai.mjs`), `ai.mjs` (replaces the dev bot's global `botSwing`), and `serve.mjs`.
- `docs/`: `SPEC.md` (env contract), `TRAINING.md` (how to train, results log, known weaknesses), `DESIGN.md` (why, history, findings).

## Commands

```sh
npm test                                            # gradient checks + env contract (~2s)
npm run train -- --name r5 --iters 400 --lr 2.5e-4  # ~20s/iter on 16 cores, ~2.3h total
npm run eval -- runs/r5/best.json --rounds 20 --escape   # 20 fresh rounds per set
npm run eval -- --set classic --rounds 1 --shots    # every shot of one round (models/agent.json)
npm run baseline -- --rounds 20                     # default-shot + scripted bot references
npm run plot -- runs/r5/log.csv docs/img/r5.svg
npm run web                                         # http://localhost:8013/rl/web/?auto=1
```

## Rules of thumb

- **Never reimplement game physics.** Call into `G`. If a new game binding is needed, add it to `API_BODY` in `rl/sim/api.mjs`, which covers both hosts.
- **Changing `observe()` or `decode()` semantics** means bumping `OBS_VERSION` and updating `docs/SPEC.md`. Old checkpoints then refuse to load, and that is intended. A shipped model must be retrained after such a change.
- **Keep the train/eval split honest.** Training uses remix seeds ≥ 1000; eval uses classic 1113 plus remix seeds 1…R. The trainer selects `best.json` on rounds 0–3, so final reports use `--from 4` (the default in eval/baseline).
- **After editing `nn.mjs` or `policy.mjs` math, run `npm test`.** It holds finite-difference checks.
- **Rule mirrors.** `GolfEnv.hazardDrop` and the pin rule mirror `Golf13K/game/game.js`, and `Observer.reference()` mirrors `aimDefault`. Re-check them if the submodule is bumped.
- **Performance.** A game instance in `node:vm` was 20× slower, which is why it is a closure. A step costs about 7 ms, and a 256-sample fwd+bwd of both nets about 0.5 s. Training saturates every core, so don't run evals alongside it unless you accept the slowdown.
- **Headless screenshots.** Headless Chrome screenshots of the WebGL game come out black. Verify the browser agent from its console log (`AI …`, `SHOT …`, `RESULT …`) instead.
- **Commits** use Conventional Commits (`feat(rl): …`, `docs: …`, `fix(env): …`). `runs/` is git-ignored, so copy a checkpoint to `models/` to ship it.
