# Design notes

Why the agent is built the way it is. The environment contract is in [SPEC.md](SPEC.md), and running and results are in [TRAINING.md](TRAINING.md).

## Constraints that shaped it

- **The game is upstream code** (a submodule), and it is not modified. `rl/sim/loader.mjs` concatenates `engineMath.js`, `course.js` and `golfSim.js`, the same three files the game's own `tools/sim.mjs` uses, and wraps them in a `Function`. Game `let` bindings are reached through accessors.
- **`node:vm` was tried first and dropped.** Global-binding lookups through a vm context made a shot about 20× slower (44 ms against 2–4 ms). A closure keeps every game binding function-local. `Math` is shadowed per instance so `Math.random` (wind, swing noise) can be seeded without touching the host.
- **The browser runs the agent against the real game.** Whatever trains it, the policy has to be the JS `Model` (`rl/policy.mjs`) over the JS observation (`rl/obs.mjs`), reading a JS-format checkpoint.

The first version was pure Node.js with zero dependencies: hand-written MLP and backprop, PPO over worker threads, and the game's physics run from its own source. That produced the shipped r4 agent in 2.3 hours on 16 cores. The training stack then moved to **Rust for the simulation and PyTorch for the learning**, to train on a GPU and to make the env fast enough to feed one.

## The Rust port (`golfsim/`)

The rule that the agent must play exactly what the browser runs still holds. It is now enforced by a test rather than by loading the source: `rl/test/parity.test.mjs` plays the same holes through the JS env (the game's own code) and through Rust, and requires `===` on every hole layout, every ball position and every observation float (1383 in v3, 1389 in v4) after every stroke. Tolerances would have let porting bugs hide behind "float noise". Bit-exactness leaves no room for them, and it held over 1500 holes and 6145 strokes, with 731 tree strikes, 131 splashes and 175 OB.

Getting there took three things:

- **V8's math, not the platform's.** `hashN` feeds `Math.sin` of arguments up to ~10⁵, amplified by 43758, into every height and surface, so a single ulp moves the terrain. Rust's `f64::sin` (the C runtime) disagrees with Node on ~2% of inputs, and the `libm` crate on ~1%. A correctly rounded sin disagrees on ~3%, which shows V8 is not correctly rounded either. Node's V8 is built without its optional glibc trig, so `Math.sin`/`cos` are V8's fdlibm, and `jsmath.rs` ports those, plus fdlibm's `log`, `log1p`, `expm1` and `tanh`, and V8's own `Math.hypot` (normalise by the max, Kahan-sum). `exp` and `atan2` from the `libm` crate already match. All of them agree with Node on 2M random inputs and in the parity test.
- **JS evaluation order.** Every `RandomGenerator` draw happens in the JS order, short-circuits included (`!index || R.bool(.5)` skips a draw on hole 1), and every float expression keeps its left-to-right operand order. Globals that persist between episodes in JS (`shotDir`, `ballSpin`, …) persist in Rust too.
- **Exact number I/O in the test harness.** `serde_json` parses floats approximately unless its `float_roundtrip` feature is on, and that alone produced 1-ulp "physics" differences.

It is faster than the JS without changing a result. The value-noise lattice hashes (one `sin` each) are cached per hole. The ball's tree test reads a grid of the props that can touch the step, and tests those in the original list order, so the first strike is the same tree. Non-colliding props (far trees, wildflowers) are never generated: the wildflowers are the last draws from the hole's generator, so skipping them shifts nothing. One stroke costs ~0.08 ms and an observation ~0.14 ms, against ~7 ms for a JS step, and 256 envs on 16 threads play ~24k strokes a second.

## The Python trainer (`py/golfrl/`)

- The env is a C ABI over the Rust crate (`golfsim/src/ffi.rs`), loaded with ctypes: a batch of envs stepped in parallel on a rayon pool, auto-resetting finished ones to new training episodes drawn in Rust. The only thing crossing the boundary is arrays.
- The model is the JS model in PyTorch: the same layers, init, tanh, hybrid distribution and log-std clamp. Checkpoints are written in the JS format (flat float32, `[in][out]` per layer, actor then critic). `golfrl.selftest` checks the forward pass and log-probabilities against `rl/policy.mjs` on real observations, and the shipped r4 model, evaluated through PyTorch and Rust, reproduces its JS eval scores exactly (−9.70 / −8.80 without escape).
- PPO is the JS trainer's: the same losses, clip, entropy terms, global grad clip, LR decay and KL early stop. It uses vectorised envs with a fixed number of strokes each per iteration instead of whole episodes per worker, so an unfinished episode is bootstrapped from the critic.

## Problem framing

- **One episode per hole, one step per stroke.** Holes are short: about 4 steps, and at most par+5 plus a penalty. The reward is −1 per stroke plus penalties, with γ = 1, so the return is exactly minus the score. There is no reward shaping. The critic learns expected strokes-to-hole from the state, which is the natural value function of golf.
- **Procedural training distribution.** Every training episode is a random hole of a random remix course, and there are about 2M × 18 layouts. The classic course and the low remix seeds are held out, so eval numbers measure generalisation, not memorisation.

## Action parameterisation

The raw action is **residual to the game's default shot**. `aim = 0, dist = 0` means the aim line and target the game sets up for a player (`aimDefault`): the pin when the suggested club can reach it, otherwise a lay-up one carry down the centreline. The network learns the corrections: wind, lines, break, and pace over or under. (v1–v2 were residual to the pin, which failed on hairpins. See below.)

- The club is absolute, a categorical over 11, plus (v4) a 12th choice, `SOLVE`, that hands the swing to the solver (below).
- (v5) The swing-meter impact is chosen too. In the game a mistimed swing is also how you shape the ball: it pushes the start line, curves the flight and costs a little power. Treating it purely as noise took the hook and slice away from both the agent and the solver. The env's noise now adds to the impact swung for. Swinging straight keeps the game's perfect-strike snap (|impact| < 0.02), so shaping has a real cost to weigh. The game's own pre-selected club is in the observation, so "use the suggested club" is a single linear feature to pick up.
- Aim is ±60° around the default line, so escapes are reachable, and the state-dependent log std lets putts get sub-degree precise while drives stay loose.
- Distance is multiplicative (`exp`) because shot lengths span 1 to 600 yards.
- Everything goes through the game's `launchBall` with swing-meter timing noise. The agent never gets a perfect strike for free, the same deal the scripted bot gets.

## Observation

The player's view, in two frames: the default aim line for everything about this shot (grid B, rays, flight preview, centreline), and the pin line for the hole overview (grid A) and the putt preview.

- **Two terrain grids.** One is scaled to the pin distance: on a 6-yard putt it samples the green's contour at sub-yard spacing, and on a drive it spans the hole. The other is fixed in yards over full-swing landing zones. Channels: hazard, sand, green, short grass, tree canopy, relative height.
- **The game's own aim previews.** The putt line with its break at two paces, and the still-air flight ring of the suggested club. A human reads these off the screen, so they are fair, and they turn putting from "infer the break from heights" into "correct the previewed miss".
- **v2 line-of-fire rays and last-shot memory.** These fix a deterministic loop in which v1 kept striking a tree it could not resolve.

## The solver (v4)

The goal was an agent that holes long shots by reading the wind perfectly, trained on nothing but −1 per stroke. Reward shaping for hero shots (a trainer-side bonus, briefly tried as `--hero`) pushed toward risk without teaching precision. So precision became a tool: a solver that computes the holing swing, and an action that uses it. The policy's job becomes strategy, deciding when a solver shot is worth it and which positions give it good odds.

- **Solve by simulation, not by model.** The physics is deterministic given the hole's wind: nothing in `ballUpdate` draws a random number. Trial shots through the real `launchBall`/`ballUpdate` are exact, and a shooting method converges in a few trials. It turns the aim by the angle of the miss and scales the target by distance ratio, aiming for the ball to *stop* at the pin, which crosses the cup at crawling pace. Bounces and slopes make the map non-smooth, so it sometimes fails, and then it says so.
- **Honest odds.** A noise-free solution is not a promise, because the swing meter still errs. Each solution is replayed at weighted noise samples. The first version sampled aim at 0 as well, which is the solution itself, so it credited every 150-yard iron with a third of a chance. The aim samples now avoid 0.
- **An action, not an override.** The env never plays the solver on its own. `SOLVE` is a 12th club logit, and the solver's result is in the observation, so PPO learns when it pays. No action masking is needed: `SOLVE` without a solution falls back to the game's club.
- **One solver, two hosts, bit for bit.** The observation includes the solver's numbers, so `rl/solver.mjs` (browser, JS env) and `golfsim/src/solver.rs` (training) must agree exactly, and the parity test plays a quarter of its shots as `SOLVE`. In the browser it runs on the live game between frames. It saves and restores every flight global, and the bounce sound is muted for the duration.
- **Optional noise-free solver shots.** `exactSolve` plays a solved swing without meter error, so it always drops, and the agent's whole job becomes positioning. It is a rules change, off by default, and it travels with the checkpoint (`meta.env`) so every evaluator and the browser play the same game the model trained on. The random draws still happen, so the stream (wind, later swings) is the same with it on or off.
- **Curves find the blocked lies.** Most lies the straight search can't hole have a tree on the line. Hooking or slicing (impact ±0.06, ±0.12) bends the flight 10–20 yards and lifted solved lies by 2–5 points, to about 90%. Curves are the last resort, since each one costs trials on exactly the lies that are already failing.
- **Lossless upgrade.** New inputs are appended with zero weights and the `SOLVE` logit is inserted with zero weights, and the impact head starts at mean 0 (straight) with a narrow std, so a v3 agent upgrades to v5 unchanged in its mode. Fine-tuning then starts from a strong player instead of from scratch, and the trainer calibrates the new logit's bias so `SOLVE` gets explored.

## Learning algorithm

PPO with GAE (λ = 0.95), a clipped surrogate (0.2), separate actor and critic MLPs (1383 → 256 → 128), tanh, Adam, global grad-norm clip 0.5, linear LR decay, and an early stop at 1.5 × target KL.

- **Parallelism (Python).** The Rust envs step on every core while the networks sit on the GPU. Each env step is one batched forward pass for all 512 envs.
- **Parallelism (the original Node trainer).** N−1 worker threads each own an env. The parameters live in one `SharedArrayBuffer` that the trainer writes only between rounds, so there are no locks. Each worker keeps its own rollout and computes the gradient of its shard of every minibatch, which is equivalent to a stratified random minibatch. The trainer sums the shards and steps Adam. Observations never cross threads.
- **Entropy.** The bonus applies to the categorical heads only (0.01). The Gaussian std is left to the policy gradient, because putting needs it to collapse.

## How the design got here

| Iteration | Change | Result |
|---|---|---|
| v1, r1 | Aim relative to the pin, pin-frame grids | Beat the scripted bot within 40 iterations, but its deterministic play could loop into a tree it could not see |
| v2, r2 | Line-of-fire rays and last-shot memory | Still got stuck, because tee-only training rarely visits trouble |
| r3 | **Exploring starts**: 30% of episodes begin at a random playable spot | Better, but pickups remained on hairpins. The pin line crosses the woods, and ±60° of pin-relative aim cannot follow the fairway. |
| v3, r4 | **Actions residual to the game's default shot** (`aimDefault`), with shot-specific features in its aim frame | −9.9 / −9.4, about 7–8 strokes per round better than the scripted bot |
| v4 | **The solver** and the `SOLVE` action, still −1 per stroke | Not trained yet. See TRAINING.md. |
| v5 | **Impact as an action** (hooks and slices), for the agent and the solver | Not trained yet |

The v3 change mattered most. With the game's own default as the zero action, the sensible shot is the prior and the network learns corrections. Even untrained, the default shot scores about +7 on classic, against +27 for aiming at the pin.

## What the agent learned

Read from `npm run eval -- --set classic --rounds 1 --shots`:

- **The driver as a universal club.** Nearly every full shot is a 1W, down to 20-yard chips, with backspin on approaches. Power sets the distance, and backspin raises the launch angle and kills the roll (`SPIN_LOFT`, the first-bounce bite and the backspin draw on greens). This follows from the game's rules: a club's number is a maximum, the meter dials anything below it, and on these courses a lofted club offers little the driver with backspin does not. Irons appear mostly out of sand and on short pitches.
- **Putting from the preview.** Putts are asked 5–15% past the hole, with the aim offset following the previewed break, so most greens are one-putts.
- **Lay-ups.** On par 5s the tee shot is asked for about 260 yards along the default line, not the full distance.

## Things that were measured, not guessed

| Measurement | JS | Rust |
|---|---|---|
| Hole generation | 3.6 ms | 0.39 ms |
| A stroke (flight, bounces, roll) | about 4–6 ms (it loops every nearby tree each step) | 0.08 ms |
| Observation | about 0.5–1 ms | 0.14 ms |
| Training throughput, 16 cores | ~670 strokes/s (18 s an iteration) | ~12k strokes/s on the CPU alone (1.4 s an iteration) |
| 256-sample fwd+bwd of both nets | about 0.5 s single-threaded (hand-written) | PyTorch |

Results live in the results log in [TRAINING.md](TRAINING.md), and in the README.
