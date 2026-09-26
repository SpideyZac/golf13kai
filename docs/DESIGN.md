# Design notes

Why the agent is built the way it is. The environment contract is in [SPEC.md](SPEC.md), and running and results are in [TRAINING.md](TRAINING.md).

## Constraints that shaped it

- **The game is upstream code** (a submodule), and it is not modified. `rl/sim/loader.mjs` concatenates `engineMath.js`, `course.js` and `golfSim.js`, the same three files the game's own `tools/sim.mjs` uses, and wraps them in a `Function`. Game `let` bindings are reached through accessors.
- **`node:vm` was tried first and dropped.** Global-binding lookups through a vm context made a shot about 20× slower (44 ms against 2–4 ms). A closure keeps every game binding function-local. `Math` is shadowed per instance so `Math.random` (wind, swing noise) can be seeded without touching the host.
- **The browser runs the agent against the real game.** Whatever trains it, the policy has to be the JS `Model` (`rl/policy.mjs`) over the JS observation (`rl/obs.mjs`), reading a JS-format checkpoint.

The first version was pure Node.js with zero dependencies: hand-written MLP and backprop, PPO over worker threads, and the game's physics run from its own source. That produced the shipped r4 agent in 2.3 hours on 16 cores. The training stack then moved to **Rust for the simulation and PyTorch for the learning**, to train on a GPU and to make the env fast enough to feed one.

## The Rust port (`golfsim/`)

The rule that the agent must play exactly what the browser runs still holds. It is now enforced by a test rather than by loading the source: `rl/test/parity.test.mjs` plays the same holes through the JS env (the game's own code) and through Rust, and requires `===` on every hole layout, every ball position and all 1383 observation floats after every stroke. Tolerances would have let porting bugs hide behind "float noise". Bit-exactness leaves no room for them, and it held over 1500 holes and 6145 strokes, with 731 tree strikes, 131 splashes and 175 OB.

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

- **One episode per hole, one step per stroke.** Holes are short: about 4 steps, and at most par+5 plus a penalty. The reward is −1 per stroke plus penalties, with γ = 1, so the return is exactly minus the score. There is no reward shaping by default. The critic learns expected strokes-to-hole from the state, which is the natural value function of golf. The optional hero mode (`--hero`, `py/golfrl/hero.py`) is the one exception. It adds bounded bonuses for long hole-outs, near misses from far and beating birdie, so the agent plays aggressively. It is applied in the trainer and not in the env, so parity is untouched, and evals and checkpoint selection still use the true score.
- **Procedural training distribution.** Every training episode is a random hole of a random remix course, and there are about 2M × 18 layouts. The classic course and the low remix seeds are held out, so eval numbers measure generalisation, not memorisation.

## Action parameterisation

The raw action is **residual to the game's default shot**. `aim = 0, dist = 0` means the aim line and target the game sets up for a player (`aimDefault`): the pin when the suggested club can reach it, otherwise a lay-up one carry down the centreline. The network learns the corrections: wind, lines, break, and pace over or under. (v1–v2 were residual to the pin, which failed on hairpins. See below.)

- The club is absolute, a categorical over 11. The game's own pre-selected club is in the observation, so "use the suggested club" is a single linear feature to pick up.
- Aim is ±60° around the default line, so escapes are reachable, and the state-dependent log std lets putts get sub-degree precise while drives stay loose.
- Distance is multiplicative (`exp`) because shot lengths span 1 to 600 yards.
- Everything goes through the game's `launchBall` with swing-meter timing noise. The agent never gets a perfect strike for free, the same deal the scripted bot gets.

## Observation

The player's view, in two frames: the default aim line for everything about this shot (grid B, rays, flight preview, centreline), and the pin line for the hole overview (grid A) and the putt preview.

- **Two terrain grids.** One is scaled to the pin distance: on a 6-yard putt it samples the green's contour at sub-yard spacing, and on a drive it spans the hole. The other is fixed in yards over full-swing landing zones. Channels: hazard, sand, green, short grass, tree canopy, relative height.
- **The game's own aim previews.** The putt line with its break at two paces, and the still-air flight ring of the suggested club. A human reads these off the screen, so they are fair, and they turn putting from "infer the break from heights" into "correct the previewed miss".
- **v2 line-of-fire rays and last-shot memory.** These fix a deterministic loop in which v1 kept striking a tree it could not resolve.

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
