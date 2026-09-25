# Design notes

Why the agent is built the way it is. The environment contract is in [SPEC.md](SPEC.md), and running and results are in [TRAINING.md](TRAINING.md).

## Constraints that shaped it

- **No Python, no GPU on the dev machine.** The game is JavaScript, so everything is Node.js with zero npm dependencies. The costs are a hand-written MLP and backprop, which `npm test` checks against finite differences. The gain is that the agent runs the game's physics source directly, with no port and no drift.
- **The game is upstream code** (a submodule), and it is not modified. `rl/sim/loader.mjs` concatenates `engineMath.js`, `course.js` and `golfSim.js`, the same three files the game's own `tools/sim.mjs` uses, and wraps them in a `Function`. Game `let` bindings are reached through accessors.
- **`node:vm` was tried first and dropped.** Global-binding lookups through a vm context made a shot about 20× slower (44 ms against 2–4 ms). A closure keeps every game binding function-local. `Math` is shadowed per instance so `Math.random` (wind, swing noise) can be seeded without touching the host.

## Problem framing

- **One episode per hole, one step per stroke.** Holes are short: about 4 steps, and at most par+5 plus a penalty. The reward is −1 per stroke plus penalties, with γ = 1, so the return is exactly minus the score. There is no reward shaping. The critic learns expected strokes-to-hole from the state, which is the natural value function of golf.
- **Procedural training distribution.** Every training episode is a random hole of a random remix course, and there are about 2M × 18 layouts. The classic course and the low remix seeds are held out, so eval numbers measure generalisation, not memorisation.

## Action parameterisation

The raw action is **residual to the obvious shot**. `aim = 0, dist = 0` means "straight at the pin, asked for exactly the pin distance". The network learns the corrections: wind, lay-ups, dogleg lines, break, and pace over or under.

- The club is absolute, a categorical over 11. The game's own pre-selected club is in the observation, so "use the suggested club" is a single linear feature to pick up.
- Aim is ±60° wide so doglegs and escapes are reachable, and the state-dependent log std lets putts get sub-degree precise while drives stay loose.
- Distance is multiplicative (`exp`) because shot lengths span 1 to 600 yards.
- Everything goes through the game's `launchBall` with swing-meter timing noise. The agent never gets a perfect strike for free, the same deal the scripted bot gets.

## Observation

The player's view, in a pin-aligned frame so that "the pin is straight ahead" is invariant:

- **Two terrain grids.** One is scaled to the pin distance: on a 6-yard putt it samples the green's contour at sub-yard spacing, and on a drive it spans the hole. The other is fixed in yards over full-swing landing zones. Channels: hazard, sand, green, short grass, tree canopy, relative height.
- **The game's own aim previews.** The putt line with its break at two paces, and the still-air flight ring of the suggested club. A human reads these off the screen, so they are fair, and they turn putting from "infer the break from heights" into "correct the previewed miss".
- **v2 line-of-fire rays and last-shot memory.** These fix a deterministic loop in which v1 kept striking a tree it could not resolve.

## Learning algorithm

PPO with GAE (λ = 0.95), a clipped surrogate (0.2), separate actor and critic MLPs (1377 → 256 → 128), tanh, Adam, global grad-norm clip 0.5, linear LR decay, and an early stop at 1.5 × target KL.

- **Parallelism.** N−1 worker threads each own an env. The parameters live in one `SharedArrayBuffer` that the trainer writes only between rounds, so there are no locks. Each worker keeps its own rollout and computes the gradient of its shard of every minibatch, which is equivalent to a stratified random minibatch. The trainer sums the shards and steps Adam. Observations never cross threads.
- **Entropy.** The bonus applies to the categorical heads only (0.01). The Gaussian std is left to the policy gradient, because putting needs it to collapse.

## Things that were measured, not guessed

| Measurement | Value |
|---|---|
| Hole generation | 3.6 ms |
| A drive's flight | about 4 ms (it loops every nearby tree each step) |
| Observation | about 0.5–1 ms |
| One env step | about 6–7 ms in all |
| 256-sample fwd+bwd of both nets | about 0.5 s single-threaded |
| One iteration (15 workers × 1024 steps) | about 18 s |

Baselines on the eval sets (`npm run baseline`):

| Policy | Classic | Remix 1–4 |
|---|---|---|
| Naive (game defaults, aim at the pin) | +27.3 | +41.5 |
| Scripted dev bot (`Golf13K/game/tools/sim.mjs`) | −3.5 | −1.0 |
