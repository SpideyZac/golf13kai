# GolfEnv specification (observation/action v3)

The environment contract between the game and the agent. `rl/env.mjs` implements the episode, and `rl/obs.mjs` implements the observation and action decoding, shared with the browser agent. `golfsim/` is the same contract in Rust, for training (see Physics).

Any change to what `observe()` writes, or to what `decode()` means, must bump `OBS_VERSION`. Checkpoints record the version they were trained on and refuse to load against another.

| Version | Change |
|---|---|
| v1 | Pin-frame grids, previews, and the game's club |
| v2 | Adds the last-shot and line-of-fire blocks. A deterministic v1 agent could loop against a tree it could not see at grid resolution. |
| v3 | Actions become residual to the **game's default shot**, and the shot-specific blocks move to its aim frame. With the aim relative to the pin and capped at 60°, the agent could not aim down the fairway of a hairpin, and drove into the woods until picked up. |

## Episodes

| | |
|---|---|
| Episode | One hole, from the tee (or, in training, an exploring start) until the ball is holed or the mercy rule picks it up |
| Step | One stroke |
| Reset spec | `{seed, remix, hole, rngSeed, start?}`: the course seed, remix or classic, hole index 0–17, the seed for the instance's `Math.random` (wind and swing noise), and an optional `start: {u, v}` in [0,1)² |
| Exploring start | Drop the ball at fraction `u` along the centreline, `(2v−1)·60` yards across it. The offset shrinks toward the centreline until the spot is not water or OB. |
| Termination | Holed, or `strokes >= par + maxOver` (default 5, the game's mercy rule). A penalty can push the final count to par+6, as it can in the game. |
| Reward | −1 per stroke, and −1 more for each water or OB penalty. The undiscounted return is minus the hole score. |

The rules follow `Golf13K/game/game.js`:

- **Penalty drop** (`updateFlight`). Walk back from the last safe point toward where the shot was played, and stop on ground that is not water, OB or green and is flat enough to hold a ball.
- **Pin** (`enterAim`). The pin is pulled when the ball is on the green within 15 yards.
- **Unsettled ball.** A ball still moving after 30 simulated seconds is played where it lies.

## Physics

There are two implementations of this contract, and they agree bit for bit:

- **JS** (`rl/env.mjs`, `rl/obs.mjs`): runs the game's own `src/engineMath.js`, `game/course.js` and `game/golfSim.js`, loaded from the submodule by `rl/sim/loader.mjs`. Each env instance gets its own closure, with its own `hole`, `ball` and `Math`. The browser agent uses `rl/obs.mjs` against the live game.
- **Rust** (`golfsim/`): a line-by-line port of those files and of `rl/obs.mjs`/`rl/env.mjs`, which the Python trainer runs. Its math goes through `golfsim/src/jsmath.rs`, which reproduces V8's `Math.sin`, `cos`, `tanh`, `log`, `log1p`, `exp`, `atan2` and `hypot` exactly (fdlibm and V8's hypot algorithm). Every RandomGenerator draw happens in the JS order, and float operations keep the JS operand order.

`rl/test/parity.test.mjs` plays the same holes and swings through both and compares every hole layout, ball position and observation float with `===`. Any change to the game (a submodule bump) or to either side has to keep it passing.

## The reference shot

`Observer.reference()` reproduces the shot the game sets up when a player addresses the ball, from `enterAim` / `aimDefault`:

- **club** is `autoClub()`.
- **reach** is that club's carry × `lieMul`, or `PUTT_MAX` for the putter.
- If the pin is within reach + 20 yards, the reference aims at the pin, and `ref.dist` is the pin distance.
- Otherwise it aims at the centreline point `lastAlong + 0.95·reach` (a lay-up that follows the fairway), and `ref.dist` is the distance to that point.

## Action

| Field | Type | Meaning |
|---|---|---|
| `club` | int 0–10 | `CLUBS` index: 1W 3W 5W 3i 5i 7i 9i 13i PW SW, and 10 is the putter |
| `spin` | int 0–2 | back / none / top. Ignored for putts. |
| `aim` | real | `yaw = ref.dir + 0.35 · clip(aim, −3, 3)` radians, so ±60° around the default aim line |
| `dist` | real | `want = ref.dist · exp(0.35 · clip(dist, −4, 2))` yards, from 0.25× to 2× the default target |

`aim = dist = 0` with the game's club is exactly the game's default shot.

Decoding follows the game's meter: `power = want / (carry(club) · lieMul(club))` clipped to [0.02, 1], or `want / PUTT_MAX` for the putter. The shot is then played through `launchBall`, as a player's swing would be, with this execution noise:

- `impactNoise` (default ±0.04, uniform): swing-meter timing error. It costs power and pushes, hooks or slices exactly as the game does. Errors under 0.02 snap to a perfect strike.
- `aimNoise` (default ±0.015 rad, uniform): full swings only.

These match the error the game's scripted dev bot is given, so their scores can be compared.

## Observation (1383 floats)

There are two frames. In each, `+fwd` points along the frame's line and `+lat` is to its right, the side a positive `aim` moves the shot toward.

- **Aim frame**: along the reference aim line.
- **Pin frame**: from the ball to the pin.

| Block | Size | Frame | Contents |
|---|---|---|---|
| Lie | 7 | | One-hot surface under the ball: rough, fairway, green, tee, sand, water, OB |
| Lie cost | 2 | | `lieMul` for a non-wedge and for a wedge |
| Wind | 2 | aim | Along (tailwind +) and across, speed / 8 |
| Par | 3 | | One-hot 3/4/5 |
| Round state | 3 | | strokes/10, strokes left before pickup /10, hole hilliness |
| Slope | 2 | aim | Ground gradient at the ball × 5 |
| Pin height | 1 | | tanh((pin h − ball h)/10) |
| Pin distance | 3 | | d/300, log1p(d)/6, whether d < 45 |
| Game's club | 11 | | One-hot `autoClub()` |
| Path progress | 3 | | Distance off the centreline /60, fraction along, yards remaining /300 |
| Last shot | 3 | | log1p(yards the last shot moved)/6, whether it hit a tree, whether it found water or OB. All 0 at the start. |
| Line of fire | 26 | aim | 13 rays at aim offsets −60°…+60°, every 10°. Each gives `1 − dist/80` to the first tree canopy on the line (0 = clear for 80 yards), and the steepest rise `tanh(2·Δh/r)` over r ∈ {4, 8, 14, 22, 32, 45, 60} |
| Default vs pin | 6 | aim | Pin (fwd, lat)/300, cos and sin of (pin dir − aim dir), ref.dist/300, log(ref.dist / d) |
| Putt preview | 8 | pin | If d < 45: the game's putt line aimed at the pin with pace d and 1.3d. For each: stop fwd error, stop lat error, closest-pass lateral miss (all ÷ d, tanh), and whether it drops. |
| Flight preview | 5 | aim | The reference shot (game's club, power for ref.dist) in still air: landing fwd error vs ref.dist, landing lat, whether the flight clips a rising face, and whether it lands in water/OB or sand |
| Centreline ahead | 20 | aim | Path points 30…300 yards further along, as (fwd, lat)/300 |
| Grid A | 108×6 | pin | Scaled by `s = max(d, 4)`: fwd ∈ {−.1 … 1.5}·s, lat ∈ {−.4 … .4}·s |
| Grid B | 105×6 | aim | Fixed yards: fwd 20…300 step 20, lat ∈ {−50, −30, −15, 0, 15, 30, 50} |

Each grid cell has 6 channels:

1. water or OB
2. sand
3. green
4. fairway or tee
5. inside a tree canopy that the ball can hit
6. relative height, `tanh((h − h_ball)/hs)` with `hs = max(.5, .05s)` for grid A and 15 for grid B

**Fairness.** The agent sees roughly what a player sees: the terrain, the wind arrow, the lie, the pre-selected club and aim, and the previews the game draws. The previews are the dashed line and ring, including the putt's break, and they fly in still air exactly as in the game. The agent is not told the exact cup physics, the tree collision state or future wind.

## Evaluation (`rl/courses.mjs`, `rl/eval.mjs`)

| Set | What it is |
|---|---|
| Training | Remix courses with seed in [1000, 2e6), a random hole, random wind, and 30% exploring starts |
| `classic` | The real classic course (seed 1113, not remix), with round *r* using wind seed `1000(r+1)+hole`. Never trained on unless `--classic-prob` is set. |
| `remix` | Remix seeds 1…R, which lie outside the training range |

Eval plays **deterministically**: argmax club and spin, mean aim and distance, with execution noise on. With `--escape`, the shot after a stuck full swing (moved < 10 yards, or found a hazard) is sampled instead (`isStuck` in `obs.mjs`). The browser agent always plays with escape.
