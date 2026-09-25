# GolfEnv specification (observation v2)

The environment contract between the game and the agent. `rl/env.mjs` implements it. v2 added the last-shot and line-of-fire blocks: v1 agents looped forever against a tree they could not see at grid resolution. Any change to the observation layout must bump `OBS_VERSION`, because checkpoints record the version they were trained on and refuse to load against another.

## Episodes

| | |
|---|---|
| Episode | One hole, from the tee until the ball is holed or the mercy rule picks it up |
| Step | One stroke |
| Reset spec | `{seed, remix, hole, rngSeed}`: the course seed, remix or classic, hole index 0–17, and the seed for the instance's `Math.random` (wind and swing noise) |
| Termination | Holed, or `strokes >= par + maxOver` (default 5, the game's mercy rule). A penalty can push the final count to par+6, as it can in the game. |
| Reward | −1 per stroke, and −1 more for each water or OB penalty. The undiscounted return is minus the hole score. |

The rules follow `Golf13K/game/game.js` `updateFlight` exactly:

- **Penalty drop.** Walk back from the last safe point toward where the shot was played, and stop on ground that is not water, OB or green and is flat enough to hold a ball.
- **Pin.** The pin is pulled for shots played from inside 10 yards.
- **Unsettled ball.** A ball still moving after 30 simulated seconds is played where it lies.

## Physics

The environment runs the game's own `src/engineMath.js`, `game/course.js` and `game/golfSim.js`, loaded from the submodule by `rl/sim/loader.mjs`. Each env instance gets its own closure, with its own `hole`, `ball` and `Math`, so instances can be interleaved safely. Nothing in the physics is reimplemented.

## Action

| Field | Type | Meaning |
|---|---|---|
| `club` | int 0–10 | `CLUBS` index: 1W 3W 5W 3i 5i 7i 9i 13i PW SW, and 10 is the putter |
| `spin` | int 0–2 | back / none / top. Ignored for putts. |
| `aim` | real | `yaw = pinDir + 0.35 * clip(aim, -3, 3)` in radians, so up to ±60° off the pin line |
| `dist` | real | `want = pinDist * exp(0.35 * clip(dist, -4, 2))` yards, from 0.25× to 2× the pin distance |

Decoding follows the game's meter: `power = want / (carry(club) * lieMul(club))` clipped to [0.02, 1], or `want / PUTT_MAX` for the putter. The shot is then played through `launchBall`, as a player's swing would be, with this execution noise:

- `impactNoise` (default ±0.04, uniform): swing-meter timing error. It costs power and pushes, hooks or slices exactly as the game does. Errors under 0.02 snap to a perfect strike.
- `aimNoise` (default ±0.015 rad, uniform): full swings only.

These match the error the game's scripted dev bot is given, so their scores can be compared.

## Observation (1377 floats)

Everything spatial is in the **pin frame**: `+fwd` points from the ball to the pin, and `+lat` is to the right of that line, the side a positive `aim` moves the shot toward.

| Block | Size | Contents |
|---|---|---|
| Lie | 7 | One-hot surface under the ball: rough, fairway, green, tee, sand, water, OB |
| Lie cost | 2 | `lieMul` for a non-wedge and for a wedge |
| Wind | 2 | Along (tailwind +) and across, speed / 8 |
| Par | 3 | One-hot 3/4/5 |
| Round state | 3 | strokes/10, strokes left before pickup /10, hole hilliness |
| Slope | 2 | Ground gradient at the ball (fwd, lat) × 5 |
| Pin height | 1 | tanh((pin h − ball h)/10) |
| Distance | 3 | d/300, log1p(d)/6, whether d < 45 |
| Game's club | 11 | One-hot `autoClub()`, the club the game pre-selects |
| Path progress | 3 | Distance off the centreline /60, fraction along, yards remaining /300 |
| Last shot | 3 | log1p(yards the last shot moved)/6, whether it hit a tree, whether it found water or OB. All 0 on the tee. |
| Line of fire | 26 | 13 rays at aim offsets −60°…+60°, every 10°. Each gives `1 − dist/80` to the first tree canopy on the line (0 = clear for 80 yards), and the steepest rise `tanh(2·Δh/r)` over r ∈ {4, 8, 14, 22, 32, 45, 60} |
| Putt preview | 8 | If d < 45: the game's putt line aimed at the pin with pace d and 1.3d. For each: stop fwd error, stop lat error, closest-pass lateral miss (all ÷ d, tanh), and whether it drops. |
| Flight preview | 5 | For the game's club aimed at the pin with power for d, in still air: landing fwd error, landing lat, whether the flight clips a rising face, and whether it lands in water/OB or sand |
| Centreline ahead | 20 | Path points 30…300 yards further along, as (fwd, lat)/300 |
| Grid A | 108×6 | Scaled by `s = max(d, 4)`: fwd ∈ {−.1 … 1.5}·s, lat ∈ {−.4 … .4}·s |
| Grid B | 105×6 | Fixed yards: fwd 20…300 step 20, lat ∈ {−50, −30, −15, 0, 15, 30, 50} |

Each grid cell has 6 channels:

1. water or OB
2. sand
3. green
4. fairway or tee
5. inside a tree canopy that the ball can hit
6. relative height, `tanh((h − h_ball)/hs)` with `hs = max(.5, .05s)` for grid A and 15 for grid B

**Fairness.** The agent sees roughly what a player sees: the terrain, the wind arrow, the lie, the pre-selected club, and the aim previews the game draws. The previews are the dashed line and ring, including the putt's break, and they fly in still air exactly as in the game. The agent is not told the exact pin cup physics, the tree collision state or future wind.

## Evaluation sets (`rl/courses.mjs`)

| Set | What it is |
|---|---|
| Training | Remix courses with seed in [1000, 2e6), a random hole, and random wind |
| `classic` | The real classic course (seed 1113, not remix), with round *r* using wind seed `1000(r+1)+hole`. Never trained on unless `--classic-prob` is set. |
| `remix` | Remix seeds 1…R, which lie outside the training range |

Eval plays deterministically: argmax club and spin, mean aim and distance. Execution noise stays on.
