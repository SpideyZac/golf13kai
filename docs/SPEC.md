# GolfEnv specification (observation/action v5)

The environment contract between the game and the agent. `rl/env.mjs` implements the episode, and `rl/obs.mjs` implements the observation and action decoding, shared with the browser agent. `golfsim/` is the same contract in Rust, for training (see Physics).

Any change to what `observe()` writes, or to what `decode()` means, must bump `OBS_VERSION`. Checkpoints record the version they were trained on and refuse to load against another.

| Version | Change |
|---|---|
| v1 | Pin-frame grids, previews, and the game's club |
| v2 | Adds the last-shot and line-of-fire blocks. A deterministic v1 agent could loop against a tree it could not see at grid resolution. |
| v3 | Actions become residual to the **game's default shot**, and the shot-specific blocks move to its aim frame. With the aim relative to the pin and capped at 60°, the agent could not aim down the fairway of a hairpin, and drove into the woods until picked up. |
| v4 | **The solver.** Every observation runs a holing-shot solver (see Solver), whose result is a new 6-float block at the end, and the club head gains a 12th choice, `CLUB_SOLVE`, that plays its shot. v3 checkpoints upgrade losslessly (`python -m golfrl.upgrade`). |
| v5 | **The impact is an action.** The agent chooses the swing-meter impact it swings for, to hook or slice the ball, and the swing noise adds to it. The solver searches curved shots too. The observation is unchanged (1389 floats). v3 checkpoints still upgrade losslessly. |

## Episodes

| | |
|---|---|
| Episode | One hole, from the tee (or, in training, an exploring start) until the ball is holed or the mercy rule picks it up |
| Step | One stroke |
| Reset spec | `{seed, remix, hole, rngSeed, start?}`: the course seed, remix or classic, hole index 0–17, the seed for the instance's `Math.random` (wind and swing noise), and an optional `start: {u, v}` in [0,1)² |
| Exploring start | Drop the ball at fraction `u` along the centreline, `(2v−1)·60` yards across it. The offset shrinks toward the centreline until the spot is not water or OB. |
| Termination | Holed, or `strokes >= par + maxOver` (default 5, the game's mercy rule). A penalty can push the final count to par+6, as it can in the game. |
| Reward | −1 per stroke, and −1 more for each water or OB penalty. The undiscounted return is minus the hole score. There is no bonus for using the solver or for holing from far: the solver pays off only through strokes saved. |

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
| `club` | int 0–11 | `CLUBS` index: 1W 3W 5W 3i 5i 7i 9i 13i PW SW, and 10 is the putter. **11 is `CLUB_SOLVE`**: the solver's shot exactly as solved (spin, aim and dist are ignored), or, when it found none, the game's club with this action's spin, aim and dist. |
| `spin` | int 0–2 | back / none / top. Ignored for putts. |
| `aim` | real | `yaw = ref.dir + 0.35 · clip(aim, −3, 3)` radians, so ±60° around the default aim line |
| `dist` | real | `want = ref.dist · exp(0.35 · clip(dist, −4, 2))` yards, from 0.25× to 2× the default target |
| `impact` | real | The swing-meter impact swung for: `0.05 · clip(impact, −2.6, 2.6)`, so ±0.13 (the meter's late limit, `METER_OVER`). + is early, − is late. In `launchBall` it pushes the start line (`impact·0.05` rad, or `·0.75` on a putt), curves the flight (`ballCurve = impact·22` yd/s² across the shot) and costs power (`·(1 − 0.35·|impact|)`). |impact| < 0.02 is a perfect, straight strike. |

`aim = dist = impact = 0` with the game's club is exactly the game's default shot.

Decoding follows the game's meter: `power = want / (carry(club) · lieMul(club))` clipped to [0.02, 1], or `want / PUTT_MAX` for the putter. The shot is then played through `launchBall`, as a player's swing would be, with this execution noise:

- `impactNoise` (default ±0.04, uniform): swing-meter timing error, added to the impact swung for. It costs power and pushes, hooks or slices exactly as the game does. A total under 0.02 snaps to a perfect strike, so a straight swing is perfect half the time at the default noise, while a shaped one never gets that snap.
- `aimNoise` (default ±0.015 rad, uniform): full swings only.

These match the error the game's scripted dev bot is given, so their scores can be compared. By default a solver shot gets the same noise, since solving it does not make the swing perfect.

**Optional rules** (env config, default off; the Python trainer records them in the checkpoint's `meta.env`, and `golfrl.evaluate`, `rl/eval.mjs` and the browser agent play by them):

| Option | Effect |
|---|---|
| `exactSolve` | A solver shot (`CLUB_SOLVE` with a solution) is played with no impact or aim error, so it **always holes**. The solver then measures its odds noise-free too: P(holed) = 1 and no replays. Every other shot keeps its noise. |
| `impactNoise` / `aimNoise` = 0 | No swing error on any shot: the game becomes fully deterministic given the hole. |

The random draws are made either way, so these options change only what the draws do. The wind and the rest of the random stream are unchanged.

## Solver (`rl/solver.mjs`, `golfsim/src/solver.rs`)

Runs at the end of every `observe()` (unless the env's `solver` option is off) and finds a swing that holes the ball from where it lies, wind, slopes, bounces, spin and cup included. It uses no model of the physics. It plays trial shots through the game's own `launchBall` and `ballUpdate`, the same loop `GolfEnv.step` runs (pin rule included), and puts every flight global back afterwards. That is why the browser agent can run it on the live game.

1. **In range.** It runs only when the game's default shot aims at the pin (the pin within the game's club's reach + 20 yards, as in the reference shot). Otherwise it reports `tried = 0`.
2. **Candidates** (club, spin, impact). The putter, when the ball is on the green, fairway or tee within `PUTT_MAX`. Then the game's club (`autoClub`) with no spin and with backspin. When the game hands over the putter off the green, the sand wedge takes the game's club's place. All of these are struck straight (impact 0). **Fallback:** when none of these holes, it tries the clubs up to 2 either side of that club, longest first, and stops at the first solution. It tries them straight first, with no spin, backspin and topspin: a longer club or topspin reaches a pin the game's club can't run up to, and a shorter club can stop on a ridge. Then it tries them **hooked and sliced** (impact ±0.06 and ±0.12) with no spin and backspin, to bend around a tree that blocks the straight line.
3. **Steering.** Start at the pin line with the pin distance as the target. Play a trial with the candidate's impact and no noise. If it holed, that is the solution. Otherwise turn the aim by the angle the ball finished off the pin line, and scale the target by pin distance ÷ yards travelled. Repeat, up to `SOLVE_ITERS` = 8 trials. It gives up on a candidate when a trial moves under 0.5 yards (a tree, a wall) or falls short at full power. Aiming for the ball to stop at the pin means it crosses the cup at crawling pace, which is the speed that drops.
4. **Odds.** Each solution is replayed across the swing noise as weighted samples. The impact error takes the centres of the quarters of ±`impactNoise`, added to the solution's impact and snapped as `launchBall` does. From a straight swing, the two inner samples are the perfect strike. The aim error (full swings) takes the centres of the quarters of ±`aimNoise`. No aim sample is 0, because the exact solution always drops. Putts have no aim noise, so a perfect putt counts as holed. Samples that snap to the same strike share one trial. The result is P(holed), P(water or OB), and the expected yards left, with a holed ball counting 0.
5. **Choice.** The candidate with the highest P(holed) wins, then the fewest yards left, then the earlier candidate.

A solution is **exact**: with a perfect strike (no meter error) it holes every time. P(holed) is below 1 only because of the swing noise the env applies to every shot.

It finds a solution for about 86–92% of in-range lies (82% with the game's club alone, about 86% with the straight fallback, and the rest from curves). The rest have no way in that it can reach: a tree blocks even the curves, or no club gets there. More steering trials, bracketing the power, steering on the closest pass to the cup, and aiming past the pin were all measured and found no more solutions. Aiming past the pin found fewer.

Typical odds (the default noise): putts 0.35–0.75, 40–100 yards about 0.2, and 100–200 yards about 0.01.

**Fairness.** The solver knows the exact physics, including the cup, trees and wind, which no player can see precisely. Agents that use it are not comparable with the scripted bot or with v3 agents, which see only what a player sees. It costs time as well: an in-range observation plays 10–60 trial shots, so the Rust env runs about 6× slower (4.2k against 27k strokes a second on 16 cores).

## Observation (1389 floats)

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
| Solver | 6 | | Whether it ran (pin in range), whether it found a holing swing, and for that swing: P(holed), P(water/OB), log1p(expected yards left)/4, and whether it is a putt. The last four are 0 without a solution. |

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
