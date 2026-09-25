// Loads the game's own simulation code (engineMath + course + golfSim) from the
// Golf13K submodule into an isolated closure, so every instance has its own
// `hole`, `ball` and Math.random. Nothing is copied or re-implemented: the RL
// agent plays exactly the physics the browser game runs.
//
// Why a closure and not node:vm: global lookups inside a vm context go through
// interceptors and made a shot ~20x slower (44ms vs ~2ms). Wrapping the source
// in a function keeps every game binding function-local and fast; `Math` is
// shadowed with a per-instance copy so seeding Math.random stays isolated.
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const GAME_ROOT = join(dirname(fileURLToPath(import.meta.url)), '../../Golf13K');
const read = (f)=> fs.readFileSync(join(GAME_ROOT, f), 'utf8');

// Same stubs as Golf13K/game/tools/sim.mjs: sound is never loaded headless.
const PRELUDE = 'let debug=0; const ASSERT=()=>{}; let time=0, frame=0; let remixMode=0;'
    + ' const sfxBounce=()=>{}; const snd_bounce={play(){}};\n';

// The epilogue exposes the game's lexical bindings through accessors.
const EPILOGUE = `
return {
    get hole() { return hole; }, set hole(h) { hole = h; },
    get noiseSeed() { return noiseSeed; }, set noiseSeed(s) { noiseSeed = s; },
    set remixMode(v) { remixMode = v; },
    get ball() { return ball; },
    get ballEvent() { return ballEvent; }, set ballEvent(v) { ballEvent = v; },
    get treeHit() { return treeHit; }, set treeHit(v) { treeHit = v; },
    get shotStart() { return shotStart; }, get ballSafe() { return ballSafe; },
    get pinOut() { return pinOut; }, set pinOut(v) { pinOut = v; },
    get rollRest() { return rollRest; }, set rollRest(v) { rollRest = v; },
    get lastAlong() { return lastAlong; },
    genCourse, genHole, groundAt, heightAt, surfaceAt, slopeAt, distToPath, pathPointAt,
    launchBall, launchPutt, ballUpdate, autoClub, lieMul, predictLanding, puttVel, rollStep,
    CLUBS, CLUB_PUTTER, PUTT_MAX, PUTT_OVER, SURF_PHYS, SURF_NAMES, GRAV, HOLE_R, CUP_SPEED,
    EV_HOLED, EV_STOPPED, EV_WATER, EV_OB,
    SURF_ROUGH, SURF_FAIRWAY, SURF_GREEN, SURF_TEE, SURF_BUNKER, SURF_WATER, SURF_OB,
    CLASSIC_HOLES, TRUNK_R, trunkH,
};`;

let source;
const gameSource = ()=> source ??= PRELUDE + read('src/engineMath.js') + read('game/course.js')
    + read('game/golfSim.js') + EPILOGUE;

// A fresh, independent game instance: {G, Math} where G exposes the game
// bindings and Math is the instance's private Math (seed Math.random on it).
export function loadGame()
{
    const M = {};
    for (const k of Object.getOwnPropertyNames(Math))
        M[k] = Math[k];
    const G = new Function('Math', 'console', gameSource())(M, console);
    return {G, Math: M};
}

// Deterministic PRNG (mulberry32), used to seed the instance's Math.random so
// wind and swing noise are reproducible.
export function mulberry32(seed)
{
    let a = seed >>> 0;
    return ()=>
    {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
