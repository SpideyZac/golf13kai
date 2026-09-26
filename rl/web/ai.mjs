// The trained agent playing the real browser game. Open rl/web/?auto=1 through
// rl/web/serve.mjs. The game's dev autoplay (debugGame.js) calls the global
// botSwing() every frame of the aim state; this module replaces it with one
// that observes the hole exactly as rl/obs.mjs does in training, asks the
// network for a shot, lines the camera up like the dev bot, and swings through
// the same launchBall with the same meter noise.
//
// The observation runs the solver (rl/solver.mjs), which plays silent trial
// shots on the live game and puts every flight global back before returning.
//
// Query params: ?model=<url of a checkpoint json> (default ../../models/agent.json)
//
// The single-file build (rl/web/build.mjs) sets window.RL_MODEL (the checkpoint
// itself, used unless ?model= is given) and window.RL_AUTO (play without
// ?auto=1; ?auto=0 still turns it off).
import { API_BODY } from '../sim/api.mjs';
import { Observer, OBS_VERSION, OBS_DIM, CLUB_NAMES, isStuck } from '../obs.mjs';
import { Model } from '../policy.mjs';
import { randnFrom } from '../nn.mjs';

// the game's globals, through the same accessor list the Node loader uses
const G = new Function(API_BODY)();
const observer = new Observer(G);
const q = new URLSearchParams(location.search);
const MODEL_URL = q.get('model') || '../../models/agent.json';
const EMBEDDED = !q.get('model') && window.RL_MODEL;

// the game's gameInit has already run (this module executes after every
// classic script), so this is the T key's path: switch the bot on and, at the
// title, deal a round
if (window.RL_AUTO && !autoPlay && q.get('auto') !== '0')
{
    autoPlay = 1;
    state == ST_TITLE && startCourse(+q.get('remix') || 0);
}

let model = null;
// the env rules the model trained under (py/golfrl/train.py records them as
// meta.env): the swing noise, and whether the solver's shots skip it
let rules = {impactNoise: .04, aimNoise: .015, exactSolve: false};
(EMBEDDED ? Promise.resolve(EMBEDDED) : fetch(MODEL_URL).then(r => r.json())).then(j =>
{
    if (j.obsVersion != OBS_VERSION || j.obsDim != OBS_DIM)
        throw new Error(`model is obs v${j.obsVersion}, page is v${OBS_VERSION}`);
    const bin = atob(j.params), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) u8[i] = bin.charCodeAt(i);
    model = new Model(j.arch).bind(new Float32Array(u8.buffer));
    rules = {...rules, ...j.meta?.env};
    console.log(`RL agent loaded: ${EMBEDDED ? 'embedded model' : MODEL_URL} (iter ${j.meta?.iter}, ${model.size} params)`);
    autoPlay || console.log('add ?auto=1 to the URL to let the agent play');
}).catch(e => console.error('RL agent failed to load:', e));

// tree strikes and penalties are announced through showMsg (the flags
// themselves are cleared by the game the frame they are seen)
const flags = {tree: 0, hazard: 0};
const showMsg0 = window.showMsg;
window.showMsg = (t)=>
{
    if (t == 'TREE!') flags.tree = 1;
    if (/SPLASH|OUT OF BOUNDS/.test(t)) flags.hazard = 1;
    showMsg0(t);
};

// the solver's trial shots bounce too: keep them silent
function quietly(f)
{
    const bounce = window.sfxBounce, blip = snd_bounce.play;
    window.sfxBounce = ()=>{};
    snd_bounce.play = ()=>{};
    try { return f(); }
    finally { window.sfxBounce = bounce; snd_bounce.play = blip; }
}

let curHole = null, prev, lastStart, shot = null, lastPutt = false;
const randn = randnFrom(Math.random);
window.botSwing = function aiSwing()
{
    if (!model) return;
    if (hole !== curHole)
    {
        curHole = hole;
        observer.newHole();
        prev = {moved: 0, tree: 0, hazard: 0};
        lastStart = null;
    }
    if (!botLined)
    {
        // DECIDE once per shot, then line up (same camera ease as the dev bot)
        if (lastStart)
            prev = {moved: Math.hypot(ball.x - lastStart.x, ball.z - lastStart.z), ...flags};
        // the noise the agent trained with, which the solver measures its odds
        // against (none for its own shots under exactSolve, as in rl/env.mjs)
        const ex = rules.exactSolve;
        const obs = quietly(()=> observer.observe({strokes, prev, maxOver: 5,
            impactNoise: ex ? 0 : rules.impactNoise, aimNoise: ex ? 0 : rules.aimNoise}));
        // the mode, unless the last full swing got nowhere (the escape rule)
        const stuck = lastStart && isStuck(prev, lastPutt);
        const a = model.act(obs, Math.random, randn, !stuck);
        shot = observer.decode(a);
        clubI = shot.club;
        spinMode = shot.spin;
        aimYaw = shot.yaw;
        setTarget(shot.want);
        const sol = observer.solution;
        console.log(`AI${stuck ? ' (escape)' : ''}${shot.solved ? ` SOLVE (holes ${(sol.pHole*100).toFixed(0)}%)` : ''}`
            + ` ${CLUB_NAMES[shot.club]} ${['back', 'flat', 'top'][shot.spin+1]}`
            + ` aim ${((shot.yaw - observer.pinDir())*180/Math.PI).toFixed(1)}deg off the pin,`
            + ` asks ${shot.want.toFixed(1)}yd of ${ballToPin().toFixed(1)}`);
        botLined = 1;
        camFrom = [camX, camY, camZ, camYaw, camPitch];
        camEase = 0;
        return;
    }
    if (camEase < SETTLE_T)
        return;
    botLined = 0;
    ++strokes;
    lastStart = {x: ball.x, z: ball.z};
    lastPutt = shot.club == CLUB_PUTTER;
    flags.tree = flags.hazard = 0;
    const putt = shot.club == CLUB_PUTTER;
    // the meter noise the agent trained with (none on an exactSolve solver shot)
    const exact = rules.exactSolve && shot.solved;
    launchBall(shot.club, shot.power, exact ? 0 : rand(rules.impactNoise, -rules.impactNoise), shot.spin,
        shot.yaw + (putt || exact ? 0 : rand(rules.aimNoise, -rules.aimNoise)), shot.lm);
    putt ? snd_putt.play(.4 + shot.power*.6) : snd_tee.play(.5 + shot.power*.5, .8 + shot.power*.4);
    startFlight();
};
