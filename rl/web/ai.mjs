// The trained agent playing the real browser game. Open rl/web/?auto=1 through
// rl/web/serve.mjs. The game's dev autoplay (debugGame.js) calls the global
// botSwing() every frame of the aim state; this module replaces it with one
// that observes the hole exactly as rl/obs.mjs does in training, asks the
// network for a shot, lines the camera up like the dev bot, and swings through
// the same launchBall with the same meter noise.
//
// Query params: ?model=<url of a checkpoint json> (default ../../models/agent.json)
import { API_BODY } from '../sim/api.mjs';
import { Observer, OBS_VERSION, OBS_DIM, CLUB_NAMES } from '../obs.mjs';
import { Model } from '../policy.mjs';

// the game's globals, through the same accessor list the Node loader uses
const G = new Function(API_BODY)();
const observer = new Observer(G);
const q = new URLSearchParams(location.search);
const MODEL_URL = q.get('model') || '../../models/agent.json';

let model = null;
fetch(MODEL_URL).then(r => r.json()).then(j =>
{
    if (j.obsVersion != OBS_VERSION || j.obsDim != OBS_DIM)
        throw new Error(`model is obs v${j.obsVersion}, page is v${OBS_VERSION}`);
    const bin = atob(j.params), u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; ++i) u8[i] = bin.charCodeAt(i);
    model = new Model(j.arch).bind(new Float32Array(u8.buffer));
    console.log(`RL agent loaded: ${MODEL_URL} (iter ${j.meta?.iter}, ${model.size} params)`);
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

let curHole = null, prev, lastStart, shot = null;
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
        const obs = observer.observe({strokes, prev, maxOver: 5});
        const a = model.act(obs, Math.random, ()=> 0, true);
        shot = observer.decode(a);
        clubI = shot.club;
        spinMode = shot.spin;
        aimYaw = shot.yaw;
        setTarget(shot.want);
        console.log(`AI ${CLUB_NAMES[shot.club]} ${['back', 'flat', 'top'][shot.spin+1]}`
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
    flags.tree = flags.hazard = 0;
    const putt = shot.club == CLUB_PUTTER;
    // the meter noise the agent trained with (rl/env.mjs DEFAULT_ENV)
    launchBall(shot.club, shot.power, rand(.04, -.04), shot.spin,
        shot.yaw + (putt ? 0 : rand(.015, -.015)), shot.lm);
    putt ? snd_putt.play(.4 + shot.power*.6) : snd_tee.play(.5 + shot.power*.5, .8 + shot.power*.4);
    startFlight();
};
