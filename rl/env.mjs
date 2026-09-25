// GolfEnv: one episode = one hole of Sunshine Golf Classic, one step = one
// stroke. Full spec in docs/SPEC.md.
//
// The agent sees what a human player sees (rl/obs.mjs): the terrain ahead, the
// wind arrow, the lie, the club the game pre-selects, and the aim previews.
// It acts the way a player does: pick a club, a spin, an aim and a distance;
// the swing itself goes through the game's launchBall with meter-timing noise.
import { loadGame, mulberry32 } from './sim/loader.mjs';
import { Observer, OBS_DIM } from './obs.mjs';
export * from './obs.mjs';

export const DEFAULT_ENV = {
    impactNoise: .04,  // uniform +- swing meter timing error (the scripted bot's)
    aimNoise: .015,    // uniform +- radians on full swings (the scripted bot's)
    maxOver: 5,        // the game's mercy rule: pick up at par+5
};

export class GolfEnv
{
    constructor(cfg = {})
    {
        this.cfg = {...DEFAULT_ENV, ...cfg};
        const {G, Math: M} = loadGame();
        this.G = G; this.M = M;
        this.observer = new Observer(G);
    }

    // opts: {seed, remix, hole (0-17), rngSeed, start}
    // start (training only): {u, v} in [0,1) drops the ball at a random spot
    // of the hole instead of the tee - u along the centreline, v across it
    reset({seed = 1113, remix = false, hole = 0, rngSeed = 1, start = null} = {})
    {
        const G = this.G;
        this.M.random = mulberry32(rngSeed);
        G.remixMode = remix ? 1 : 0;
        const rows = G.genCourse(seed, remix ? 1 : 0);
        G.genHole(seed, hole, rows[hole]);
        this.h = G.hole;
        this.seed = seed; this.remix = remix; this.holeIndex = hole;
        const b = G.ball;
        b.x = b.z = b.vx = b.vy = b.vz = 0;
        if (start) this.placeBall(start);
        b.y = G.groundAt(b.x, b.z).h;
        G.ballEvent = 0;
        this.strokes = 0; this.penalties = 0; this.done = false;
        this.log = [];
        this.prev = {moved: 0, tree: 0, hazard: 0};
        this.observer.newHole();
        return this.observe();
    }

    // EXPLORING STARTS: anywhere in the corridor that is not water or OB -
    // rough under the trees, sand, a hillside, the green. Trouble is rare from
    // the tee, so without these the agent never practises getting out of it.
    // The lateral offset shrinks toward the centreline until the spot is legal.
    placeBall({u, v})
    {
        const G = this.G, h = this.h, b = G.ball;
        const along = u*h.len*.97;
        const p = G.pathPointAt(along), q = G.pathPointAt(Math.min(along + 2, h.len));
        const tl = Math.hypot(q.x-p.x, q.z-p.z) || 1;
        const nx = (q.z-p.z)/tl, nz = -(q.x-p.x)/tl;
        for (let lat = (2*v - 1)*60; ; lat *= .6)
        {
            const x = p.x + nx*lat, z = p.z + nz*lat, s = G.surfaceAt(x, z);
            if (s < G.SURF_WATER) { b.x = x; b.z = z; return; }
            if (Math.abs(lat) < 1) return; // centreline is wet (a river): keep the tee
        }
    }

    pinDist() { return this.observer.pinDist(); }
    observe() { return this.observer.observe({strokes: this.strokes, prev: this.prev, maxOver: this.cfg.maxOver}); }
    decode(action) { return this.observer.decode(action); }

    // action: {club 0-10, spin 0-2 (back/none/top), aim, dist}
    // returns {obs, reward, done, info}
    step(action)
    {
        if (this.done) throw new Error('step after done');
        const G = this.G, M = this.M, cfg = this.cfg;
        const s = this.decode(action);
        const putt = s.club == G.CLUB_PUTTER;
        const lie = G.SURF_NAMES[G.groundAt(G.ball.x, G.ball.z).s], d0 = this.pinDist();
        // the swing meter: a player's timing is never perfect
        const impact = (M.random()*2 - 1)*cfg.impactNoise;
        const yaw = s.yaw + (putt ? 0 : (M.random()*2 - 1)*cfg.aimNoise);
        // the pin is pulled for a shot from the green inside 15yd, as enterAim does
        G.pinOut = d0 < 15 && lie == 'GREEN' ? 1 : 0;
        G.treeHit = 0;
        G.launchBall(s.club, s.power, impact, s.spin, yaw, s.lm);
        ++this.strokes;
        let reward = -1;
        for (let t = 0; t < 60*30 && !G.ballEvent; ++t)
            G.ballUpdate();
        const ev = G.ballEvent || G.EV_STOPPED; // a ball that never settles is played where it lies
        G.ballEvent = 0;
        let result = ev == G.EV_HOLED ? 'holed' : ev == G.EV_WATER ? 'water' : ev == G.EV_OB ? 'ob' : 'stopped';
        if (ev == G.EV_WATER || ev == G.EV_OB)
        {
            ++this.strokes; ++this.penalties; reward -= 1;
            this.hazardDrop();
        }
        const b = G.ball;
        b.vx = b.vy = b.vz = 0;
        this.prev = {moved: Math.hypot(b.x - G.shotStart.x, b.z - G.shotStart.z),
            tree: G.treeHit ? 1 : 0, hazard: result == 'water' || result == 'ob' ? 1 : 0};
        this.log.push({club: G.CLUBS[s.club][0], spin: s.spin, lie, from: d0, want: s.want,
            power: s.power, result, to: this.pinDist(), tree: G.treeHit});
        if (ev == G.EV_HOLED || this.strokes >= this.h.par + cfg.maxOver)
            this.done = true;
        return {obs: this.done ? null : this.observe(), reward, done: this.done,
            info: {result, strokes: this.strokes, par: this.h.par, holed: ev == G.EV_HOLED}};
    }

    // Penalty drop, verbatim from game.js updateFlight: walk back along the shot
    // from the last safe point until the ball can stay.
    hazardDrop()
    {
        const G = this.G, b = G.ball, safe = G.ballSafe, start = G.shotStart;
        const dx = start.x-safe.x, dz = start.z-safe.z;
        const dl = Math.hypot(dx, dz) || 1;
        for (let d = 2; ; d += 2)
        {
            const t = Math.min(d, dl);
            b.x = safe.x + dx/dl*t;
            b.z = safe.z + dz/dl*t;
            const g = G.groundAt(b.x, b.z);
            b.y = g.h;
            if (t == dl || g.s < G.SURF_WATER && g.s != G.SURF_GREEN
                && Math.hypot(...G.slopeAt(b.x, b.z))*G.GRAV < G.SURF_PHYS[g.s][2])
                break;
        }
    }
}
