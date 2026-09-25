// The observation and action decoding, shared by the Node env (rl/env.mjs)
// and the in-browser agent (rl/web/ai.mjs). Pure: it only talks to the game
// through G, the accessor object from rl/sim/api.mjs. Contract: docs/SPEC.md -
// bump OBS_VERSION with any change to what observe() writes or decode() means.

export const OBS_VERSION = 3;
export const CLASSIC_SEED = 1113;
export const N_CLUBS = 11, N_SPIN = 3;
export const CLUB_NAMES = ['1W', '3W', '5W', '3i', '5i', '7i', '9i', '13i', 'PW', 'SW', 'PT'];

// Action decoding (see SPEC.md), RESIDUAL TO THE GAME'S DEFAULT SHOT (the aim
// line and target the game sets up for a player, see reference()):
//   yaw  = ref.dir + AIM_SCALE*aimRaw            (radians)
//   want = ref.dist * exp(DIST_SCALE*distRaw)    (yards the shot is asked to go)
export const AIM_SCALE = .35, AIM_CLIP = 3;
export const DIST_SCALE = .35, DIST_LO = -4, DIST_HI = 2;

// Two frames. The PIN frame (+fwd from the ball to the pin, +lat to its right)
// holds grid A, scaled by the pin distance: the green's contour on a putt, the
// whole hole on a drive. The AIM frame (+fwd along the game's default aim line,
// +lat the side a +aim action moves the shot) holds everything about THIS
// shot: grid B over full-swing landing zones, the line-of-fire rays, the
// flight preview and the centreline ahead.
const GA_FWD = [-.1, .1, .25, .4, .55, .7, .8, .9, 1, 1.1, 1.25, 1.5];
const GA_LAT = [-.4, -.25, -.12, -.05, 0, .05, .12, .25, .4];
const GB_FWD = Array.from({length: 15}, (_, i)=> 20 + 20*i);
const GB_LAT = [-50, -30, -15, 0, 15, 30, 50];
const LOOK = Array.from({length: 10}, (_, i)=> 30 + 30*i);
// Line of fire: rays at these aim offsets (radians, like the aim action) out
// to RAY_LEN yards. Each reports the first tree canopy on the line and the
// steepest rise of the ground along it - what stands between the ball and a
// clean strike, at a resolution the grids cannot give right next to the ball.
const RAYS = Array.from({length: 13}, (_, i)=> (i - 6)*Math.PI/18);
const RAY_LEN = 80, RAY_STEPS = [4, 8, 14, 22, 32, 45, 60];
const CH = 6; // grid channels: hazard, sand, green, short grass, tree, height

export const OBS_DIM = 7 + 2 + 2 + 3 + 3 + 2 + 1 + 3 + 11 + 3 + 3 + 2*RAYS.length + 6
    + 8 + 5 + 2*LOOK.length + CH*(GA_FWD.length*GA_LAT.length + GB_FWD.length*GB_LAT.length);

// THE ESCAPE RULE for deterministic play: after a full swing that moved the
// ball under ESCAPE_YD or found a hazard, the next shot is SAMPLED from the
// policy instead of taken at its mode. The mode of a nearly unchanged state is
// a nearly unchanged shot, so without this a deterministic agent can replay
// the same strike into the same tree until the mercy rule. Training always
// samples and never needs it.
export const ESCAPE_YD = 10;
export const isStuck = (prev, lastWasPutt)=> !lastWasPutt && (prev.hazard || prev.moved < ESCAPE_YD);

export const clip = (v, lo, hi)=> v < lo ? lo : v > hi ? hi : v;

const TREE_CELL = 8;

// One per game instance. Call newHole() after every genHole; observe() reads
// the round state it is handed: {strokes, prev: {moved, tree, hazard}, maxOver}.
export class Observer
{
    constructor(G)
    {
        this.G = G;
        this.obs = new Float32Array(OBS_DIM);
    }

    newHole()
    {
        this.h = this.G.hole;
        this.buildTreeHash();
    }

    buildTreeHash()
    {
        this.trees = new Map();
        for (const t of this.h.near)
        {
            const r = Math.ceil(t.s/TREE_CELL);
            const cx = Math.floor(t.x/TREE_CELL), cz = Math.floor(t.z/TREE_CELL);
            for (let i = -r; i <= r; ++i)
                for (let j = -r; j <= r; ++j)
                {
                    const k = (cx+i)*65536 + (cz+j);
                    let a = this.trees.get(k);
                    a || this.trees.set(k, a = []);
                    a.push(t);
                }
        }
    }

    treeAt(x, z)
    {
        const a = this.trees.get(Math.floor(x/TREE_CELL)*65536 + Math.floor(z/TREE_CELL));
        if (a) for (const t of a)
            if ((x-t.x)**2 + (z-t.z)**2 < t.s*t.s) return 1;
        return 0;
    }

    pinDist() { const G = this.G; return Math.hypot(G.hole.pin.x-G.ball.x, G.hole.pin.z-G.ball.z); }
    pinDir() { const G = this.G; return Math.atan2(G.hole.pin.x-G.ball.x, G.hole.pin.z-G.ball.z); }

    // THE GAME'S DEFAULT SHOT, as enterAim sets it up for a player: the club
    // autoClub picks, aimed by aimDefault - at the pin when that club can
    // plausibly reach it (its max from this lie plus 20yd of roll), otherwise
    // at a lay-up one carry (x.95) down the centreline. The pin line on a
    // hairpin runs through the woods; this line follows the fairway.
    reference()
    {
        const G = this.G, b = G.ball, h = this.h;
        const club = G.autoClub(), d = this.pinDist();
        const reach = club == G.CLUB_PUTTER ? G.PUTT_MAX : G.CLUBS[club][1]*G.lieMul(club);
        if (d < reach + 20)
            return {club, dir: this.pinDir(), dist: d};
        G.distToPath(b.x, b.z);
        const t = G.pathPointAt(Math.min(G.lastAlong + reach*.95, h.len));
        return {club, dir: Math.atan2(t.x-b.x, t.z-b.z), dist: Math.max(1, Math.hypot(t.x-b.x, t.z-b.z))};
    }

    observe({strokes, prev, maxOver})
    {
        const G = this.G, h = this.h, b = G.ball, o = this.obs;
        let i = 0;
        const put = (v)=> { o[i++] = v; };
        const d = this.pinDist(), pdir = this.pinDir();
        const ref = this.ref = this.reference(), dir = ref.dir, auto = ref.club;
        const frame = (a)=>
        {
            const sn = Math.sin(a), cs = Math.cos(a);
            return {
                toW: (f, l)=> [b.x + sn*f + cs*l, b.z + cs*f - sn*l],
                toP: (x, z)=> { const dx = x-b.x, dz = z-b.z; return [dx*sn + dz*cs, dx*cs - dz*sn]; },
                vec: (x, z)=> [x*sn + z*cs, x*cs - z*sn],
            };
        };
        const A = frame(dir), P = frame(pdir);
        const g = G.groundAt(b.x, b.z), hb = g.h;

        for (let s = 0; s < 7; ++s) put(g.s == s ? 1 : 0);                    // 7 lie
        put(G.lieMul(0)); put(G.lieMul(8));                                    // 2 lie multipliers
        const wr = h.wind.a - dir;
        put(h.wind.s*Math.cos(wr)/8); put(h.wind.s*Math.sin(wr)/8);           // 2 wind along/across the aim
        for (const p of [3, 4, 5]) put(h.par == p ? 1 : 0);                    // 3 par
        put(strokes/10); put((h.par + maxOver - strokes)/10); put(h.hills);    // 3 round state
        const [sx, sz] = A.vec(...G.slopeAt(b.x, b.z));
        put(sx*5); put(sz*5);                                                  // 2 slope at ball
        put(Math.tanh((G.heightAt(h.pin.x, h.pin.z) - hb)/10));                // 1 pin height
        put(d/300); put(Math.log1p(d)/6); put(d < 45 ? 1 : 0);                 // 3 pin distance
        for (let c = 0; c < 11; ++c) put(c == auto ? 1 : 0);                   // 11 game's club
        const dp = G.distToPath(b.x, b.z), along = G.lastAlong;
        put(dp/60); put(along/h.len); put((h.len - along)/300);                // 3 path progress
        put(Math.log1p(prev.moved)/6); put(prev.tree); put(prev.hazard);       // 3 last shot
        this.rays(dir, hb, put);                                               // 26 line of fire
        // 6: the default shot against the pin
        const [pf, pl] = A.toP(h.pin.x, h.pin.z);
        put(pf/300); put(pl/300); put(Math.cos(pdir - dir)); put(Math.sin(pdir - dir));
        put(ref.dist/300); put(Math.log(ref.dist/Math.max(d, 1)));

        // 8: the putt preview, twice (to the cup, and to the bar's default
        // top), aimed at the pin and read in the pin frame
        for (const k of [1, G.PUTT_OVER])
        {
            if (d < 45)
            {
                const t = this.puttPreview(d*k, pdir);
                const [f, l] = P.toP(t.x, t.z);
                put(Math.tanh((f - d)/Math.max(d, 1))); put(Math.tanh(l/Math.max(d, 1)*5));
                put(Math.tanh(t.lat/Math.max(d, 1)*5)); put(t.best < G.HOLE_R ? 1 : 0);
            }
            else { put(0); put(0); put(0); put(0); }
        }
        // 5: the flight preview of the default shot, in still air
        if (auto != G.CLUB_PUTTER)
        {
            const lm = G.lieMul(auto);
            const power = Math.min(1, ref.dist/(G.CLUBS[auto][1]*lm));
            const p = G.predictLanding(auto, dir, 0, lm, power);
            const [f, l] = A.toP(p.x, p.z);
            put(Math.tanh((f - ref.dist)/50)); put(Math.tanh(l/30)); put(p.hit ? 1 : 0);
            const s = G.surfaceAt(p.x, p.z);
            put(s >= G.SURF_WATER ? 1 : 0); put(s == G.SURF_BUNKER ? 1 : 0);
        }
        else { put(0); put(0); put(0); put(0); put(0); }
        // 20: the centreline ahead
        for (const a of LOOK)
        {
            const p = G.pathPointAt(Math.min(along + a, h.len));
            const [f, l] = A.toP(p.x, p.z);
            put(f/300); put(l/300);
        }

        const cell = (x, z, hs)=>
        {
            const s = G.surfaceAt(x, z);
            put(s >= G.SURF_WATER ? 1 : 0);
            put(s == G.SURF_BUNKER ? 1 : 0);
            put(s == G.SURF_GREEN ? 1 : 0);
            put(s == G.SURF_FAIRWAY || s == G.SURF_TEE ? 1 : 0);
            put(this.treeAt(x, z));
            put(Math.tanh((G.heightAt(x, z) - hb)/hs));
        };
        const sc = Math.max(d, 4);
        for (const f of GA_FWD)
            for (const l of GA_LAT)
                cell(...P.toW(f*sc, l*sc), Math.max(.5, sc*.05));
        for (const f of GB_FWD)
            for (const l of GB_LAT)
                cell(...A.toW(f, l), 15);
        if (i != OBS_DIM) throw new Error(`obs size ${i} != ${OBS_DIM}`);
        return o;
    }

    rays(dir, hb, put)
    {
        const G = this.G, b = G.ball;
        const near = [];
        for (const t of this.h.near)
            if (Math.hypot(t.x-b.x, t.z-b.z) < RAY_LEN + t.s) near.push(t);
        for (const off of RAYS)
        {
            const a = dir + off, ux = Math.sin(a), uz = Math.cos(a);
            let hit = RAY_LEN;
            for (const t of near)
            {
                const dx = t.x-b.x, dz = t.z-b.z, al = dx*ux + dz*uz;
                const pp = Math.abs(dx*uz - dz*ux);
                if (al > 0 && pp < t.s) hit = Math.min(hit, Math.max(0, al - Math.sqrt(t.s*t.s - pp*pp)));
            }
            let rise = -1;
            for (const r of RAY_STEPS)
                rise = Math.max(rise, Math.tanh((G.heightAt(b.x + ux*r, b.z + uz*r) - hb)/r*2));
            put(1 - hit/RAY_LEN); put(rise);
        }
    }

    // The game's putt preview (predictLanding's roll loop), plus the closest
    // pass to the cup - the same line the player watches bend on screen.
    puttPreview(dist, dir)
    {
        const G = this.G, px = G.hole.pin.x, pz = G.hole.pin.z;
        const sn = Math.sin(dir), cs = Math.cos(dir);
        const r = G.puttVel({x: G.ball.x, y: G.ball.y, z: G.ball.z}, dist, dir);
        let best = 1e9, lat = 0;
        G.rollRest = 0;
        for (let i = 0; i < 600 && !G.rollRest; ++i)
        {
            G.rollStep(r);
            const cd = Math.hypot(r.x-px, r.z-pz);
            if (cd < best) { best = cd; lat = (r.x-px)*cs - (r.z-pz)*sn; }
        }
        return {x: r.x, z: r.z, best, lat};
    }

    // Decode a raw action into the shot the game will play. Relative to the
    // reference of the LAST observe() - the state the action was chosen in.
    decode({club, spin, aim, dist})
    {
        const G = this.G, ref = this.ref ?? this.reference();
        const yaw = ref.dir + AIM_SCALE*clip(aim, -AIM_CLIP, AIM_CLIP);
        const want = Math.max(.3, ref.dist*Math.exp(DIST_SCALE*clip(dist, DIST_LO, DIST_HI)));
        const lm = G.lieMul(club);
        const power = club == G.CLUB_PUTTER
            ? clip(want/G.PUTT_MAX, .005, 1)
            : clip(want/(G.CLUBS[club][1]*lm), .02, 1);
        return {club, spin: club == G.CLUB_PUTTER ? 0 : spin - 1, yaw, power, lm, want};
    }
}
