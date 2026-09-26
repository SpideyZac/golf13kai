// THE SOLVER: when the pin is in range, find the exact swing that holes the
// ball, wind, slopes, spin and all, by playing trial shots through the game's
// own launchBall/ballUpdate and steering aim and power until one drops. Then
// replay that swing across the swing-meter noise to learn how often it
// really goes in. The agent sees the result in the observation and can play
// the shot by choosing CLUB_SOLVE (obs.mjs). Spec: docs/SPEC.md (Solver).
//
// Pure like obs.mjs: it only talks to the game through G (rl/sim/api.mjs),
// and it leaves every flight global exactly as it found it, so the browser
// agent can run it on the live game between frames. golfsim/src/solver.rs is
// the Rust twin and must agree bit for bit (rl/test/parity.test.mjs), so keep
// the operation order when editing either.

// trial shots per candidate while steering (most converge in 3-5)
export const SOLVE_ITERS = 8;
// ball-flight frames per trial: the env's own limit (30 simulated seconds)
const SIM_FRAMES = 60*30;
// impact errors under this snap to a perfect strike (launchBall)
const PERFECT = .02;
// the sand wedge: the chip candidate when the game hands over the putter off the green
const SW = 9;
// the fallback searches this many clubs either side of the game's (see fallback())
const FALLBACK_CLUBS = 2;
// the fallback's shaped swings: the impact swung for (+ early, - late), which
// curves the flight (launchBall's ballCurve = impact*22) around what blocks
// the straight line
const CURVES = [-.12, -.06, .06, .12];

const TAU = 2*Math.PI;

// the impact launchBall actually plays
const snap = (e)=> Math.abs(e) < PERFECT ? 0 : e;

// Every global a shot writes (launchBall, shotBegin, ballUpdate, flyStep, rollStep).
function save(G)
{
    const b = G.ball;
    return {x: b.x, y: b.y, z: b.z, vx: b.vx, vy: b.vy, vz: b.vz, air: G.ballAir, rolling: G.ballRolling,
        bounces: G.bounces, pinHit: G.pinHit, pinOut: G.pinOut, spin: G.ballSpin, curve: G.ballCurve,
        shotDir: G.shotDir, treeHit: G.treeHit, treeCool: G.treeCool, event: G.ballEvent,
        shotStart: G.shotStart, ballSafe: G.ballSafe, rollRest: G.rollRest};
}

function restore(G, s)
{
    const b = G.ball;
    b.x = s.x; b.y = s.y; b.z = s.z; b.vx = s.vx; b.vy = s.vy; b.vz = s.vz;
    G.ballAir = s.air; G.ballRolling = s.rolling; G.bounces = s.bounces; G.pinHit = s.pinHit; G.pinOut = s.pinOut;
    G.ballSpin = s.spin; G.ballCurve = s.curve; G.shotDir = s.shotDir; G.treeHit = s.treeHit;
    G.treeCool = s.treeCool; G.ballEvent = s.event; G.shotStart = s.shotStart; G.ballSafe = s.ballSafe;
    G.rollRest = s.rollRest;
}

// The swing-meter noise as weighted samples [impact error, aim error, weight]:
// each uniform on +-n, as the centres of its quarters. The impact error adds
// to the impact swung for, and the sum snaps to perfect under .02, so from a
// straight swing (impact 0) the two inner samples ARE the perfect strike,
// half the weight at the default noise. No aim sample is 0: that one is the
// solution itself and always drops, which would credit a 150-yard iron with a
// third of a chance. Putts have no aim noise.
function noiseSamples(impactNoise, aimNoise, putt)
{
    const q = (n)=> n > 0 ? [[-n*.75, .25], [-n*.25, .25], [n*.25, .25], [n*.75, .25]] : [[0, 1]];
    const imp = q(impactNoise), aim = putt ? [[0, 1]] : q(aimNoise);
    const out = [];
    for (const [i, wi] of imp)
        for (const [a, wa] of aim)
            out.push([i, a, wi*wa]);
    return out;
}

export class Solver
{
    constructor(G) { this.G = G; }

    // One trial swing from where the ball lies, exactly as GolfEnv.step plays
    // it (pin rule included); returns {ev, x, z} and restores everything.
    trial(club, spin, yaw, power, lm, impact, aimErr, pinOut)
    {
        const G = this.G, s = save(G);
        G.pinOut = pinOut;
        G.treeHit = 0;
        G.launchBall(club, power, impact, spin, yaw + aimErr, lm);
        for (let t = 0; t < SIM_FRAMES && !G.ballEvent; ++t)
            G.ballUpdate();
        const r = {ev: G.ballEvent || G.EV_STOPPED, x: G.ball.x, z: G.ball.z};
        restore(G, s);
        return r;
    }

    // The swings worth solving from here, as [club, spin, impact]: the putter
    // where it can roll there, and the game's own club with no spin and with
    // backspin (or, when the game hands over the putter off the green, a sand
    // wedge chip), all struck straight.
    candidates(auto, lie, d)
    {
        const G = this.G, c = [];
        if (d < G.PUTT_MAX && (lie == G.SURF_GREEN || lie == G.SURF_FAIRWAY || lie == G.SURF_TEE))
            c.push([G.CLUB_PUTTER, 0, 0]);
        const full = auto == G.CLUB_PUTTER ? (lie == G.SURF_GREEN ? -1 : SW) : auto;
        if (full >= 0) c.push([full, 0, 0], [full, -1, 0]);
        return c;
    }

    // When none of those holes: the clubs up to FALLBACK_CLUBS either side of
    // the game's (the wedge's, when it handed over the putter), longest first.
    // First straight with no spin, backspin and topspin - a longer club or
    // topspin for a pin the game's club cannot run up to, a shorter one to
    // stop on a ridge - then hooked and sliced (CURVES) with no spin and
    // backspin, to bend around a tree on the line.
    // Measured on in-range lies: 82% -> 86% solved straight, 90% with curves.
    fallback(auto, tried)
    {
        const G = this.G, c = [], base = auto == G.CLUB_PUTTER ? SW : auto;
        const lo = Math.max(0, base - FALLBACK_CLUBS), hi = Math.min(SW, base + FALLBACK_CLUBS);
        const add = (k, spin, imp)=>
            tried.some(([tc, ts, ti])=> tc == k && ts == spin && ti == imp) || c.push([k, spin, imp]);
        for (let k = lo; k <= hi; ++k)
            for (const spin of [0, -1, 1]) add(k, spin, 0);
        for (let k = lo; k <= hi; ++k)
            for (const spin of [0, -1])
                for (const imp of CURVES) add(k, spin, imp);
        return c;
    }

    // Steer one (club, spin, impact) onto the cup: {yaw, power, want} of a
    // holing swing, or null.
    steer(club, spin, impact, d, pdir, pinOut)
    {
        const G = this.G, b = G.ball, putt = club == G.CLUB_PUTTER;
        const lm = G.lieMul(club), max = putt ? G.PUTT_MAX : G.CLUBS[club][1]*lm;
        const lo = putt ? .005 : .02;
        let yaw = pdir, want = d;
        for (let k = 0; k < SOLVE_ITERS; ++k)
        {
            const power = Math.min(Math.max(want/max, lo), 1);
            const r = this.trial(club, spin, yaw, power, lm, impact, 0, pinOut);
            if (r.ev == G.EV_HOLED) return {yaw, power, want};
            const dx = r.x - b.x, dz = r.z - b.z, got = Math.hypot(dx, dz);
            // it went nowhere (a tree, a wall), or it cannot get there at full power
            if (got < .5 || (power == 1 && got < d)) return null;
            // steer: turn by the angle it missed by, scale by how far it went
            let turn = pdir - Math.atan2(dx, dz);
            if (turn > Math.PI) turn -= TAU;
            else if (turn < -Math.PI) turn += TAU;
            yaw += turn;
            want *= d/got;
        }
        return null;
    }

    // Solve the current lie. cfg: {impactNoise, aimNoise}. Returns
    // {tried, found, club, spin, impact, yaw, power, lm, want, pHole, pHazard, leave}.
    solve({impactNoise, aimNoise})
    {
        const G = this.G, b = G.ball, px = G.hole.pin.x, pz = G.hole.pin.z;
        const d = Math.hypot(px - b.x, pz - b.z), pdir = Math.atan2(px - b.x, pz - b.z);
        const auto = G.autoClub();
        const reach = auto == G.CLUB_PUTTER ? G.PUTT_MAX : G.CLUBS[auto][1]*G.lieMul(auto);
        // IN RANGE: the same test that makes the game aim its default shot at the pin
        if (!(d < reach + 20)) return {tried: 0, found: 0};
        const lie = G.groundAt(b.x, b.z).s;
        const pinOut = d < 15 && lie == G.SURF_GREEN ? 1 : 0;
        let best = {tried: 1, found: 0};
        const cands = this.candidates(auto, lie, d);
        // every candidate is solved and the best odds win; the fallback stops
        // at its first solution
        for (const pass of [0, 1])
        for (const [club, spin, impact] of pass ? this.fallback(auto, cands) : cands)
        {
            if (pass && best.found) break;
            const putt = club == G.CLUB_PUTTER, lm = G.lieMul(club);
            const hit = this.steer(club, spin, impact, d, pdir, pinOut);
            if (!hit) continue;
            // the solution across the swing noise
            let pHole = 0, pHazard = 0, leave = 0;
            // samples that snap to the same strike play the same shot: one trial each
            const seen = [];
            for (const [imp, aim, w] of noiseSamples(impactNoise, aimNoise, putt))
            {
                // the swing that plays exactly as the solution drops
                if (snap(impact + imp) == snap(impact) && aim == 0) { pHole += w; continue; }
                const e = snap(impact + imp);
                let r = seen.find(q => q.e == e && q.aim == aim)?.r;
                if (!r)
                {
                    r = this.trial(club, spin, hit.yaw, hit.power, lm, impact + imp, aim, pinOut);
                    seen.push({e, aim, r});
                }
                if (r.ev == G.EV_HOLED) pHole += w;
                else
                {
                    if (r.ev == G.EV_WATER || r.ev == G.EV_OB) pHazard += w;
                    leave += w*Math.hypot(px - r.x, pz - r.z);
                }
            }
            if (!best.found || pHole > best.pHole || (pHole == best.pHole && leave < best.leave))
                best = {tried: 1, found: 1, club, spin, impact, yaw: hit.yaw, power: hit.power, lm, want: hit.want,
                    pHole, pHazard, leave};
        }
        return best;
    }
}
