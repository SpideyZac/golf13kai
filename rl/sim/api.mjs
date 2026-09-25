// The game API surface the agent uses, as a function BODY. Evaluated inside
// the game's scope it returns an object of getters/setters over the game's
// lexical bindings: the Node loader appends it to the concatenated sources,
// and the browser runs it with `new Function` at global scope, where the
// game's classic-script `let`s are visible. One list, two hosts.
export const API_BODY = `
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
