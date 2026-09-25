#!/usr/bin/env node
// Reference scores the RL agent is measured against, on the same eval sets:
//   naive    - the game's own defaults: the pre-selected club (autoClub) aimed
//              straight at the pin, asked for exactly the pin distance
//   scripted - the game's hand-written dev bot (Golf13K/game/tools/sim.mjs):
//              wind-compensated layups along the centreline and a putt solver
//              that rolls ~55 trial putts. Its wind is unseeded, so its
//              numbers carry run-to-run noise.
// usage: node rl/baseline.mjs [--rounds 4]
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { GolfEnv, CLASSIC_SEED } from './env.mjs';
import { GAME_ROOT } from './sim/loader.mjs';
import { evalSet } from './courses.mjs';
import { summarise } from './pool.mjs';

const {values: a} = parseArgs({options: {rounds: {type: 'string', default: '4'}}});
const R = +a.rounds;

function naive(set)
{
    const env = new GolfEnv();
    return summarise(evalSet(set, R).map(spec =>
    {
        env.reset(spec);
        let r;
        do
        {
            const club = env.G.autoClub();
            r = env.step({club, spin: 1, aim: 0, dist: 0});
        } while (!r.done);
        return {round: spec.round, strokes: env.strokes, par: env.h.par, penalties: env.penalties,
            holed: env.log.at(-1).result == 'holed'};
    }));
}

function scripted(set)
{
    const res = [];
    for (let r = 0; r < R; ++r)
    {
        const args = set == 'classic' ? [String(CLASSIC_SEED)] : [String(r + 1), '--remix'];
        const out = execFileSync(process.execPath, [join(GAME_ROOT, 'game/tools/sim.mjs'), ...args], {encoding: 'utf8'});
        const m = out.match(/TOTAL (\d+) \(par (\d+)/);
        const st = JSON.parse(out.match(/stats (.*)/)[1]);
        res.push({round: r, strokes: +m[1], par: +m[2], penalties: st.water + st.ob, holed: true});
    }
    return summarise(res);
}

const fmt = (s)=> `${s.toPar >= 0 ? '+' : ''}${s.toPar.toFixed(2)} to par  (best ${s.best}, worst ${s.worst}, penalties ${s.penalties.toFixed(1)}/round)`;
for (const set of ['classic', 'remix'])
{
    console.log(`${set} x${R}`);
    console.log(`  naive    ${fmt(naive(set))}`);
    console.log(`  scripted ${fmt(scripted(set))}`);
}
