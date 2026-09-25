#!/usr/bin/env node
// Evaluate a checkpoint on full 18-hole rounds it never trained on.
// usage: node rl/eval.mjs runs/<name>/best.json [--set classic|remix|both]
//        [--rounds 8] [--stochastic] [--card] [--shots]
//   --card   print the hole-by-hole scorecard of the first round
//   --shots  also print every shot of that round
import { parseArgs } from 'node:util';
import { Pool, summarise } from './pool.mjs';
import { loadCheckpoint } from './checkpoint.mjs';
import { evalSet } from './courses.mjs';

const {values: a, positionals} = parseArgs({allowPositionals: true, options: {
    set: {type: 'string', default: 'both'},
    rounds: {type: 'string', default: '8'},
    stochastic: {type: 'boolean', default: false},
    card: {type: 'boolean', default: false},
    shots: {type: 'boolean', default: false},
    workers: {type: 'string'},
}});
const file = positionals[0] ?? 'runs/r1/best.json';
const ck = loadCheckpoint(file);
const pool = new Pool({workers: a.workers && +a.workers, arch: ck.arch});
pool.params.set(ck.params);
console.log(`${file}  (iter ${ck.meta.iter}, ${ck.meta.totalSteps} steps)`);

const SCORE = ['albatross', 'eagle', 'birdie', 'par', 'bogey', 'double', 'triple+'];
for (const set of a.set == 'both' ? ['classic', 'remix'] : [a.set])
{
    const res = await pool.evaluate(evalSet(set, +a.rounds), !a.stochastic);
    const s = summarise(res);
    const dist = new Array(SCORE.length).fill(0);
    for (const e of res) dist[Math.min(Math.max(e.strokes - e.par + 3, 0), 6)]++;
    console.log(`${set}: ${s.rounds} rounds, ${s.toPar >= 0 ? '+' : ''}${s.toPar.toFixed(2)} to par`
        + ` (mean ${s.strokes.toFixed(1)}, best ${s.best}, worst ${s.worst}),`
        + ` ${s.penalties.toFixed(1)} penalties, ${s.pickups.toFixed(2)} pickups per round`);
    console.log('  ' + SCORE.map((n, i)=> `${n} ${(dist[i]/res.length*100).toFixed(1)}%`).join('  '));
    if (a.card || a.shots)
        for (const e of res.filter(e => e.round == 0))
        {
            console.log(`  hole ${String(e.hole+1).padStart(2)} par ${e.par}: ${e.strokes}`
                + (e.penalties ? ` (${e.penalties} pen)` : ''));
            if (a.shots)
                for (const sh of e.shots)
                    console.log(`      ${sh.club.padEnd(3)} ${['back', '    ', 'top '][sh.spin+1]} from ${sh.lie.padEnd(7)}`
                        + ` ${sh.from.toFixed(1).padStart(6)}yd  asked ${sh.want.toFixed(1).padStart(6)}`
                        + ` -> ${sh.result.padEnd(7)} ${sh.to.toFixed(1).padStart(6)}yd left${sh.tree ? '  TREE' : ''}`);
        }
}
await pool.close();
