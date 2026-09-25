#!/usr/bin/env node
// Learning curve of a run as a standalone SVG (light/dark aware): held-out
// eval score per iteration, classic and remix, against the scripted bot.
// usage: node rl/plot.mjs runs/<name>/log.csv out.svg [--title "..."]
import fs from 'node:fs';
import { parseArgs } from 'node:util';

const {values: a, positionals: [csv, out]} = parseArgs({allowPositionals: true,
    options: {title: {type: 'string'}, scripted: {type: 'string', default: '-3.5,-1'}}});
const rows = fs.readFileSync(csv, 'utf8').trim().split('\n');
const head = rows.shift().split(',');
const col = (k)=> head.indexOf(k);
const pts = rows.map(r => r.split(',')).filter(r => r[col('evalClassic')] !== '')
    .map(r => ({it: +r[col('iter')], c: +r[col('evalClassic')], r: +r[col('evalRemix')]}));
const [bc, br] = a.scripted.split(',').map(Number);

const W = 720, H = 360, L = 56, R = 110, T = 40, B = 44;
const xMax = Math.max(...pts.map(p => p.it));
const ys = pts.flatMap(p => [p.c, p.r]).concat(bc, br);
const yLo = Math.floor(Math.min(...ys)/5)*5, yHi = Math.min(Math.ceil(Math.max(...ys)/5)*5, 30);
const X = (v)=> L + v/xMax*(W - L - R), Y = (v)=> T + (Math.min(v, yHi) - yLo)/(yHi - yLo)*(H - T - B);
const line = (k)=> pts.map((p, i)=> `${i ? 'L' : 'M'}${X(p.it).toFixed(1)},${Y(p[k]).toFixed(1)}`).join('');
let grid = '';
for (let v = yLo; v <= yHi; v += 5)
    grid += `<line class="grid" x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}"/>`
        + `<text class="tick" x="${L-8}" y="${Y(v)+4}" text-anchor="end">${v > 0 ? '+' : ''}${v}</text>`;
const step = xMax > 200 ? 100 : 50;
for (let v = 0; v <= xMax; v += step)
    grid += `<text class="tick" x="${X(v)}" y="${H-B+18}" text-anchor="middle">${v}</text>`;
const last = pts.at(-1);
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" font-family="system-ui,sans-serif" role="img" aria-label="${a.title ?? 'Learning curve'}">
<style>
  .bg{fill:#fcfcfb}.grid{stroke:#e4e3dc;stroke-width:1}.tick,.lab{fill:#6b6a63;font-size:12px}.ttl{fill:#1a1a19;font-size:15px;font-weight:600}
  .c{stroke:#2a78d6}.r{stroke:#eb6834}.s{stroke-width:1.5;stroke-dasharray:4 4}.dl{fill:#1a1a19;font-size:12px}
  @media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.grid{stroke:#3a3a37}.tick,.lab{fill:#c3c2b7}.ttl,.dl{fill:#fff}.c{stroke:#3987e5}.r{stroke:#d95926}}
</style>
<rect class="bg" width="${W}" height="${H}" rx="8"/>
<text class="ttl" x="${L}" y="24">${a.title ?? 'Held-out eval, strokes to par per 18-hole round (up is better)'}</text>
${grid}
<line class="grid" x1="${L}" x2="${W-R}" y1="${Y(0)}" y2="${Y(0)}" style="stroke-width:1.5"/>
<path class="c s" fill="none" d="M${L},${Y(bc)}H${W-R}"/><path class="r s" fill="none" d="M${L},${Y(br)}H${W-R}"/>
<path class="c" fill="none" stroke-width="2" stroke-linejoin="round" d="${line('c')}"/>
<path class="r" fill="none" stroke-width="2" stroke-linejoin="round" d="${line('r')}"/>
<text class="dl" x="${W-R+8}" y="${Y(last.c)+4}">classic ${last.c}</text>
<text class="dl" x="${W-R+8}" y="${Y(last.r)+4}">remix ${last.r}</text>
<text class="lab" x="${(L+W-R)/2}" y="${H-8}" text-anchor="middle">training iteration (~15k strokes each) · dashed: scripted bot</text>
</svg>`;
fs.writeFileSync(out, svg);
console.log(`${out}: ${pts.length} eval points`);
