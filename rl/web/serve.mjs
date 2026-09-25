#!/usr/bin/env node
// Static server over the repo root, so the page can load both the game
// (Golf13K/) and the agent (rl/, models/). No dependencies.
// usage: node rl/web/serve.mjs [port]   then open http://localhost:8013/rl/web/?auto=1
import http from 'node:http';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, normalize, extname } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const PORT = +process.argv[2] || 8013;
const MIME = {'.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
    '.json': 'application/json', '.png': 'image/png', '.css': 'text/css'};

http.createServer((req, res)=>
{
    let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (p.endsWith('/')) p += 'index.html';
    const file = normalize(join(ROOT, p));
    if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end(); }
    fs.readFile(file, (err, data)=>
    {
        if (err) { res.writeHead(404); return res.end('not found'); }
        res.writeHead(200, {'content-type': MIME[extname(file)] || 'application/octet-stream', 'cache-control': 'no-store'});
        res.end(data);
    });
}).listen(PORT, ()=> console.log(`http://localhost:${PORT}/rl/web/?auto=1`));
