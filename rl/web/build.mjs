#!/usr/bin/env node
// Builds the agent-playing game into ONE self-contained html file, like
// Golf13K's own build/index.html: no server, no other files, open it straight
// from disk and the agent plays a round. No dependencies and no minifier: the
// game ships its dev sources as-is (the dev bot's autoplay lives in them), the
// agent's ES modules are bundled into one inline module, and the checkpoint is
// embedded as window.RL_MODEL (see rl/web/ai.mjs).
//
// The script list is read from rl/web/index.html, so that page stays the one
// place it is written down.
//
// usage: node rl/web/build.mjs [--model models/agent.json] [--out build/index.html]
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, relative } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '../..');
const argv = process.argv.slice(2);
const opt = (name, def)=> { const i = argv.indexOf('--' + name); return i < 0 ? def : argv[i+1]; };
const MODEL = resolve(opt('model', join(ROOT, 'models/agent.json')));
const OUT = resolve(opt('out', join(ROOT, 'build/index.html')));

const fail = (msg)=> { console.error('build failed: ' + msg); process.exit(1); };
const read = (file)=> fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');

// Text inside an inline <script> ends at the first "</script", and "<!--"
// can switch the tokenizer into its escaped state. Neither may appear.
function inlineSafe(code, what)
{
    if (/<\/script/i.test(code) || code.includes('<!--'))
        fail(`${what} contains "</script" or "<!--", which cannot be inlined`);
    return code;
}

///////////////////////////////////////////////////////////////////////////////
// A minimal ES module bundler for rl/'s own modules. Each module becomes a
// function scope that returns its exports, and each import becomes a
// destructure of one of those. It knows only the forms rl/ uses (named
// imports from relative paths, `export` in front of a top-level declaration)
// and fails loudly on anything else.

// the names a `const`/`let` statement declares: split at top-level commas
// up to the top-level `;`, skipping strings, templates and nested brackets
function declaredNames(src, from)
{
    const names = [];
    let depth = 0, start = from;
    for (let i = from; i < src.length; ++i)
    {
        const c = src[i];
        if (c == '"' || c == "'" || c == '`')
        {
            for (++i; src[i] != c; ++i)
                if (src[i] == '\\') ++i;
        }
        else if ('([{'.includes(c)) ++depth;
        else if (')]}'.includes(c)) --depth;
        else if (!depth && (c == ',' || c == ';'))
        {
            const m = /^\s*([\w$]+)\s*=/.exec(src.slice(start, i));
            if (!m) fail('cannot read the declaration at ' + JSON.stringify(src.slice(start, start + 40)));
            names.push(m[1]);
            start = i + 1;
            if (c == ';') return names;
        }
    }
    fail('unterminated declaration at ' + JSON.stringify(src.slice(from, from + 40)));
}

function bundle(entry)
{
    const mods = new Map(), order = [], visiting = new Set();
    const visit = (file)=>
    {
        if (mods.has(file)) return mods.get(file);
        if (visiting.has(file)) fail('import cycle through ' + relative(ROOT, file));
        visiting.add(file);
        let src = read(file);
        const name = relative(ROOT, file).replace(/\\/g, '/');
        const imports = [];

        // imports -> destructures of the dependency's export object
        src = src.replace(/^import\s*\{([^}]*)\}\s*from\s*'(\.[^']*)';?[ \t]*$/mg, (_, names, spec)=>
        {
            const dep = visit(resolve(dirname(file), spec));
            const list = names.split(',').map(s => s.trim()).filter(Boolean)
                .map(s => s.split(/\s+as\s+/));
            for (const [n] of list)
                imports.push([n, dep]);
            return `const {${list.map(([n, as]) => as ? `${n}: ${as}` : n).join(', ')}} = ${dep.id};`;
        });
        if (/^\s*import\b/m.test(src) || /\bimport\s*[(.]/.test(src))
            fail(`${name}: only named imports from relative paths can be bundled`);

        // exports -> plain declarations, their names returned at the end
        const exports = [];
        src = src.replace(/^export\s+(?=(?:async\s+)?(?:function\*?|class|const|let|var)\s)/mg, (m, at)=>
        {
            const decl = /^(?:async\s+)?(function\*?|class|const|let|var)\s+([\w$]+)/.exec(src.slice(at + m.length));
            if (decl[1] == 'function' || decl[1] == 'function*' || decl[1] == 'class')
                exports.push(decl[2]);
            else
                exports.push(...declaredNames(src, at + m.length + decl[0].length - decl[2].length));
            return '';
        });
        if (/^\s*export\b/m.test(src))
            fail(`${name}: only \`export\` in front of a declaration can be bundled`);

        for (const [n, dep] of imports)
            dep.exports.includes(n) || fail(`${name} imports ${n}, which ${dep.name} does not export`);

        const mod = {id: `__mod${order.length}`, name, exports: [...new Set(exports)], src};
        visiting.delete(file);
        mods.set(file, mod);
        order.push(mod);
        return mod;
    };
    visit(entry);
    return order.map(m => `// ---- ${m.name}\nconst ${m.id} = (()=>\n{\n${m.src}\nreturn {${m.exports.join(', ')}};\n})();\n`).join('\n');
}

///////////////////////////////////////////////////////////////////////////////

const page = read(join(HERE, 'index.html'));
const model = fs.readFileSync(MODEL, 'utf8');
const ckpt = JSON.parse(model);
if (!ckpt.params || !ckpt.arch) fail(relative(ROOT, MODEL) + ' is not a checkpoint');

let classic = 0, module = 0;
let html = page
    // the game: every classic <script src> inlined in place, in order
    .replace(/<script src=([^\s>]+)><\/script>/g, (_, src)=>
    {
        ++classic;
        return `<script>\n${inlineSafe(read(join(HERE, src)), src)}\n</script>`;
    })
    // the agent: the checkpoint and the auto switch, then the bundled module
    .replace(/<script type=module src=([^\s>]+)><\/script>/g, (_, src)=>
    {
        ++module;
        return `<script>window.RL_MODEL = ${inlineSafe(model.trim(), 'the model')};\nwindow.RL_AUTO = 1;</script>\n`
            + `<script type=module>\n${inlineSafe(bundle(join(HERE, src)), src)}</script>`;
    });
if (!classic || module != 1)
    fail(`rl/web/index.html: expected game scripts and one module, found ${classic} and ${module}`);
if (/<script[^>]*\ssrc=/.test(html))
    fail('rl/web/index.html has a <script src> in a form this build does not inline');
// NO DOCTYPE, as in index.html: the engine sizes its canvas for quirks mode
if (/<!doctype/i.test(html))
    fail('the page has a doctype: the game canvas would be 0px tall');

html = `<!-- Sunshine Golf Classic played by the RL agent. Single-file build of rl/web/index.html
     by rl/web/build.mjs, model ${relative(ROOT, MODEL).replace(/\\/g, '/')} (iter ${ckpt.meta?.iter ?? '?'}).
     Open it from disk; ?auto=0 hands the game back, ?model=<url> loads another checkpoint. -->\n` + html;

fs.mkdirSync(dirname(OUT), {recursive: true});
fs.writeFileSync(OUT, html);
console.log(`${relative(process.cwd(), OUT)}: ${(html.length/1e6).toFixed(2)} MB (${classic} game scripts, model ${(model.length/1e6).toFixed(2)} MB)`);
