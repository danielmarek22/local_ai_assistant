import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const staticRoot = path.join(repositoryRoot, 'static');

function importedSpecifiers(source) {
    const moduleSource = source
        .replace(/\/\*[\s\S]*?\*\//g, '')
        .replace(/^\s*\/\/.*$/gm, '');
    const specifiers = [];
    const patterns = [
        /\bimport\s*(?:[^"'();]*?\bfrom\s*)?["']([^"']+)["']/g,
        /\bimport\s*\(\s*["']([^"']+)["']/g,
        /\bexport\s+[^"'();]*?\bfrom\s*["']([^"']+)["']/g,
    ];

    for (const pattern of patterns) {
        for (const match of moduleSource.matchAll(pattern)) {
            specifiers.push(match[1]);
        }
    }
    return specifiers;
}

function resolveSpecifier(specifier, importerPath, imports) {
    assert.doesNotMatch(specifier, /^https?:\/\//, `remote browser import: ${specifier}`);

    if (specifier.startsWith('/static/')) {
        return path.join(staticRoot, specifier.slice('/static/'.length));
    }
    if (specifier.startsWith('./') || specifier.startsWith('../')) {
        return path.resolve(path.dirname(importerPath), specifier);
    }

    const importMapKey = Object.keys(imports)
        .filter((key) => key === specifier || (key.endsWith('/') && specifier.startsWith(key)))
        .sort((left, right) => right.length - left.length)[0];
    assert.ok(importMapKey, `unmapped browser import: ${specifier}`);

    const mapped = imports[importMapKey] + specifier.slice(importMapKey.length);
    assert.match(mapped, /^\/static\//, `non-local import-map target: ${mapped}`);
    return path.join(staticRoot, mapped.slice('/static/'.length));
}

test('browser runtime import graph is pinned, local, and complete', async () => {
    const indexPath = path.join(staticRoot, 'index.html');
    const indexSource = await readFile(indexPath, 'utf8');
    const importMapSource = indexSource.match(
        /<script\s+type=["']importmap["']>([\s\S]*?)<\/script>/,
    )?.[1];
    assert.ok(importMapSource, 'index.html must declare an import map');

    const { imports } = JSON.parse(importMapSource);
    assert.deepEqual(imports, {
        three: '/static/vendor/three/0.169.0/build/three.module.min.js',
        'three/addons/': '/static/vendor/three/0.169.0/examples/jsm/',
        '@pixiv/three-vrm': '/static/vendor/three-vrm/3.0.0/lib/three-vrm.module.min.js',
    });

    const entrySpecifier = indexSource.match(
        /<script\s+type=["']module["']\s+src=["']([^"']+)["']><\/script>/,
    )?.[1];
    assert.equal(entrySpecifier, '/static/main.js');

    const pending = [resolveSpecifier(entrySpecifier, indexPath, imports)];
    const visited = new Set();
    while (pending.length > 0) {
        const modulePath = pending.pop();
        if (visited.has(modulePath)) continue;

        assert.ok(modulePath.startsWith(`${staticRoot}${path.sep}`), `import escapes static root: ${modulePath}`);
        assert.ok((await stat(modulePath)).size > 0, `empty browser module: ${modulePath}`);
        visited.add(modulePath);

        const source = await readFile(modulePath, 'utf8');
        for (const specifier of importedSpecifiers(source)) {
            pending.push(resolveSpecifier(specifier, modulePath, imports));
        }
    }

    assert.ok(visited.size >= 20, `unexpectedly small browser import graph: ${visited.size} modules`);

    const licensePaths = [
        'vendor/three/0.169.0/LICENSE',
        'vendor/three-vrm/3.0.0/LICENSE',
        'vendor/marked/13.0.2/LICENSE.md',
        'vendor/dompurify/3.1.6/LICENSE',
    ];
    for (const licensePath of licensePaths) {
        assert.ok(
            (await stat(path.join(staticRoot, licensePath))).size > 0,
            `missing vendor license: ${licensePath}`,
        );
    }
});
