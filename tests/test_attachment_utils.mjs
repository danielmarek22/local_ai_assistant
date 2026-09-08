import test from 'node:test';
import assert from 'node:assert/strict';

import {
    extractBase64Payload,
    extractImageFilesFromDataTransfer,
    insertTextAtCursor,
    isImageFile,
    repairImageBase64Payload,
} from '../static/js/attachment-utils.mjs';

test('extractBase64Payload strips the data URL prefix', () => {
    assert.equal(
        extractBase64Payload('data:image/png;base64,aGVsbG8='),
        'aGVsbG8=',
    );
    assert.equal(extractBase64Payload('plain-base64'), 'plain-base64');
});

test('repairImageBase64Payload strips a short clipboard prefix before a PNG', () => {
    const pngBytes = '\x89PNG\r\n\x1a\nimage-data';
    const prefixed = btoa(`clipboard-prefix${pngBytes}`);

    assert.equal(
        repairImageBase64Payload(prefixed, 'image/png'),
        btoa(pngBytes),
    );
});

test('repairImageBase64Payload preserves valid and unrecognized image payloads', () => {
    const validPng = btoa('\x89PNG\r\n\x1a\nimage-data');
    const invalidPng = btoa('not-a-png');

    assert.equal(repairImageBase64Payload(validPng, 'image/png'), validPng);
    assert.equal(repairImageBase64Payload(invalidPng, 'image/png'), invalidPng);
    assert.equal(repairImageBase64Payload(invalidPng, 'image/webp'), invalidPng);
});

test('extractImageFilesFromDataTransfer collects unique image files from files and items', () => {
    const pngFile = {
        name: 'paste.png',
        type: 'image/png',
        size: 5,
        lastModified: 1,
    };
    const txtFile = {
        name: 'notes.txt',
        type: 'text/plain',
        size: 3,
        lastModified: 2,
    };

    const files = extractImageFilesFromDataTransfer({
        files: [pngFile, txtFile],
        items: [
            { kind: 'file', type: 'image/png', getAsFile: () => pngFile },
            { kind: 'string', type: 'text/plain', getAsFile: () => null },
        ],
    });

    assert.deepEqual(files, [pngFile]);
});

test('clipboard item MIME recovers an unnamed generically typed image file', () => {
    const clipboardBlob = new File(['image-data'], '', {
        type: 'application/octet-stream',
        lastModified: 7,
    });

    const files = extractImageFilesFromDataTransfer({
        items: [
            { kind: 'file', type: 'image/png', getAsFile: () => clipboardBlob },
        ],
    });

    assert.equal(files.length, 1);
    assert.equal(files[0].name, 'clipboard-1.png');
    assert.equal(files[0].type, 'image/png');
    assert.equal(files[0].size, clipboardBlob.size);
});

test('clipboard image MIME aliases are canonicalized for the attachment contract', () => {
    const clipboardBlob = new File(['image-data'], 'clipboard', {
        type: 'application/octet-stream',
    });

    const files = extractImageFilesFromDataTransfer({
        items: [
            { kind: 'file', type: 'image/x-png', getAsFile: () => clipboardBlob },
        ],
    });

    assert.equal(files.length, 1);
    assert.equal(files[0].type, 'image/png');
});

test('non-image clipboard content is left for the browser to paste normally', () => {
    const files = extractImageFilesFromDataTransfer({
        files: [],
        items: [
            { kind: 'string', type: 'text/plain', getAsFile: () => null },
        ],
    });

    assert.deepEqual(files, []);
});

test('insertTextAtCursor inserts text at the current selection', () => {
    const textarea = {
        value: 'hello world',
        selectionStart: 6,
        selectionEnd: 11,
        setSelectionRange(start, end) {
            this.selectionStart = start;
            this.selectionEnd = end;
        },
    };

    insertTextAtCursor(textarea, 'there');

    assert.equal(textarea.value, 'hello there');
    assert.equal(textarea.selectionStart, 11);
    assert.equal(textarea.selectionEnd, 11);
});

test('isImageFile matches image mime types only', () => {
    assert.equal(isImageFile({ type: 'image/jpeg' }), true);
    assert.equal(isImageFile({ type: 'text/plain' }), false);
    assert.equal(isImageFile(null), false);
});
