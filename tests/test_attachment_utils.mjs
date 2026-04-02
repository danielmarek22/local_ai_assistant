import test from 'node:test';
import assert from 'node:assert/strict';

import {
    extractBase64Payload,
    extractImageFilesFromDataTransfer,
    insertTextAtCursor,
    isImageFile,
} from '../static/js/attachment-utils.mjs';

test('extractBase64Payload strips the data URL prefix', () => {
    assert.equal(
        extractBase64Payload('data:image/png;base64,aGVsbG8='),
        'aGVsbG8=',
    );
    assert.equal(extractBase64Payload('plain-base64'), 'plain-base64');
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
