export function isImageFile(file) {
    return Boolean(file && typeof file.type === 'string' && file.type.startsWith('image/'));
}

function buildFileKey(file) {
    return [
        typeof file.name === 'string' ? file.name : '',
        typeof file.type === 'string' ? file.type : '',
        Number.isFinite(file.size) ? file.size : '',
        Number.isFinite(file.lastModified) ? file.lastModified : '',
    ].join(':');
}

export function extractImageFilesFromDataTransfer(dataTransfer) {
    if (!dataTransfer) {
        return [];
    }

    const seen = new Map();

    for (const file of Array.from(dataTransfer.files || [])) {
        if (!isImageFile(file)) continue;
        seen.set(buildFileKey(file), file);
    }

    for (const item of Array.from(dataTransfer.items || [])) {
        if (!item || item.kind !== 'file' || typeof item.type !== 'string' || !item.type.startsWith('image/')) {
            continue;
        }

        const file = typeof item.getAsFile === 'function' ? item.getAsFile() : null;
        if (!isImageFile(file)) continue;
        seen.set(buildFileKey(file), file);
    }

    return Array.from(seen.values());
}

export function extractBase64Payload(dataUrl) {
    if (typeof dataUrl !== 'string' || !dataUrl) {
        return '';
    }

    return dataUrl.includes(',') ? dataUrl.split(',', 2)[1] : dataUrl;
}

export function insertTextAtCursor(textarea, text) {
    if (!textarea || typeof textarea.value !== 'string' || !text) {
        return;
    }

    const start = Number.isInteger(textarea.selectionStart)
        ? textarea.selectionStart
        : textarea.value.length;
    const end = Number.isInteger(textarea.selectionEnd)
        ? textarea.selectionEnd
        : start;

    textarea.value = textarea.value.slice(0, start) + text + textarea.value.slice(end);

    const cursor = start + text.length;
    if (typeof textarea.setSelectionRange === 'function') {
        textarea.setSelectionRange(cursor, cursor);
    }
}
