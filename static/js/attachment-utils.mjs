export function isImageFile(file) {
    return Boolean(file && typeof file.type === 'string' && file.type.startsWith('image/'));
}

const IMAGE_MIME_ALIASES = new Map([
    ['image/jpg', 'image/jpeg'],
    ['image/pjpeg', 'image/jpeg'],
    ['image/x-png', 'image/png'],
]);

const IMAGE_FILE_EXTENSIONS = new Map([
    ['image/gif', 'gif'],
    ['image/jpeg', 'jpg'],
    ['image/png', 'png'],
    ['image/webp', 'webp'],
]);

const IMAGE_BINARY_SIGNATURES = new Map([
    ['image/gif', 'GIF8'],
    ['image/jpeg', '\xff\xd8\xff'],
    ['image/png', '\x89PNG\r\n\x1a\n'],
]);

function canonicalImageMimeType(value) {
    if (typeof value !== 'string') return '';

    const normalized = value.trim().toLowerCase();
    const canonical = IMAGE_MIME_ALIASES.get(normalized) || normalized;
    return canonical.startsWith('image/') ? canonical : '';
}

function normalizeClipboardImageFile(file, itemMimeType, index) {
    if (!file) return null;

    const fileMimeType = canonicalImageMimeType(file.type);
    const clipboardMimeType = canonicalImageMimeType(itemMimeType);
    const mimeType = fileMimeType || clipboardMimeType;
    if (!mimeType) return null;

    const extension = IMAGE_FILE_EXTENSIONS.get(mimeType) || 'img';
    const name = typeof file.name === 'string' && file.name.trim()
        ? file.name
        : `clipboard-${index + 1}.${extension}`;

    if (file.type === mimeType && file.name === name) {
        return file;
    }

    return new File([file], name, {
        type: mimeType,
        lastModified: Number.isFinite(file.lastModified) ? file.lastModified : Date.now(),
    });
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

    for (const [index, file] of Array.from(dataTransfer.files || []).entries()) {
        const normalizedFile = normalizeClipboardImageFile(file, file?.type, index);
        if (!normalizedFile) continue;
        seen.set(buildFileKey(normalizedFile), normalizedFile);
    }

    for (const [index, item] of Array.from(dataTransfer.items || []).entries()) {
        if (!item || item.kind !== 'file' || !canonicalImageMimeType(item.type)) {
            continue;
        }

        const file = typeof item.getAsFile === 'function' ? item.getAsFile() : null;
        const normalizedFile = normalizeClipboardImageFile(file, item.type, index);
        if (!normalizedFile) continue;
        seen.set(buildFileKey(normalizedFile), normalizedFile);
    }

    return Array.from(seen.values());
}

export function extractBase64Payload(dataUrl) {
    if (typeof dataUrl !== 'string' || !dataUrl) {
        return '';
    }

    return dataUrl.includes(',') ? dataUrl.split(',', 2)[1] : dataUrl;
}

export function repairImageBase64Payload(base64Data, mimeType) {
    if (typeof base64Data !== 'string' || !base64Data) return '';

    const signature = IMAGE_BINARY_SIGNATURES.get(canonicalImageMimeType(mimeType));
    if (!signature) return base64Data;

    try {
        const binaryData = atob(base64Data);
        if (binaryData.startsWith(signature)) return base64Data;

        const maxPrefixBytes = 32;
        const signatureOffset = binaryData.indexOf(signature, 1);
        if (signatureOffset < 1 || signatureOffset > maxPrefixBytes) {
            return base64Data;
        }

        return btoa(binaryData.slice(signatureOffset));
    } catch {
        return base64Data;
    }
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
