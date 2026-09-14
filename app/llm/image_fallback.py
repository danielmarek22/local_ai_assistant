from __future__ import annotations

import json

import requests


def resolve_request_retries(
    messages,
    max_retries_override: int | None,
) -> int | None:
    if max_retries_override is not None:
        return max_retries_override
    if messages_include_images(messages):
        return 0
    return None


def should_retry_without_images(
    exc: requests.HTTPError,
    messages,
    *,
    multimodal_supported: bool,
) -> bool:
    if not multimodal_supported:
        return False

    _, has_images = strip_images_from_messages(messages)
    if not has_images:
        return False

    status_code = getattr(exc.response, "status_code", None)
    if status_code not in {400, 415, 422}:
        return False

    error_text = http_error_text(exc)
    if not error_text:
        return True

    image_error_markers = (
        "image",
        "vision",
        "multimodal",
        "unsupported",
        "not support",
        "base64",
    )
    return any(marker in error_text for marker in image_error_markers)


def error_indicates_model_without_images(error_text: str) -> bool:
    if not error_text:
        return False

    capability_markers = (
        "does not support image",
        "doesn't support image",
        "vision is not supported",
        "multimodal is not supported",
        "model does not support vision",
    )
    return any(marker in error_text for marker in capability_markers)


def http_error_text(exc: requests.HTTPError) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return ""

    try:
        text = response.text
    except Exception:
        text = ""

    if not text:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            text = str(payload.get("error") or payload.get("message") or "")

    return str(text).strip().lower()


def strip_images_from_messages(messages) -> tuple[list, bool]:
    stripped = []
    removed_images = False

    for message in messages:
        if not isinstance(message, dict) or "images" not in message:
            stripped.append(message)
            continue

        message_without_images = dict(message)
        message_without_images.pop("images", None)
        stripped.append(message_without_images)
        removed_images = True

    return stripped, removed_images


def messages_include_images(messages) -> bool:
    return any(
        isinstance(message, dict) and bool(_message_images(message))
        for message in messages
    )


def build_fallback_messages(messages) -> list[tuple[list, str, int]]:
    candidates: list[tuple[list, str, int]] = []
    seen_keys: set[str] = set()

    image_message_indices = _image_message_indices(messages)
    if not image_message_indices:
        return candidates

    current_image_message_index = _current_user_message_index_with_images(messages)
    ordered_indices = []
    if current_image_message_index is not None:
        ordered_indices.append(current_image_message_index)
    ordered_indices.extend(
        index
        for index in reversed(image_message_indices)
        if index != current_image_message_index
    )

    for message_index in ordered_indices:
        image_count = _image_count_for_message(messages, message_index)
        if image_count > 1:
            for image_index in range(image_count):
                candidate, removed = _strip_message_images(
                    messages,
                    message_index=message_index,
                    image_indexes={image_index},
                )
                if removed:
                    _add_fallback_candidate(
                        candidates,
                        seen_keys,
                        candidate,
                        strategy=(
                            f"without image {image_index + 1} "
                            f"from message {message_index + 1}"
                        ),
                        dropped_current_images_count=(
                            1 if message_index == current_image_message_index else 0
                        ),
                    )

        candidate, removed = _strip_message_images(
            messages,
            message_index=message_index,
            image_indexes=None,
        )
        if removed:
            label = (
                "without current message images"
                if message_index == current_image_message_index
                else f"without images from message {message_index + 1}"
            )
            _add_fallback_candidate(
                candidates,
                seen_keys,
                candidate,
                strategy=label,
                dropped_current_images_count=(
                    image_count if message_index == current_image_message_index else 0
                ),
            )

    candidate, removed = strip_images_from_messages(messages)
    if removed:
        current_images_total = (
            _image_count_for_message(messages, current_image_message_index)
            if current_image_message_index is not None
            else 0
        )
        _add_fallback_candidate(
            candidates,
            seen_keys,
            candidate,
            strategy="without all images",
            dropped_current_images_count=current_images_total,
        )

    return candidates


def _message_images(message: dict) -> list:
    images = message.get("images")
    return images if isinstance(images, list) else []


def _image_message_indices(messages) -> list[int]:
    return [
        index
        for index, message in enumerate(messages)
        if isinstance(message, dict) and _message_images(message)
    ]


def _current_user_message_index_with_images(messages) -> int | None:
    if not messages:
        return None

    index = len(messages) - 1
    message = messages[index]
    if (
        isinstance(message, dict)
        and message.get("role") == "user"
        and _message_images(message)
    ):
        return index
    return None


def _image_count_for_message(messages, message_index: int | None) -> int:
    if message_index is None or message_index < 0 or message_index >= len(messages):
        return 0

    message = messages[message_index]
    images = _message_images(message) if isinstance(message, dict) else []
    return len(images)


def _strip_message_images(
    messages,
    message_index: int,
    image_indexes: set[int] | None,
) -> tuple[list, bool]:
    stripped = []
    removed_images = False

    for index, message in enumerate(messages):
        if index != message_index or not isinstance(message, dict):
            stripped.append(message)
            continue

        images = _message_images(message)
        if not images:
            stripped.append(message)
            continue

        updated_message = dict(message)
        if image_indexes is None:
            updated_message.pop("images", None)
            removed_images = True
        else:
            remaining_images = [
                image
                for image_index, image in enumerate(images)
                if image_index not in image_indexes
            ]
            if len(remaining_images) != len(images):
                removed_images = True
            if remaining_images:
                updated_message["images"] = remaining_images
            else:
                updated_message.pop("images", None)

        stripped.append(updated_message)

    return stripped, removed_images


def _add_fallback_candidate(
    candidates: list[tuple[list, str, int]],
    seen_keys: set[str],
    candidate_messages: list,
    strategy: str,
    dropped_current_images_count: int,
) -> None:
    key = json.dumps(candidate_messages, sort_keys=True)
    if key in seen_keys:
        return
    seen_keys.add(key)
    candidates.append((candidate_messages, strategy, dropped_current_images_count))
