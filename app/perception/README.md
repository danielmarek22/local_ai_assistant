
# Perception

Input interpretation and preprocessing.

## Responsibilities
- Parsing and normalizing user input
- Multimodal input handling for text plus attachments
- Defining shared attachment models and constructors
- Preparing structured representations for planners

## Current Notes

- `attachments.py` contains the shared `Attachment` base type plus the current `ImageAttachment` implementation.
- The current transport and prompt pipeline supports images, but the attachment model now has a cleaner seam for future types such as PDFs.
- `state.py` is focused on `PerceptionState` and perception entries rather than attachment parsing details.

This layer translates *raw input* into *actionable signals*.
