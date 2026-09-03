---
title: PocketTTS
description: Prepare a reusable PocketTTS voice state and use it for Astra's spoken output.
---

# PocketTTS

Astra loads PocketTTS in-process at startup. It loads the base model once, restores a baked voice state, and writes generated speech to WAV files for the browser. Baking the voice once avoids reprocessing the reference recording on every launch.

## Current Astra contract

- Engine aliases: `pocket-tts`, `pocket_tts`, or `pocket`.
- Voice state expected by the current wrapper: `app/tts/escoffier.safetensors`.
- Runtime implementation: `app/tts/pocket_tts.py`.
- Engine selection: `app/tts/factory.py`.

## 1. Prepare reference audio

Choose a clean WAV recording containing only the target speaker. Avoid background music, overlapping voices, clipping, and long silence. A representative speaking style matters more than a very long sample.

!!! warning "Use audio you have permission to use"
    A baked state reproduces characteristics of the reference speaker. Keep both the recording and resulting state private unless you have permission to distribute them.

## 2. Bake the voice

From the Astra repository root, run:

```bash
venv_app/bin/python scripts/bake_pocket_voice.py /absolute/path/to/reference.wav
```

The default output is the file currently expected by Astra:

```text
app/tts/escoffier.safetensors
```

Use a different destination when preparing or comparing a voice without replacing the active one:

```bash
venv_app/bin/python scripts/bake_pocket_voice.py \
  /absolute/path/to/reference.wav \
  --output /tmp/candidate-voice.safetensors
```

The helper refuses to overwrite an existing state. Add `--force` only after confirming the destination.

## 3. Select PocketTTS

In `app/config/assistant.yaml`:

```yaml
tts:
  engine: pocket-tts
```

Restart Astra after changing the voice state or engine. The model and voice are loaded during server startup.

## 4. Verify

1. Watch startup logs for the PocketTTS model and voice-state messages.
2. Send a short message in the browser.
3. Confirm Astra returns text and the browser plays the generated speech.
4. Inspect `static/audio/` if text succeeds but no audio is heard. Generated UUID-named
   `.wav` files are cleared whenever Astra starts; reference files with other names are
   preserved.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError: pocket_tts` | Install the project's runtime dependencies in `venv_app` |
| Voice-state file is missing | Bake it to `app/tts/escoffier.safetensors` or restore the existing private file |
| Baking refuses to continue | Confirm the input is a `.wav`; use `--force` only for an intentional replacement |
| Startup is slow | The base model loads once at process startup; later utterances reuse it |
| Speech sounds unstable | Use a cleaner, single-speaker reference with natural pacing |
| Text appears but audio does not | Check synthesis errors, file creation in `static/audio/`, and browser audio permissions |

The helper follows PocketTTS's official Python workflow: create a state from an audio prompt, export it, and later reload the `.safetensors` state. See the [PocketTTS README](https://github.com/kyutai-labs/pocket-tts/blob/main/README.md) and [Python API reference](https://github.com/kyutai-labs/pocket-tts/blob/main/docs/API%20Reference/python-api.md).
