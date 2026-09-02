---
title: Mindcraft integration
description: Run the Astra-specific Mindcraft fork and connect it to Astra's integration runtime.
---

# Mindcraft integration

The project uses the [Astra Mindcraft fork](https://github.com/danielmarek22/local-assistant-mindcraft-integration), not an unmodified upstream checkout. The fork exposes a typed external-control protocol while preserving Mindcraft navigation and safety behavior.

## Choose a controller mode

| Mode | Use it when | Behavior |
| --- | --- | --- |
| `external` | Astra should be the only planner | Mindcraft accepts typed actions and emits events without initializing its own chat, embedding, or code model |
| `hybrid` | Mindcraft should also interpret higher-level requests | Native Mindcraft model behavior remains available alongside typed external actions |

Start with `external` when testing the integration boundary. Move to `hybrid` only when you explicitly want two planning layers.

## 1. Install the fork

In a directory next to Astra:

```bash
git clone https://github.com/danielmarek22/local-assistant-mindcraft-integration.git
cd local-assistant-mindcraft-integration
npm install
```

Use Node.js 18 or 20. The bot targets Minecraft Java Edition.

## 2. Configure Mindcraft

Edit `settings.js` in the fork. The important Astra-facing values are:

```javascript
export const settings = {
  host: "127.0.0.1",
  port: 55916,
  auth: "offline",
  mindserver_port: 8081,
  profiles: ["./andy.json"],
  base_profile: "assistant",
  controller_mode: "external",
  allow_insecure_coding: false,
};
```

- `port` must match the port shown when the Minecraft world is opened to LAN.
- `mindserver_port` must match Astra's integration URL.
- `profiles` selects the bot profile.
- Keep `allow_insecure_coding: false` unless you are deliberately testing generated code in an isolated environment.

The current project profile is `andy.json`:

```json
{
  "name": "Astra",
  "model": "ollama/gemma4:e4b-it-qat",
  "embedding": "ollama/embeddinggemma"
}
```

The model fields matter in `hybrid` mode. In `external` mode, the typed protocol does not need Mindcraft's chat model.

## 3. Prepare Minecraft

1. Start Minecraft Java Edition and enter the target world.
2. Open the world to LAN.
3. Copy the displayed LAN port into `settings.js`.
4. Keep the world open while Mindcraft runs.

For a persistent server, replace the LAN host and port with that server's connection details and select the correct authentication mode.

## 4. Start and verify Mindcraft

From the fork directory:

```bash
npm start
```

Confirm the bot joins the world and the integration server listens on port `8081`. The fork's protocol tests can be run independently with:

```bash
npm test
```

## 5. Enable the integration in Astra

In `app/config/assistant.yaml`:

```yaml
integrations:
  mindcraft:
    enabled: true
    url: http://localhost:8081
    agent_name: Astra
    connect_timeout: 3.0
    reconnect_delay_s: 2.0
    reconnect_max_delay_s: 30.0
    context_enabled: true
    recent_output_limit: 3
    events_enabled: true
    ambient_session_id: ""
    autonomous_events: [critical_health, died, disconnected]
    attachment_dir: static/uploads/events/mindcraft
```

Then start or restart Astra. It connects in the background and retries with a bounded delay if Mindcraft is temporarily unavailable.

## 6. End-to-end smoke test

1. Confirm the Minecraft bot is visible in the world.
2. Start Astra and check that the Mindcraft integration connects.
3. Ask Astra for a small, observable action such as looking at a player or moving a short distance.
4. Confirm the action reaches a terminal state: completed, failed, cancelled, timed out, or unavailable.
5. Trigger a harmless world event and confirm it appears in Astra's integration context when events are enabled.

The fork currently supports typed operations for movement, following, collecting, chopping, looking, speaking, stopping, and capturing the bot view. Captures arrive as JPEG attachments. Autonomous reactions still obey Astra's `autonomy` limits and are disabled globally unless explicitly enabled.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Bot cannot join | Minecraft Java world is open, LAN/server port matches, and authentication mode is correct |
| Astra cannot connect | Mindcraft is listening on `mindserver_port` and Astra uses the same URL |
| Astra connects to the wrong bot | `agent_name` matches the profile's `name` exactly |
| Commands are interpreted twice | Prefer `external` mode when Astra is the sole planner |
| Hybrid mode has no model response | Ollama is running and the profile model names exist locally |
| Captures fail | The bot view is available and Astra can write to `attachment_dir` |
| Events arrive but Astra does nothing | `events_enabled` is true; autonomous action also requires `autonomy.enabled: true` and an allowed event |

For protocol details, see `docs/astra-integration.md` inside the fork. For the original project and general Mindcraft behavior, consult [upstream Mindcraft](https://github.com/mindcraft-bots/mindcraft).
