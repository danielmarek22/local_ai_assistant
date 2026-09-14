# Avatar

Model-facing expression and gesture controls plus discovery of avatar assets.

## Responsibilities

- Discover normalized gesture and outfit catalogs from static assets
- Add the allowed expression and gesture contract to the system prompt
- Parse streamed bracket controls against the exact runtime allowlists
- Preserve protected Markdown while filtering invalid control candidates

The package produces semantic expression and animation choices. Browser rendering and
asset playback remain under `static/`, while outfit changes exposed as model tools remain
integration adapters under `app/integrations/`.
