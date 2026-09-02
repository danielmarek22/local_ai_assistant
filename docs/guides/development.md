---
title: Working on Astra
description: Development practices for extending Astra without eroding its boundaries.
---

# Working on Astra

## Prefer a narrow vertical change

Astra has many interacting subsystems. A strong change usually includes:

1. a clearly stated invariant or failure case;
2. the smallest responsible implementation boundary;
3. a regression test close to that boundary;
4. integration coverage when transport or persistence behavior changes;
5. documentation updates when architecture or user-visible behavior changes.

## Preserve authority boundaries

Treat browser payloads, model output, integration data, retrieved memory, and participant content as untrusted at their entry points. Identity, session ownership, source messages, allowed capabilities, and approval decisions belong to application code.

## Keep local inference bounded

- Do not resend historical binary media when text representation is sufficient.
- Bound injected integration and memory context.
- Keep tool loops and autonomous event chains finite.
- Avoid overlapping local model calls.
- Prefer cheap event detection before expensive model inspection.

## Tests

Python tests use `unittest`, in-memory SQLite, and lightweight fakes for external systems. Browser modules use Node's built-in test runner. A bug fix should start with or include a regression test that would have failed before the change.

Run the full Python suite:

```bash
venv_app/bin/python -m unittest discover -s tests -v
```

## Documentation expectations

Update the generated documentation when a change affects system boundaries, data flow, configuration meaning, or a stable public contract. Keep the explanation close to the relevant architecture or guide page instead of maintaining a separate decision log.
