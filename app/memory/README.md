
# Memory

Memory management and policies.

## Responsibilities
- Storing conversational and semantic memories
- Applying memory importance / decay policies
- Exposing memory retrieval APIs to planners

Memory should be treated as an *active system*, not just a database.

## Data Access

Memory store classes currently execute SQL via the shared SQLite connection
provided by `app/storage/database.py` (`Database.conn`).
