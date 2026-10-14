import logging
import threading
import uuid
from datetime import datetime, timezone

from app.storage.session_lifecycle import require_writable_session
from app.storage.vector_candidates import ranked_candidates

logger = logging.getLogger("scratchpad")
MAX_CONTENT_CHARS = 4000
MAX_QUERY_CHARS = 2000
MAX_RESULTS = 20


class ScratchpadConflict(ValueError):
    """The caller's revision is no longer current."""


class ScratchpadNotFound(ValueError):
    """No entry belongs to this owner under the supplied ID."""


def _bounded_text(value, maximum, label):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must contain 1–{maximum} characters and not be blank")
    return value


def _integer(value, minimum, maximum, label):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}")
    return value


class ScratchpadStore:
    """SQLite is authoritative; vectors contain only derived retrieval candidates.

    The factory creates one store. Its lock spans mutations and vector writes,
    while the shared database lock is released before running embeddings.
    """

    def __init__(self, db, collection, owner_agent_id, *, max_distance=0.70):
        self.db = db
        self.collection = collection
        self.owner_agent_id = _bounded_text(owner_agent_id, 128, "Owner ID")
        self.max_distance = float(max_distance)
        if not 0 <= self.max_distance <= 2:
            raise ValueError("max_distance must be between 0 and 2")
        self._lock = threading.RLock()

    def _read(self, conn, entry_id):
        row = conn.execute(
            "SELECT id, content, created_at, updated_at, revision FROM scratchpad "
            "WHERE id = ? AND owner_agent_id = ?",
            (entry_id, self.owner_agent_id),
        ).fetchone()
        if row is None:
            raise ScratchpadNotFound("Note not found")
        return dict(row)

    def read(self, entry_id):
        _bounded_text(entry_id, 128, "Note ID")
        with self.db.connection() as conn:
            return self._read(conn, entry_id)

    def recent(self, limit=10, offset=0):
        _integer(limit, 1, MAX_RESULTS, "limit")
        _integer(offset, 0, 100000, "offset")
        with self.db.connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM scratchpad WHERE owner_agent_id = ?",
                (self.owner_agent_id,),
            ).fetchone()[0]
            rows = conn.execute(
                "SELECT id, content, created_at, updated_at, revision FROM scratchpad "
                "WHERE owner_agent_id = ? ORDER BY updated_at DESC, id ASC LIMIT ? OFFSET ?",
                (self.owner_agent_id, limit, offset),
            ).fetchall()
        return {"entries": [dict(row) for row in rows], "total": total,
                "limit": limit, "offset": offset}

    def _index(self, entry):
        self.collection.upsert(
            ids=[entry["id"]], documents=[entry["content"]],
            metadatas=[{"owner_agent_id": self.owner_agent_id, "revision": entry["revision"]}],
        )

    def _sync_result(self, entry, *, deleted=False):
        indexed = True
        try:
            if deleted:
                self.collection.delete(ids=[entry["id"]])
            else:
                self._index(entry)
        except Exception:
            indexed = False
            logger.exception("Note %s committed but vector synchronization failed", entry["id"])
        result = {"deleted" if deleted else "saved": True, "indexed": indexed,
                  "entry": entry}
        if not indexed:
            result["warning"] = (
                "The canonical change is saved. Do not repeat the mutation. "
                "Semantic indexing needs offline reconcile-indexes maintenance."
            )
        return result

    def write(self, content, session_id):
        _bounded_text(content, MAX_CONTENT_CHARS, "Content")
        now = datetime.now(timezone.utc).isoformat()
        entry = {"id": str(uuid.uuid4()), "content": content, "created_at": now,
                 "updated_at": now, "revision": 1}
        with self._lock:
            with self.db.transaction() as conn:
                require_writable_session(conn, session_id)
                conn.execute(
                    "INSERT INTO scratchpad (id, owner_agent_id, content, created_at, updated_at, revision) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (entry["id"], self.owner_agent_id, content, now, now, 1),
                )
            return self._sync_result(entry)

    def _mutate(self, entry_id, expected_revision, session_id, content=None):
        _bounded_text(entry_id, 128, "Note ID")
        _integer(expected_revision, 1, 2**63 - 2, "expected_revision")
        with self._lock:
            with self.db.transaction() as conn:
                require_writable_session(conn, session_id)
                entry = self._read(conn, entry_id)
                if entry["revision"] != expected_revision:
                    raise ScratchpadConflict("Note changed; read it again before editing or deleting")
                if content is None:
                    conn.execute("DELETE FROM scratchpad WHERE id = ? AND owner_agent_id = ?",
                                 (entry_id, self.owner_agent_id))
                else:
                    entry.update(content=content, revision=expected_revision + 1,
                                 updated_at=datetime.now(timezone.utc).isoformat())
                    conn.execute(
                        "UPDATE scratchpad SET content = ?, revision = ?, updated_at = ? "
                        "WHERE id = ? AND owner_agent_id = ?",
                        (content, entry["revision"], entry["updated_at"], entry_id, self.owner_agent_id),
                    )
            # Deletion results need identity/version, not the deleted text.
            if content is None:
                entry = {"id": entry_id, "revision": expected_revision}
            return self._sync_result(entry, deleted=content is None)

    def edit(self, entry_id, content, expected_revision, session_id):
        _bounded_text(content, MAX_CONTENT_CHARS, "Content")
        return self._mutate(entry_id, expected_revision, session_id, content)

    def delete(self, entry_id, expected_revision, session_id):
        return self._mutate(entry_id, expected_revision, session_id)

    def search(self, query, limit=5):
        _bounded_text(query, MAX_QUERY_CHARS, "Query")
        _integer(limit, 1, MAX_RESULTS, "limit")
        with self._lock:
            count = self.collection.count()
            if not count:
                return []
            candidate_limit = min(count, limit * 4)
            result = self.collection.query(
                query_texts=[query], n_results=candidate_limit,
                where={"owner_agent_id": self.owner_agent_id},
                include=["distances", "metadatas"],
            )
            ids = result.get("ids") or []
            metadatas = result.get("metadatas") or []
            if not ids or not metadatas or not isinstance(ids[0], list) or not isinstance(metadatas[0], list):
                return []
            metadata_by_id = {
                record_id: metadata for record_id, metadata in zip(ids[0], metadatas[0])
                if isinstance(record_id, str) and isinstance(metadata, dict)
            }
            matches = []
            with self.db.connection() as conn:
                for entry_id, distance in ranked_candidates(result, candidate_limit):
                    metadata = metadata_by_id.get(entry_id, {})
                    revision = metadata.get("revision")
                    if (not 0 <= distance <= self.max_distance
                            or metadata.get("owner_agent_id") != self.owner_agent_id
                            or isinstance(revision, bool) or not isinstance(revision, int)):
                        continue
                    try:
                        entry = self._read(conn, entry_id)
                    except ScratchpadNotFound:
                        continue
                    if entry["revision"] != revision:
                        continue
                    matches.append({**entry, "distance": distance})
                    if len(matches) == limit:
                        break
            return matches

    def reconcile_index(self, batch_size=100):
        """Offline repair of this owner's index; preserve other owners' vectors."""
        _integer(batch_size, 1, 1000, "batch_size")
        with self._lock:
            with self.db.connection() as conn:
                rows = conn.execute(
                    "SELECT id, content, revision FROM scratchpad WHERE owner_agent_id = ?",
                    (self.owner_agent_id,),
                ).fetchall()
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                self.collection.upsert(
                    ids=[row["id"] for row in batch], documents=[row["content"] for row in batch],
                    metadatas=[{"owner_agent_id": self.owner_agent_id, "revision": row["revision"]}
                               for row in batch],
                )
            canonical = {row["id"] for row in rows}
            indexed = set(self.collection.get(where={"owner_agent_id": self.owner_agent_id}, include=[])["ids"])
            orphans = sorted(indexed - canonical)
            if orphans:
                self.collection.delete(ids=orphans)
            return {"canonical_count": len(rows), "upserted_count": len(rows),
                    "deleted_orphan_count": len(orphans)}
