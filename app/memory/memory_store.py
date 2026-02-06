from app.storage.database import Database
import re


class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    # --------------------------------------------------
    # Write
    # --------------------------------------------------

    def add(
        self,
        content: str,
        category: str = "general",
        importance: int = 1,
    ) -> None:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory (category, content, importance)
            VALUES (?, ?, ?)
            """,
            (category, content, importance),
        )
        self.db.conn.commit()

    # --------------------------------------------------
    # Read (bulk)
    # --------------------------------------------------

    def get_all(self, limit: int = 20) -> list[str]:
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content
            FROM memory
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [row["content"] for row in cursor.fetchall()]

    # --------------------------------------------------
    # Read (relevance-ranked)
    # --------------------------------------------------

    def get_relevant(self, query: str, limit: int = 5) -> list[str]:
        """
        Return memories ranked by simple lexical relevance + importance.
        """
        query_terms = self._tokenize(query)

        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content, importance
            FROM memory
            """
        )

        scored: list[tuple[float, str]] = []

        for row in cursor.fetchall():
            content = row["content"]
            importance = row["importance"]

            memory_terms = self._tokenize(content)
            overlap = len(query_terms & memory_terms)

            # Require actual lexical overlap OR high importance
            if overlap == 0 and importance < 2:
                continue

            score = overlap + importance * 0.3
            scored.append((score, content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:limit]]

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"\b\w+\b", text.lower()))
