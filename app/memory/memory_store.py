from app.storage.database import Database
import re
from collections import Counter

class MemoryStore:
    def __init__(self, db: Database):
        self.db = db

    def add(self, content: str, category: str = "general", importance: int = 1):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory (content, category, importance)
            VALUES (?, ?, ?)
            """,
            (content, category, importance)
        )
        self.db.conn.commit()

    def get_all(self, limit: int = 20):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content
            FROM memory
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
            """,
            (limit,)
        )
        return [row["content"] for row in cursor.fetchall()]
    
    def get_relevant(self, query: str, limit: int = 5):
        """
        Returns memories ranked by lexical relevance to the query.
        """
        query_terms = self._tokenize(query)

        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT content, importance
            FROM memory
            """
        )

        scored = []

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

        scored.sort(reverse=True, key=lambda x: x[0])
        return [content for _, content in scored[:limit]]

    def _tokenize(self, text: str) -> set[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(tokens)
