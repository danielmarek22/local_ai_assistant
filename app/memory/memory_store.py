import uuid
from app.storage.database import Database
from app.storage.vector_store import VectorStore

class MemoryStore:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.collection = self.vector_store.semantic_collection

    def add(self, content: str, category: str = "general", importance: int = 1) -> None:
        mem_id = str(uuid.uuid4())
        
        # 1. Save to SQLite (for easy viewing/backups in the frontend)
        cursor = self.db.conn.cursor()
        cursor.execute(
            "INSERT INTO memory (category, content, importance) VALUES (?, ?, ?)",
            (category, content, importance),
        )
        self.db.conn.commit()

        # 2. Save to Vector Store (for AI semantic retrieval)
        self.collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[{"category": category, "importance": importance}]
        )

    def get_all(self, limit: int = 20) -> list[str]:
        """
        Retrieves recent memories for the UI/frontend, sorted by importance and recency.
        """
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

    def get_relevant(self, query: str, limit: int = 3) -> list[str]:
        """
        True semantic search using CPU embeddings for the Orchestrator.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # Chroma returns a list of lists for documents
        if not results["documents"] or not results["documents"][0]:
            return []
            
        return results["documents"][0]