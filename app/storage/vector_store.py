import logging
import chromadb
from chromadb.utils import embedding_functions

from app.paths import DATA_DIR, resolve_app_path

logger = logging.getLogger("vector_store")

class VectorStore:
    def __init__(self, path: str = str(DATA_DIR / "vectordb")):
        logger.info("Initializing CPU-based Vector Store...")
        
        # Persistent local storage
        self.client = chromadb.PersistentClient(path=str(resolve_app_path(path)))
        
        # This downloads (once) and runs a tiny, fast embedding model purely on your CPU
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # 1. Semantic Memory Collection (Facts & Rules)
        self.semantic_collection = self.client.get_or_create_collection(
            name="semantic_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} # Best for text similarity
        )

        # 2. Episodic Memory Collection (Conversations)
        self.episodic_collection = self.client.get_or_create_collection(
            name="episodic_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
