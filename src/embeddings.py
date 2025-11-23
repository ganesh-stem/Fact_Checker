"""
Embedding & Retrieval System Module

Handles text embedding using sentence-transformers and
vector storage/retrieval using FAISS.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None


@dataclass
class RetrievalResult:
    """Result from vector similarity search."""
    fact: str
    score: float
    metadata: Dict
    rank: int


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model.

    Provides methods to embed text and compute similarities.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(embedding1, embedding2) / (norm1 * norm2)


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search.

    Supports:
    - Adding vectors with metadata
    - Similarity search with top-k retrieval
    - Persistence (save/load)
    """

    def __init__(self, embedding_dim: int, index_type: str = "flat"):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        if faiss is None:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

        self.embedding_dim = embedding_dim
        self.index_type = index_type

        # Create FAISS index
        if index_type == "flat":
            # Exact search (best for small datasets)
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim for normalized vectors)
        elif index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 10)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World graph
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Store metadata separately (FAISS only stores vectors)
        self.metadata: List[Dict] = []
        self.texts: List[str] = []

    def add(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """
        Add embeddings to the index.

        Args:
            embeddings: Numpy array of embeddings
            texts: Original texts
            metadata: Optional metadata for each embedding
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Train index if needed (for IVF)
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store texts and metadata
        self.texts.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """
        Search for most similar vectors.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append(RetrievalResult(
                    fact=self.texts[idx],
                    score=float(score),
                    metadata=self.metadata[idx],
                    rank=rank + 1
                ))

        return results

    def save(self, path: str):
        """Save index and metadata to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")

        # Save metadata and texts
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)

    def load(self, path: str):
        """Load index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")

        # Load metadata
        with open(f"{path}.meta", "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            self.embedding_dim = data["embedding_dim"]
            self.index_type = data["index_type"]

    def __len__(self):
        return self.index.ntotal


class ChromaVectorStore:
    """
    Alternative vector store using ChromaDB.

    Provides a simpler API with built-in metadata support.
    """

    def __init__(self, collection_name: str = "facts", persist_directory: str = None):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence (None for in-memory)
        """
        if chromadb is None:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict] = None):
        """Add embeddings to the collection."""
        ids = [f"fact_{i + self.collection.count()}" for i in range(len(texts))]

        if metadata is None:
            metadata = [{"source": "unknown"}] * len(texts)

        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
            ids=ids
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Search for most similar vectors."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieval_results = []
        for i, (doc, distance, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )):
            # ChromaDB returns distance, convert to similarity
            similarity = 1 - distance

            retrieval_results.append(RetrievalResult(
                fact=doc,
                score=similarity,
                metadata=meta,
                rank=i + 1
            ))

        return retrieval_results

    def __len__(self):
        return self.collection.count()


class RetrievalSystem:
    """
    High-level retrieval system combining embedding model and vector store.

    Provides a simple interface for fact retrieval.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "faiss",
        persist_path: str = None
    ):
        """
        Initialize the retrieval system.

        Args:
            embedding_model: Name of the embedding model
            vector_store_type: Type of vector store ('faiss' or 'chroma')
            persist_path: Path for persistence
        """
        self.embedding_model = EmbeddingModel(embedding_model)
        self.persist_path = persist_path

        # Initialize vector store
        if vector_store_type == "faiss":
            self.vector_store = FAISSVectorStore(self.embedding_model.embedding_dim)
        elif vector_store_type == "chroma":
            self.vector_store = ChromaVectorStore(persist_directory=persist_path)
        else:
            raise ValueError(f"Unknown vector store type: {vector_store_type}")

        self.vector_store_type = vector_store_type
        self.is_indexed = False

    def index_facts(self, facts: List[str], metadata: List[Dict] = None, show_progress: bool = True):
        """
        Index a list of facts.

        Args:
            facts: List of fact strings
            metadata: Optional metadata for each fact
            show_progress: Whether to show progress
        """
        print(f"Embedding {len(facts)} facts...")
        embeddings = self.embedding_model.embed(facts, show_progress=show_progress)

        print("Adding to vector store...")
        self.vector_store.add(embeddings, facts, metadata)

        self.is_indexed = True
        print(f"Indexed {len(facts)} facts successfully!")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve most relevant facts for a query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        if not self.is_indexed and len(self.vector_store) == 0:
            raise ValueError("No facts indexed. Call index_facts() first.")

        # Embed query
        query_embedding = self.embedding_model.embed_single(query)

        # Search
        results = self.vector_store.search(query_embedding, top_k)

        return results

    def save(self, path: str = None):
        """Save the vector store to disk."""
        path = path or self.persist_path
        if path and self.vector_store_type == "faiss":
            self.vector_store.save(path)
            print(f"Saved vector store to {path}")

    def load(self, path: str = None):
        """Load the vector store from disk."""
        path = path or self.persist_path
        if path and self.vector_store_type == "faiss":
            self.vector_store.load(path)
            self.is_indexed = True
            print(f"Loaded vector store from {path}")


# Example usage
if __name__ == "__main__":
    # Sample facts for testing
    sample_facts = [
        "The Indian government launched PM-KISAN scheme providing Rs 6000 per year to farmers.",
        "India's Chandrayaan-3 landed on the Moon's south pole in August 2023.",
        "Ayushman Bharat covers up to Rs 5 lakh per family per year for hospitalization.",
        "There is no free electricity scheme for all farmers in India.",
        "PM-KUSUM provides subsidized solar panels for farmers.",
    ]

    # Create retrieval system
    retrieval = RetrievalSystem()

    # Index facts
    retrieval.index_facts(sample_facts)

    # Test query
    query = "free electricity to farmers"
    results = retrieval.retrieve(query, top_k=3)

    print(f"\nQuery: {query}")
    print("\nTop results:")
    for result in results:
        print(f"  [{result.rank}] Score: {result.score:.4f}")
        print(f"      {result.fact}")
