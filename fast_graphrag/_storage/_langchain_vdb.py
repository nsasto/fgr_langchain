import uuid
from typing import Any, Dict, List, Optional

import chromadb
from langchain.vectorstores import Chroma

from fast_graphrag._storage._base import BaseStorage, Chunk, Edge, Node, Vector
from fast_graphrag._utils import logger


class LangChainVDB(BaseStorage):
    """
    LangChainVDB: A vector database storage implementation using LangChain and ChromaDB.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initializes the LangChainVDB with an optional persist directory for ChromaDB.

        Args:
            persist_directory (Optional[str]): The directory to persist ChromaDB data.
        """
        super().__init__()
        self._persist_directory = persist_directory
        self._db: Optional[Chroma] = None

    async def _initialize_db(self) -> None:
        """Initializes or loads the ChromaDB instance."""
        if self._db is None:
            client = chromadb.PersistentClient(path=self._persist_directory) if self._persist_directory else chromadb.Client()
            self._db = Chroma(client=client, embedding_function=lambda x: [x])

    async def add_data(
        self,
        chunks: List[Chunk] = [],
        nodes: List[Node] = [],
        edges: List[Edge] = [],
        vectors: List[Vector] = [],
    ) -> None:
        """
        Adds data (chunks, nodes, edges, vectors) to the vector database.

        Args:
            chunks (List[Chunk]): List of Chunk objects to add.
            nodes (List[Node]): List of Node objects to add (not directly added to the VDB).
            edges (List[Edge]): List of Edge objects to add (not directly added to the VDB).
            vectors (List[Vector]): List of Vector objects to add.
        """
        await self._initialize_db()
        if not vectors:
            return

        try:
            self._db.add_embeddings(
                embeddings=[vector.embedding for vector in vectors],
                metadatas=[{} for _ in vectors],
                ids=[vector.id for vector in vectors]
            )

        except Exception as e:
            logger.error(f"Error adding vectors to ChromaDB: {e}")
            raise

    async def retrieve_data(
        self, query_embedding: List[float], top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieves data from the vector database based on a query embedding.

        Args:
            query_embedding (List[float]): The query embedding.
            top_k (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Dict[str, Any]]: List of retrieved data items.
        """
        await self._initialize_db()

        try:
            docs = self._db.similarity_search_by_vector(query_embedding, k=top_k)
            return [{"id": doc.metadata.get("id"), "content": doc.page_content} for doc in docs]
        except Exception as e:
            logger.error(f"Error retrieving data from ChromaDB: {e}")
            raise

    async def delete_data(self, ids: List[str]) -> None:
        """
        Deletes data from the vector database based on IDs.

        Args:
            ids (List[str]): List of IDs to delete.
        """
        await self._initialize_db()
        try:
            self._db._collection.delete(ids=ids)
        except Exception as e:
            logger.error(f"Error deleting data from ChromaDB: {e}")
            raise

    async def update_data(
        self,
        chunks: List[Chunk] = [],
        nodes: List[Node] = [],
        edges: List[Edge] = [],
        vectors: List[Vector] = [],
    ) -> None:
        """
        Updates data in the vector database.

        Args:
            chunks (List[Chunk]): List of Chunk objects to update (not directly updated in the VDB).
            nodes (List[Node]): List of Node objects to update (not directly updated in the VDB).
            edges (List[Edge]): List of Edge objects to update (not directly updated in the VDB).
            vectors (List[Vector]): List of Vector objects to update.
        """
        await self._initialize_db()
        if not vectors:
            return
        try:
            self._db._collection.update(
                ids=[vector.id for vector in vectors],
                embeddings=[vector.embedding for vector in vectors]
            )
        except Exception as e:
            logger.error(f"Error updating vectors in ChromaDB: {e}")
            raise

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves metadata from the vector database.

        Returns:
            Dict[str, Any]: A dictionary of metadata.
        """
        await self._initialize_db()
        try:
            return self._db._collection.count()

        except Exception as e:
            logger.error(f"Error getting metadata from ChromaDB: {e}")
            raise

    async def _insert_start(self) -> None:
        """Prepares the storage for inserting."""
        await self._initialize_db()
        pass

    async def _insert_done(self) -> None:
        """Commits the storage operations after inserting."""
        pass

    async def _query_start(self) -> None:
        """Prepares the storage for querying."""
        await self._initialize_db()
        pass

    async def _query_done(self) -> None:
        """Releases the storage after querying."""
        pass