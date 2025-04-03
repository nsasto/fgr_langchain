from typing import Any, Dict, List, Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._graphrag import BaseGraphRAG, InsertParam, QueryParam
from fast_graphrag._llm import BaseLLMService
from fast_graphrag._services._state_manager import BaseStateManagerService
from fast_graphrag._storage._base import BaseStorage, Chunk, Edge, Node, Vector
from fast_graphrag._storage._langchain_gdb import LangChainGraphDB
from fast_graphrag._storage._langchain_vdb import LangChainVDB
from fast_graphrag._utils import logger


async def test_insertion_and_query():
    """Tests the insertion and query functionality of the BaseGraphRAG class."""
    try:
        # Create LangChainGraphDB and LangChainVDB instances
        graph_db = LangChainGraphDB()
        vector_db = LangChainVDB()

        await graph_db.insert_start()
        
        # Create BaseStateManagerService instance
        state_manager: BaseStateManagerService[Node, Edge, str, Chunk, str, List[float]] = BaseStateManagerService(storage=graph_db)

        await state_manager.query_start()
        # Create BaseGraphRAG instance
        graph_rag: BaseGraphRAG[Vector, str, Chunk, Node, Edge, str] = BaseGraphRAG(
            working_dir=".",
            domain="Test Domain",
            example_queries="Test Queries",
            entity_types=["Test Entity"],
        )
        graph_rag.state_manager = state_manager

        # Data to insert
        data_to_insert: List[str] = [
            "This is the first piece of data.",
            "Here is the second piece of data.",
            "And finally, the third piece of data.",
        ]
        metadata_list: List[Optional[Dict[str, Any]]] = [
            {"source": "document1"},
            {"source": "document2"},
            {"source": "document3"},
        ]

        # Insert data
        logger.info("Inserting data...")
        await graph_rag.insert(content=data_to_insert, metadata=metadata_list, params=InsertParam())
        logger.info("Data insertion completed.")

        # Query data
        logger.info("Querying data...")
        result = graph_rag.query(query="What is the data?", params=QueryParam())
        logger.info("Data query completed.")
        
        logger.info(f"Query result: {result.response}")

        await graph_db.insert_done()
        await state_manager.query_done()

    except InvalidStorageError as e:
        logger.error(f"Error interacting with storage: {e}")
        await graph_db.insert_done()
        await state_manager.query_done()
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_insertion_and_query())