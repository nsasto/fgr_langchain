# fgr_langchain
An implementation of circlemind's fast graph leveraging langchain

## Objectives

An experiemnt to enhance [fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) by CircleMind AI by:

1.  Adding an abstraction layer for storage.
2.  Integrating with LangChain's graph storage options.
3.  Integrating with LangChain's vector storage options.
4.  Using LangChain for chunking.
5.  Keeping the existing storage implementations for backwards compatibility.

with luck, this will create extensibility without impacting the performance advantage

**Overall Goal:**

The primary goal is to enhance the Fast GraphRAG library by leveraging LangChain's robust components for chunking, vector storage, and, crucially, graph storage and management. This will provide Fast GraphRAG with:

*   **Increased Flexibility:** Users can choose from various chunking strategies, vector stores, and now, graph databases, as offered by LangChain.
*   **Improved Scalability:** Leveraging established vector and graph databases will enable Fast GraphRAG to handle larger datasets and more complex graphs.
*   **Ecosystem Integration:** Tying into the LangChain ecosystem makes Fast GraphRAG more accessible to LangChain users and allows for seamless combination with other LangChain tools.
*   **Focus on Core Strength:** By offloading these common tasks to LangChain, the Fast GraphRAG project can focus even more on refining its unique graph creation and retrieval algorithms.
*   **Standardization:** By adopting langchain implementations, we will create an easier to learn and use API.

**Key Components:**

1.  **Abstraction Layer (Central to the Change):**

*   fast\_graphrag/\_storage/\_base.py will be refactored to act as an abstraction layer.
*   This layer will allow users to select their preferred storage backend:

*   **Original Bespoke Implementations:** The current in-memory vector store and the igraph-based graph store will still be available.
*   **LangChain Vector Stores:** Any of LangChain's supported vector databases (e.g., Chroma, FAISS, Pinecone, etc.) can be used.
*   **LangChain Graph Stores:** Any of LangChain's supported graph databases (e.g., Neo4j, Memgraph, NetworkX) can be used.

*   The abstraction will provide a unified API for common operations like:

*   Adding chunks/nodes/vectors/relationships.
*   Retrieving chunks/nodes/vectors/relationships.
*   Deleting/updating.
*   Getting metadata.

3.  **LangChain Graph Database Integration:**

*   A new module, fast\_graphrag/\_storage/\_langchain\_gdb.py, will be created to house the LangChain graph database integration.
*   This module will implement the methods required by the \_base.py abstraction layer, but using LangChain's graph database APIs.
*   We will initially support at least one graph database (e.g., Neo4j) but make it easy to add others later.

5.  **LangChain Vector Database Integration:**

*   The existing fast\_graphrag/\_storage/\_vdb\_hnswlib.py provides a basic integration with the HNSWlib vector store, we will need to review if it needs to be re-written to work with the abstraction layer.
*   Other langchain integrations can be added later.
*   If necessary, we will create a fast\_graphrag/\_storage/\_langchain\_vdb.py to abstract away the vector db integrations.

7.  **LangChain Chunking:**

*   Replace the current, likely basic, chunking logic within Fast GraphRAG with LangChain's text splitters.
*   Allow users to select and configure the desired text splitter through the Fast GraphRAG API.

9.  **Maintain Old Implementations**

*   We will keep the original implementations for users who prefer that or to use as fallback if necessary.
