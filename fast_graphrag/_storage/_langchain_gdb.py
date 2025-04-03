from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from fast_graphrag._storage._base import BaseStorage, Chunk, Edge, Node, Vector

logger = logging.getLogger(__name__)


class LangChainGraphDB(BaseStorage):
    """
    LangChainGraphDB: A storage implementation that uses LangChain's graph database APIs with NetworkX.

    This class provides methods for managing graph data, including adding, retrieving, deleting, and updating nodes and edges.
    It leverages NetworkX for graph operations.
    """

    def __init__(self, graph: Optional[nx.Graph] = None) -> None:
        """
        Initialize the LangChainGraphDB.

        Args:
            graph (Optional[nx.Graph]): An existing NetworkX graph to use. If None, a new graph is created.
        """
        super().__init__()
        self.graph: nx.Graph = graph if graph is not None else nx.Graph()

    async def add_data(
        self,
        chunks: Optional[List[Chunk]] = None,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
        vectors: Optional[List[Vector]] = None,
    ) -> None:
        """
        Add data to the graph database.

        Args:
            chunks (Optional[List[Chunk]]): A list of Chunk objects to add.
            nodes (Optional[List[Node]]): A list of Node objects to add.
            edges (Optional[List[Edge]]): A list of Edge objects to add.
            vectors (Optional[List[Vector]]): A list of Vector objects to add (not directly added to the graph).

        Raises:
            ValueError: If there are duplicated nodes in the data.
        """
        if nodes:
            node_ids = [node.id for node in nodes]
            if len(node_ids) != len(set(node_ids)):
                raise ValueError("Duplicated nodes in the add_data operation.")

            for node in nodes:
                self.graph.add_node(node.id, content=node.content, metadata=node.metadata)
                logger.debug(f"Added node: {node.id}")

        if edges:
            for edge in edges:
                self.graph.add_edge(
                    edge.source, edge.target, relation=edge.relation, metadata=edge.metadata
                )
                logger.debug(f"Added edge: {edge.source} -> {edge.target}")

    async def retrieve_data(self, node_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieve data from the graph database.

        Args:
            node_ids (Optional[List[str]]): A list of node IDs to retrieve. If None, all nodes are retrieved.

        Returns:
            Dict[str, Any]: A dictionary containing the retrieved data.
        """
        retrieved_data: Dict[str, Any] = {}
        if node_ids is None:
            node_ids = list(self.graph.nodes)
        for node_id in node_ids:
            if node_id in self.graph.nodes:
                retrieved_data[node_id] = self.graph.nodes[node_id]
                logger.debug(f"Retrieved node: {node_id}")
            else:
                logger.warning(f"Node {node_id} not found.")

        return retrieved_data

    async def delete_data(self, node_ids: Optional[List[str]] = None) -> None:
        """
        Delete data from the graph database.

        Args:
            node_ids (Optional[List[str]]): A list of node IDs to delete. If None, all nodes and edges are deleted.
        """
        if node_ids is None:
            self.graph.clear()
            logger.info("Deleted all nodes and edges.")
        else:
            for node_id in node_ids:
                if node_id in self.graph.nodes:
                    self.graph.remove_node(node_id)
                    logger.debug(f"Deleted node: {node_id}")
                else:
                    logger.warning(f"Node {node_id} not found.")

    async def update_data(
        self,
        chunks: Optional[List[Chunk]] = None,
        nodes: Optional[List[Node]] = None,
        edges: Optional[List[Edge]] = None,
        vectors: Optional[List[Vector]] = None,
    ) -> None:
        """
        Update data in the graph database.

        Args:
            chunks (Optional[List[Chunk]]): A list of Chunk objects to update.
            nodes (Optional[List[Node]]): A list of Node objects to update.
            edges (Optional[List[Edge]]): A list of Edge objects to update.
            vectors (Optional[List[Vector]]): A list of Vector objects to update (not directly updated in the graph).

        Raises:
            ValueError: If there are duplicated nodes in the data.
        """
        if nodes:
            node_ids = [node.id for node in nodes]
            if len(node_ids) != len(set(node_ids)):
                raise ValueError("Duplicated nodes in the update_data operation.")
            for node in nodes:
                if node.id in self.graph.nodes:
                    self.graph.nodes[node.id].update(
                        {"content": node.content, "metadata": node.metadata}
                    )
                    logger.debug(f"Updated node: {node.id}")
                else:
                    logger.warning(f"Node {node.id} not found for update.")
                    await self.add_data(nodes=[node])

        if edges:
            for edge in edges:
                if self.graph.has_edge(edge.source, edge.target):
                    self.graph[edge.source][edge.target].update(
                        {"relation": edge.relation, "metadata": edge.metadata}
                    )
                    logger.debug(f"Updated edge: {edge.source} -> {edge.target}")
                else:
                    logger.warning(f"Edge {edge.source} -> {edge.target} not found for update.")
                    await self.add_data(edges=[edge])

    async def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the graph database.

        Returns:
            Dict[str, Any]: A dictionary containing metadata about the graph.
        """
        metadata: Dict[str, Any] = {}
        metadata["num_nodes"] = self.graph.number_of_nodes()
        metadata["num_edges"] = self.graph.number_of_edges()
        return metadata

    async def _insert_start(self) -> None:
        """Prepare the storage for inserting."""
        logger.debug("Preparing the LangChainGraphDB storage for inserting.")

    async def _insert_done(self) -> None:
        """Commit the storage operations after inserting."""
        logger.debug("Committing LangChainGraphDB storage operations after inserting.")

    async def _query_start(self) -> None:
        """Prepare the storage for querying."""
        logger.debug("Preparing the LangChainGraphDB storage for querying.")

    async def _query_done(self) -> None:
        """Release the storage after querying."""
        logger.debug("Releasing LangChainGraphDB storage after querying.")