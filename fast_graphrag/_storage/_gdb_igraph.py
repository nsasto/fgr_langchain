import gzip
import os
from dataclasses import asdict, field
from typing import Any, Iterable, List, Mapping, Optional, Tuple, Union

import igraph as ig  # type: ignore
import numpy as np
from scipy.sparse import csr_matrix

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._types import GTId, TIndex
from fast_graphrag._utils import csr_from_indices_list, logger

from ._base import BaseStorage, Chunk, Edge, Node, Vector


class IGraphDB(BaseStorage):
    """
    Graph database implementation using igraph.

    This class inherits from BaseStorage and provides methods for managing
    a graph database using the igraph library.
    """

    RESOURCE_NAME = "igraph_data.pklz"
    ppr_damping: float = field(default=0.85)
    _graph: Optional[ig.Graph] = field(init=False, default=None)  # type: ignore

    async def save_graphml(self, path: str) -> None:
        """Saves the graph to a GraphML file."""
        try:
            if self._graph is not None:  # type: ignore
                ig.Graph.write_graphmlz(self._graph, path + ".gz")  # type: ignore

                with gzip.open(path + ".gz", "rb") as f:
                    file_content = f.read()
                with open(path, "wb") as f:
                    f.write(file_content)
                os.remove(path + ".gz")
        except Exception as e:
            logger.error(f"Error saving graph to GraphML: {e}")
            raise

    async def node_count(self) -> int:
        """Returns the number of nodes in the graph."""
        return self._graph.vcount()  # type: ignore

    async def edge_count(self) -> int:
        return self._graph.ecount()  # type: ignore

    async def get_node(self, node: Union[GTNode, GTId]) -> Union[Tuple[GTNode, TIndex], Tuple[None, None]]:
        if isinstance(node, self.config.node_cls):
            node_id = node.id
        else:
            node_id = node

        try:
            vertex = self._graph.vs.find(name=node_id)  # type: ignore
            return (
                Node(
                    id=vertex["name"],
                    content=vertex.attributes().get("content", ""),
                    metadata=vertex.attributes().get("metadata", {}),
                ),
                vertex.index,
            )
        except ValueError:
            return (None, None)
        except Exception as e:
            logger.error(f"Error retrieving node: {e}")
            raise

    async def get_all_edges(self) -> Iterable[Edge]:
        """Retrieves all edges from the graph."""
        return [
            Edge(source=e.source_vertex["name"], target=e.target_vertex["name"], relation="", metadata=e.attributes())
            for e in self._graph.es
        ]

    async def get_edge_indices(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[TIndex]:
        if type(source_node) is TIndex:
            source_node = self._graph.vs.find(name=source_node).index  # type: ignore
        if type(target_node) is TIndex:
            target_node = self._graph.vs.find(name=target_node).index  # type: ignore
        edges = self._graph.es.select(_source=source_node, _target=target_node)  # type: ignore

        return [edge.index for edge in edges]  # type: ignore

    async def get_edges(
        self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]
    ) -> Iterable[Tuple[Edge, TIndex]]:
        """Retrieves edges between two nodes."""
        indices = await self.get_edge_indices(source_node, target_node)
        edges: List[Tuple[Edge, TIndex]] = []
        for index in indices:
            edge = await self.get_edge_by_index(index)
            if edge:
                edges.append((edge, index))
        return edges

    async def get_edge_ids(self) -> Iterable[GTId]:
        raise NotImplementedError

    async def get_node_by_index(self, index: TIndex) -> Union[GTNode, None]:
        node = self._graph.vs[index] if index < self._graph.vcount() else None  # type: ignore
        return self.config.node_cls(**node.attributes()) if index < self._graph.vcount() else None  # type: ignore

    async def get_edge_by_index(self, index: TIndex) -> Union[GTEdge, None]:
        edge = self._graph.es[index] if index < self._graph.ecount() else None  # type: ignore
        if edge:
            return Edge(
                source=edge.source_vertex["name"],
                target=edge.target_vertex["name"],
                relation="",  # Assuming relation is not stored as an attribute
                metadata=edge.attributes(),
            )
        else:
            return None

    async def get_all_nodes(self) -> Iterable[Node]:
        raise NotImplementedError

    async def upsert_node(self, node: GTNode, node_index: Union[TIndex, None]) -> TIndex:
        if node_index is not None:
            if node_index >= self._graph.vcount():  # type: ignore
                logger.error(
                    f"Trying to update node with index {node_index} but graph has only {self._graph.vcount()} nodes."  # type: ignore
                )
                raise ValueError(f"Index {node_index} is out of bounds")
            try:
                already_node = self._graph.vs[node_index]  # type: ignore
                already_node.update_attributes(name=node.id, content=node.content, **node.metadata)  # type: ignore
                return already_node.index  # type: ignore
            except Exception as e:
                logger.error(f"Error updating node: {e}")
                raise
        else:
            try:
                return self._graph.add_vertex(name=node.id, content=node.content, **node.metadata).index  # type: ignore
            except Exception as e:
                logger.error(f"Error inserting node: {e}")
                raise
    async def upsert_edge(self, edge: GTEdge, edge_index: Union[TIndex, None]) -> TIndex:
        if edge_index is not None:
            if edge_index >= self._graph.ecount():  # type: ignore
                logger.error(
                    f"Trying to update edge with index {edge_index} but graph has only {self._graph.ecount()} edges."  # type: ignore
                )
                raise ValueError(f"Index {edge_index} is out of bounds")
            try:
                already_edge = self._graph.es[edge_index]  # type: ignore
                already_edge.update_attributes(**edge.metadata)  # type: ignore
                return already_edge.index  # type: ignore
            except Exception as e:
                logger.error(f"Error updating edge: {e}")
                raise
        else:
            try:
                source_index = self._graph.vs.find(name=edge.source).index  # type: ignore
                target_index = self._graph.vs.find(name=edge.target).index  # type: ignore
                return self._graph.add_edge(
                    source=source_index, target=target_index, **edge.metadata
                ).index  # type: ignore
            except Exception as e:
                logger.error(f"Error inserting edge: {e}")
                raise

    async def insert_edges(
        self,
        edges: Optional[Iterable[GTEdge]] = None,
        indices: Optional[Iterable[Tuple[TIndex, TIndex]]] = None,
        attrs: Optional[Mapping[str, Iterable[Any]]] = None,
    ) -> List[TIndex]:
        if indices is not None:
            assert edges is None, "Cannot provide both indices and edges."
            assert attrs is not None, "Attributes must be provided with indices."
            indices = list(indices)
            if len(indices) == 0:
                return []
            self._graph.add_edges(  # type: ignore
                indices,
                attributes=attrs,
            )
            # TODO: not sure if this is the best way to get the indices of the new edges
            return list(range(self._graph.ecount() - len(indices), self._graph.ecount()))  # type: ignore
        elif edges is not None:
            assert indices is None, "Cannot provide both indices and edges."
            edges = list(edges)
            if len(edges) == 0:
                return []
            try:
                source_target_pairs = []
                attrs_list: List[dict] = []
                for edge in edges:
                    source_index = self._graph.vs.find(name=edge.source).index  # type: ignore
                    target_index = self._graph.vs.find(name=edge.target).index  # type: ignore
                    source_target_pairs.append((source_index, target_index))
                    attrs_list.append(edge.metadata)
                self._graph.add_edges(source_target_pairs, attributes=attrs_list)  # type: ignore
                # TODO: not sure if this is the best way to get the indices of the new edges
                return list(range(self._graph.ecount() - len(edges), self._graph.ecount()))  # type: ignore
            except Exception as e:
                logger.error(f"Error inserting edges: {e}")
                raise
        else:
            return []

    async def are_neighbours(self, source_node: Union[GTId, TIndex], target_node: Union[GTId, TIndex]) -> bool:
        return self._graph.get_eid(source_node, target_node, directed=False, error=False) != -1  # type: ignore

    async def delete_data(
        self, chunks: Optional[List[Chunk]] = None, nodes: Optional[List[Node]] = None, edges: Optional[List[Edge]] = None, vectors: Optional[List[Vector]] = None
    ) -> None:
        """Deletes data from the graph."""
        if edges:
            try:
                edge_indices = [self._graph.es.find(source=e.source, target=e.target).index for e in edges]  # type: ignore
                self._graph.delete_edges(edge_indices)  # type: ignore
            except Exception as e:
                logger.error(f"Error deleting edges: {e}")
                raise

    async def update_data(self, chunks: Optional[List[Chunk]] = None, nodes: Optional[List[Node]] = None, edges: Optional[List[Edge]] = None, vectors: Optional[List[Vector]] = None) -> None:
        raise NotImplementedError
    async def score_nodes(self, initial_weights: Optional[csr_matrix]) -> csr_matrix:
        if self._graph.vcount() == 0:  # type: ignore
            logger.info("Trying to score nodes in an empty graph.")
            return csr_matrix((1, 0))

        reset_prob = initial_weights.toarray().flatten() if initial_weights is not None else None

        ppr_scores = self._graph.personalized_pagerank(
            damping=self.config.ppr_damping, directed=False, reset=reset_prob
        )
        ppr_scores = np.array(ppr_scores, dtype=np.float32)  # type: ignore

        return csr_matrix(
            ppr_scores.reshape(1, -1)  # type: ignore
        )

    async def get_entities_to_relationships_map(self) -> csr_matrix:
        if len(self._graph.vs) == 0:  # type: ignore
            return csr_matrix((0, 0))

        return csr_from_indices_list(
            [
                [edge.index for edge in vertex.incident()]  # type: ignore
                for vertex in self._graph.vs  # type: ignore
            ],
            shape=(await self.node_count(), await self.edge_count()),
        )

    async def get_metadata(self) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError

    async def get_relationships_attrs(self, key: str) -> List[List[Any]]:
        if len(self._graph.es) == 0:  # type: ignore
            return []
        try:
            return [list(attr) for attr in self._graph.es[key]]  # type: ignore
        except KeyError:
            return []
        lists_of_attrs: List[List[TIndex]] = []
        for attr in self._graph.es[key]:  # type: ignore
            lists_of_attrs.append(list(attr))  # type: ignore

        return lists_of_attrs

    async def _insert_start(self):
        """
        Prepares the storage for insert operations.

        If a namespace is provided, it attempts to load an existing graph.
        Otherwise, it creates a new in-memory graph.
        """
        try:
            if self.namespace:
                graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)

                if graph_file_name:
                    try:
                        self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                        logger.debug(f"Loaded graph storage '{graph_file_name}'.")
                    except Exception as e:
                        t = f"Error loading graph from {graph_file_name}: {e}"
                        logger.error(t)
                        raise InvalidStorageError(t) from e
                else:
                    logger.info(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
                    self._graph = ig.Graph(directed=False)
            else:
                self._graph = ig.Graph(directed=False)
                logger.debug("Creating new volatile graphdb storage.")
        except Exception as e:
            logger.error(f"Error during insert start: {e}")
            raise

    async def _insert_done(self):
        """
        Commits the insert operations by saving the graph to a file
        if a namespace is provided.
        """
        try:
            if self.namespace:
                graph_file_name = self.namespace.get_save_path(self.RESOURCE_NAME)
                try:
                    ig.Graph.write_picklez(self._graph, graph_file_name)  # type: ignore
                    logger.debug(f"Saved graph to '{graph_file_name}'.")
                except Exception as e:
                    t = f"Error saving graph to {graph_file_name}: {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
        except Exception as e:
            logger.error(f"Error during insert done: {e}")
            raise

    async def _query_start(self):
        """
        Prepares the storage for query operations by loading the graph
        from a file if a namespace is provided.
        """
        try:
            assert self.namespace, "Loading a graph requires a namespace."
            graph_file_name = self.namespace.get_load_path(self.RESOURCE_NAME)
            if graph_file_name:
                try:
                    self._graph = ig.Graph.Read_Picklez(graph_file_name)  # type: ignore
                    logger.debug(f"Loaded graph storage '{graph_file_name}'.")
                except Exception as e:
                    t = f"Error loading graph from '{graph_file_name}': {e}"
                    logger.error(t)
                    raise InvalidStorageError(t) from e
            else:
                logger.warning(f"No data file found for graph storage '{graph_file_name}'. Loading empty graph.")
                self._graph = ig.Graph(directed=False)
        except Exception as e:
            logger.error(f"Error during query start: {e}")
            raise

    async def _query_done(self):
        pass
