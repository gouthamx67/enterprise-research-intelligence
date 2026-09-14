from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    name: str


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    relation: str
    target_id: str


@dataclass
class GraphSearchResult:
    node: GraphNode
    related_nodes: list[GraphNode] = field(default_factory=list)
    relations: list[GraphEdge] = field(default_factory=list)


class KnowledgeGraph:
    """
    Simple in-memory knowledge graph for Graph RAG.

    Nodes represent entities such as companies, products,
    customers, competitors, and markets.

    Edges represent relationships between those entities.
    """

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
    ) -> GraphNode:
        if not node_id or not node_id.strip():
            raise ValueError("node_id cannot be empty")

        if not node_type or not node_type.strip():
            raise ValueError("node_type cannot be empty")

        if not name or not name.strip():
            raise ValueError("name cannot be empty")

        node = GraphNode(
            node_id=node_id.strip(),
            node_type=node_type.strip(),
            name=name.strip(),
        )

        self.nodes[node.node_id] = node

        return node

    def add_edge(
        self,
        source_id: str,
        relation: str,
        target_id: str,
    ) -> GraphEdge:
        if source_id not in self.nodes:
            raise ValueError(
                f"unknown source node: {source_id}"
            )

        if target_id not in self.nodes:
            raise ValueError(
                f"unknown target node: {target_id}"
            )

        if not relation or not relation.strip():
            raise ValueError("relation cannot be empty")

        edge = GraphEdge(
            source_id=source_id,
            relation=relation.strip(),
            target_id=target_id,
        )

        self.edges.append(edge)

        return edge

    def neighbors(
        self,
        node_id: str,
    ) -> list[GraphNode]:
        if node_id not in self.nodes:
            raise ValueError(
                f"unknown node: {node_id}"
            )

        neighbor_ids: list[str] = []

        for edge in self.edges:
            if edge.source_id == node_id:
                neighbor_ids.append(edge.target_id)

            elif edge.target_id == node_id:
                neighbor_ids.append(edge.source_id)

        return [
            self.nodes[neighbor_id]
            for neighbor_id in neighbor_ids
        ]

    def search(
        self,
        node_id: str,
    ) -> GraphSearchResult:
        if node_id not in self.nodes:
            raise ValueError(
                f"unknown node: {node_id}"
            )

        related_nodes = self.neighbors(node_id)

        related_edges = [
            edge
            for edge in self.edges
            if (
                edge.source_id == node_id
                or edge.target_id == node_id
            )
        ]

        return GraphSearchResult(
            node=self.nodes[node_id],
            related_nodes=related_nodes,
            relations=related_edges,
        )