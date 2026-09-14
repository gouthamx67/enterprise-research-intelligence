from src.rag_engine.advanced.graph import KnowledgeGraph


def main():
    graph = KnowledgeGraph()

    graph.add_node(
        "company",
        "company",
        "Example Corp",
    )

    graph.add_node(
        "product",
        "product",
        "Example AI",
    )

    graph.add_node(
        "customer",
        "segment",
        "Enterprise",
    )

    graph.add_edge(
        "company",
        "launched",
        "product",
    )

    graph.add_edge(
        "company",
        "targets",
        "customer",
    )

    result = graph.search("company")

    assert result.node.name == "Example Corp"
    assert len(result.related_nodes) == 2
    assert len(result.relations) == 2

    neighbor_names = {
        node.name
        for node in result.related_nodes
    }

    assert "Example AI" in neighbor_names
    assert "Enterprise" in neighbor_names

    print("Graph RAG test passed.")


if __name__ == "__main__":
    main()