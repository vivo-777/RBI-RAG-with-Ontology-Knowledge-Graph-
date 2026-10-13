from ontology.graph_store import graph_db
from ingestion.decontextualize import METADATA_MAP

def build_ontology_from_metadata():
    """Register all known circulars and their supersession relationships."""
    for filename, meta in METADATA_MAP.items():
        graph_db.add_circular(
            doc_id=filename,
            status="active",
            filename=filename,
            issue_date=meta.get("issue_date"),
        )

    # Second pass: wire up supersessions
    for filename, meta in METADATA_MAP.items():
        superseded_by = meta.get("superseded_by")
        if superseded_by and superseded_by in METADATA_MAP:
            graph_db.add_supersession(superseded_by, filename)

    node_count = graph_db.graph.number_of_nodes()
    edge_count = graph_db.graph.number_of_edges()
    print(f"Ontology built: {node_count} circulars, {edge_count} relationships")
