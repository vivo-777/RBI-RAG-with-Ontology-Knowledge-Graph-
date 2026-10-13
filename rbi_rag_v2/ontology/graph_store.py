import networkx as nx
from typing import List, Dict, Any, Optional
import json
import os

GRAPH_PERSIST_PATH = os.path.join(os.path.dirname(__file__), "..", "qdrant_data", "graph.json")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(GRAPH_PERSIST_PATH):
            with open(GRAPH_PERSIST_PATH, "r") as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data)

    def save_to_disk(self):
        os.makedirs(os.path.dirname(GRAPH_PERSIST_PATH), exist_ok=True)
        with open(GRAPH_PERSIST_PATH, "w") as f:
            json.dump(nx.node_link_data(self.graph), f)

    def add_circular(self, doc_id: str, status: str = "active", **kwargs):
        self.graph.add_node(doc_id, type="Circular", status=status, **kwargs)

    def add_supersession(self, new_doc_id: str, old_doc_id: str):
        self.graph.add_edge(new_doc_id, old_doc_id, relation="supersedes")
        # Update old doc status
        if self.graph.has_node(old_doc_id):
            self.graph.nodes[old_doc_id]["status"] = "superseded"

    def add_amendment(self, new_doc_id: str, old_doc_id: str):
        self.graph.add_edge(new_doc_id, old_doc_id, relation="amends")

    def get_active_circular(self, doc_id: str) -> Optional[str]:
        """
        If a circular is superseded, trace up to find the active one that supersedes it.
        Returns the active doc_id, or None if the circular is fundamentally invalid.
        """
        if not self.graph.has_node(doc_id):
            return doc_id

        status = self.graph.nodes[doc_id].get("status", "active")
        if status == "active":
            return doc_id

        # Trace predecessors for supersedes relation
        for pred in self.graph.predecessors(doc_id):
            if self.graph.edges[pred, doc_id].get("relation") == "supersedes":
                return self.get_active_circular(pred)
                
        return doc_id
        
# Global singleton for local dev
graph_db = KnowledgeGraph()
