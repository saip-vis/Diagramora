"""Canonical workflow model used throughout the application.

The AI still exchanges JSON, but the backend immediately converts that JSON into
these typed objects. Mermaid, Graphviz, future editors, and persistence can all
use the same representation instead of treating renderer output as the source
of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

ALLOWED_NODE_TYPES = {"start", "process", "decision", "input_output", "end"}


@dataclass
class Node:
    id: str
    label: str
    type: str = "process"
    phase: Optional[str] = None
    phase_order: Optional[int] = None
    actor: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        node_type = str(data.get("type", "process")).strip().lower()
        if node_type not in ALLOWED_NODE_TYPES:
            node_type = "process"

        phase_order = data.get("phase_order")
        try:
            phase_order = int(phase_order) if phase_order is not None else None
        except (TypeError, ValueError):
            phase_order = None

        known = {"id", "label", "type", "phase", "phase_order", "actor", "metadata"}
        extra = {k: v for k, v in data.items() if k not in known}
        metadata = dict(data.get("metadata") or {})
        metadata.update(extra)

        return cls(
            id=str(data.get("id", "")).strip(),
            label=str(data.get("label", "")).strip(),
            type=node_type,
            phase=(str(data.get("phase", "")).strip() or None),
            phase_order=phase_order,
            actor=(str(data.get("actor", "")).strip() or None),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "type": self.type,
        }
        if self.phase:
            data["phase"] = self.phase
        if self.phase_order is not None:
            data["phase_order"] = self.phase_order
        if self.actor:
            data["actor"] = self.actor
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass
class Edge:
    source: str
    target: str
    label: str = ""
    primary: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        known = {"from", "to", "source", "target", "label", "primary", "metadata"}
        extra = {k: v for k, v in data.items() if k not in known}
        metadata = dict(data.get("metadata") or {})
        metadata.update(extra)
        return cls(
            source=str(data.get("from", data.get("source", ""))).strip(),
            target=str(data.get("to", data.get("target", ""))).strip(),
            label=str(data.get("label", "")).strip(),
            primary=bool(data.get("primary", True)),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "from": self.source,
            "to": self.target,
            "label": self.label,
            "primary": self.primary,
        }
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass
class Flowchart:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    title: Optional[str] = None
    start_node_id: Optional[str] = None
    phases: List[str] = field(default_factory=list)
    actors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Flowchart":
        data = data or {}
        nodes = [Node.from_dict(item) for item in data.get("nodes", []) if isinstance(item, dict)]
        edges = [Edge.from_dict(item) for item in data.get("edges", []) if isinstance(item, dict)]

        phases = list(data.get("phases") or [])
        if not phases:
            phase_pairs = sorted(
                {(n.phase_order, n.phase) for n in nodes if n.phase and n.phase_order is not None},
                key=lambda pair: pair[0],
            )
            phases = [phase for _, phase in phase_pairs]

        actors = list(data.get("actors") or [])
        if not actors:
            actors = list(dict.fromkeys(n.actor for n in nodes if n.actor))

        known = {"nodes", "edges", "title", "start_node_id", "phases", "actors", "metadata", "lint_warnings"}
        extra = {k: v for k, v in data.items() if k not in known}
        metadata = dict(data.get("metadata") or {})
        metadata.update(extra)
        if "lint_warnings" in data:
            metadata["lint_warnings"] = list(data.get("lint_warnings") or [])

        return cls(
            nodes=nodes,
            edges=edges,
            title=(str(data.get("title", "")).strip() or None),
            start_node_id=(str(data.get("start_node_id", "")).strip() or None),
            phases=phases,
            actors=actors,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if self.title:
            data["title"] = self.title
        if self.start_node_id:
            data["start_node_id"] = self.start_node_id
        if self.phases:
            data["phases"] = list(self.phases)
        if self.actors:
            data["actors"] = list(self.actors)
        if self.metadata:
            metadata = dict(self.metadata)
            lint_warnings = metadata.pop("lint_warnings", None)
            if metadata:
                data["metadata"] = metadata
            if lint_warnings is not None:
                data["lint_warnings"] = lint_warnings
        return data

    def node_by_id(self) -> Dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def outgoing_edges(self, node_id: str) -> List[Edge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def incoming_edges(self, node_id: str) -> List[Edge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def copy(self) -> "Flowchart":
        return Flowchart.from_dict(self.to_dict())


def ensure_flowchart(value: Any) -> Flowchart:
    if isinstance(value, Flowchart):
        return value
    if isinstance(value, dict) or value is None:
        return Flowchart.from_dict(value)
    raise TypeError(f"Expected Flowchart or dict, received {type(value).__name__}")


def ensure_graph_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, Flowchart):
        return value.to_dict()
    if isinstance(value, dict):
        return value
    raise TypeError(f"Expected Flowchart or dict, received {type(value).__name__}")
