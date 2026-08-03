"""Renderer-neutral layout planning for the canonical workflow model.

The layout engine does not draw the chart. It annotates the workflow with
stable phase, actor, depth, branch, and routing hints that every renderer can
consume. Keeping these hints in the canonical model prevents Mermaid,
Graphviz, and the future editor from independently guessing the structure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple

from graph_model import Edge, Flowchart, Node, ensure_flowchart

NEGATIVE_LABELS = {
    "no", "invalid", "failure", "failed", "declined", "reject", "rejected",
    "error", "expired", "unavailable", "in use", "not available",
}
POSITIVE_LABELS = {"yes", "valid", "success", "successful", "approved", "available", "pass", "passed"}
RETRY_LABELS = {"retry", "try again", "resubmit", "repeat", "revise", "modify", "correct"}


def _phase_key(node: Node) -> Tuple[int, str]:
    order = node.phase_order if isinstance(node.phase_order, int) else 10_000
    return (order, (node.phase or "Workflow").lower())


def _forward_adjacency(flowchart: Flowchart) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """Return main/forward adjacency while excluding obvious retry edges."""
    node_map = flowchart.node_by_id()
    forward: Dict[str, List[str]] = defaultdict(list)
    indegree = {node.id: 0 for node in flowchart.nodes}

    for edge in flowchart.edges:
        source = node_map.get(edge.source)
        target = node_map.get(edge.target)
        if not source or not target:
            continue
        label = edge.label.strip().lower()
        source_phase = source.phase_order if isinstance(source.phase_order, int) else 0
        target_phase = target.phase_order if isinstance(target.phase_order, int) else source_phase
        obvious_retry = label in RETRY_LABELS or target_phase < source_phase
        if obvious_retry:
            continue
        forward[edge.source].append(edge.target)
        indegree[edge.target] = indegree.get(edge.target, 0) + 1
    return forward, indegree


def _topological_depths(flowchart: Flowchart) -> Dict[str, int]:
    """Compute stable visual depths without allowing retry loops to distort rank."""
    forward, indegree = _forward_adjacency(flowchart)
    queue = deque(node.id for node in flowchart.nodes if indegree.get(node.id, 0) == 0)
    depths = {node.id: 0 for node in flowchart.nodes}
    visited = set()

    while queue:
        current = queue.popleft()
        visited.add(current)
        for target in forward.get(current, []):
            depths[target] = max(depths.get(target, 0), depths.get(current, 0) + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    # Deterministic fallback for nodes participating only in cycles.
    max_depth = max(depths.values(), default=0)
    for node in flowchart.nodes:
        if node.id not in visited:
            max_depth += 1
            depths[node.id] = max_depth
    return depths


def _ordered_phases(flowchart: Flowchart, depths: Dict[str, int]) -> List[str]:
    """Combine explicit phase order with causal depth for a stable phase sequence."""
    phase_stats: Dict[str, Dict[str, int]] = {}
    for index, node in enumerate(flowchart.nodes):
        if node.type in {"start", "end"}:
            continue
        phase = node.phase or "Workflow"
        stats = phase_stats.setdefault(phase, {"explicit": 10_000, "depth": 10_000, "first": index})
        if isinstance(node.phase_order, int):
            stats["explicit"] = min(stats["explicit"], node.phase_order)
        stats["depth"] = min(stats["depth"], depths.get(node.id, 10_000))
        stats["first"] = min(stats["first"], index)

    return sorted(
        phase_stats,
        key=lambda phase: (
            phase_stats[phase]["explicit"],
            phase_stats[phase]["depth"],
            phase_stats[phase]["first"],
            phase.lower(),
        ),
    )


def _ordered_actors(flowchart: Flowchart, depths: Dict[str, int]) -> List[str]:
    actor_stats: Dict[str, Tuple[int, int]] = {}
    for index, node in enumerate(flowchart.nodes):
        if not node.actor:
            continue
        candidate = (depths.get(node.id, 10_000), index)
        actor_stats[node.actor] = min(actor_stats.get(node.actor, candidate), candidate)
    return sorted(actor_stats, key=lambda actor: (*actor_stats[actor], actor.lower()))


def _classify_edge(edge: Edge, node_map: Dict[str, Node], depths: Dict[str, int], retry_index: int) -> Dict[str, object]:
    source = node_map.get(edge.source)
    target = node_map.get(edge.target)
    label = edge.label.strip().lower()
    source_phase = source.phase_order if source and isinstance(source.phase_order, int) else 0
    target_phase = target.phase_order if target and isinstance(target.phase_order, int) else source_phase
    source_depth = depths.get(edge.source, 0)
    target_depth = depths.get(edge.target, source_depth + 1)

    backward = target_phase < source_phase or target_depth <= source_depth
    explicit_retry = label in RETRY_LABELS
    is_retry = backward or explicit_retry or (edge.primary is False and backward)

    if is_retry:
        # Alternate outer routing sides when several retry loops coexist. This
        # reduces the chance that all loopbacks overlap one another.
        side = "left" if retry_index % 2 == 0 else "right"
        return {
            "role": "retry",
            "preferred_side": side,
            "route": f"outer_{side}",
            "constraint": False,
            "priority": 0,
        }

    if edge.primary is False or label in NEGATIVE_LABELS:
        return {
            "role": "alternate",
            "preferred_side": "right",
            "route": "branch_right",
            "constraint": True,
            "priority": 1,
        }

    if label in POSITIVE_LABELS or edge.primary:
        return {
            "role": "forward",
            "preferred_side": "center",
            "route": "main",
            "constraint": True,
            "priority": 3,
        }

    return {
        "role": "alternate",
        "preferred_side": "left",
        "route": "branch_left",
        "constraint": True,
        "priority": 1,
    }


def build_layout_plan(value) -> Flowchart:
    """Return a copy enriched with shared layout metadata for all renderers."""
    flowchart = ensure_flowchart(value).copy()
    depths = _topological_depths(flowchart)
    original_position = {node.id: index for index, node in enumerate(flowchart.nodes)}

    phases = _ordered_phases(flowchart, depths)
    if not phases and any(node.type not in {"start", "end"} for node in flowchart.nodes):
        phases = ["Workflow"]
    phase_index = {phase: index + 1 for index, phase in enumerate(phases)}

    for node in flowchart.nodes:
        if node.type not in {"start", "end"}:
            node.phase = node.phase or "Workflow"
            if node.phase not in phase_index:
                phases.append(node.phase)
                phase_index[node.phase] = len(phases)
            node.phase_order = phase_index[node.phase]

    actors = _ordered_actors(flowchart, depths)
    actor_index = {actor: index + 1 for index, actor in enumerate(actors)}

    # Node metadata survives JSON roundtrips and is ready for future swimlanes.
    for node in flowchart.nodes:
        node.metadata.setdefault("layout", {})
        node.metadata["layout"].update(
            {
                "depth": depths.get(node.id, 0),
                "rank": depths.get(node.id, 0),
                "phase_index": node.phase_order,
                "actor_index": actor_index.get(node.actor),
                "lane": node.actor,
                "order": original_position.get(node.id, 0),
                "main_axis": "vertical",
            }
        )

    node_map = flowchart.node_by_id()
    retry_index = 0
    for edge in flowchart.edges:
        classification = _classify_edge(edge, node_map, depths, retry_index)
        if classification["role"] == "retry":
            retry_index += 1
        edge.metadata.setdefault("layout", {})
        edge.metadata["layout"].update(classification)

    flowchart.phases = phases
    flowchart.actors = actors
    flowchart.metadata.setdefault("layout", {})
    flowchart.metadata["layout"].update(
        {
            "version": 2,
            "direction": "TB",
            "group_by": "phase",
            "phase_order": phases,
            "actor_order": actors,
            "swimlane_ready": bool(actors),
            "retry_routes": "outer_alternating",
            "main_path": "center",
            "branch_policy": "primary-center alternate-side",
        }
    )

    type_priority = {"start": -1, "process": 0, "input_output": 0, "decision": 1, "end": 99}
    flowchart.nodes.sort(
        key=lambda node: (
            -1 if node.type == "start" else (99_999 if node.type == "end" else node.phase_order or 10_000),
            depths.get(node.id, 0),
            actor_index.get(node.actor, 10_000),
            type_priority.get(node.type, 0),
            original_position.get(node.id, 0),
        )
    )
    return flowchart
