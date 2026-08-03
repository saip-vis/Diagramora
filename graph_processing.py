"""Deterministic graph cleanup, validation, and structural utilities."""

import difflib

def merge_similar_nodes(graph: dict, threshold: float = 0.55) -> dict:
    """Deterministically merge near-duplicate nodes based on label text similarity —
    a plain-code safety net that doesn't depend on the LLM correctly self-reporting
    duplicates, since that's proven unreliable on its own across several rounds of
    prompt tuning. Redirects any edges pointing at a merged-away node to its survivor."""
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))
    id_redirect = {}
    survivors = []

    for node in nodes:
        merged_into = None
        for survivor in survivors:
            similarity = difflib.SequenceMatcher(
                None, node["label"].lower().strip(), survivor["label"].lower().strip()
            ).ratio()
            if similarity >= threshold:
                merged_into = survivor
                break
        if merged_into:
            id_redirect[node["id"]] = merged_into["id"]
            if len(node["label"]) > len(merged_into["label"]):
                merged_into["label"] = node["label"]  # keep the more complete phrasing
        else:
            survivors.append(dict(node))
            id_redirect[node["id"]] = node["id"]

    new_edges = []
    seen_keys = set()
    for edge in edges:
        new_from = id_redirect.get(edge["from"], edge["from"])
        new_to = id_redirect.get(edge["to"], edge["to"])
        if new_from == new_to:
            continue  # collapsed into a self-loop after merging — drop it
        key = (new_from, new_to)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_edge = dict(edge)
        merged_edge["from"] = new_from
        merged_edge["to"] = new_to
        new_edges.append(merged_edge)

    return {"nodes": survivors, "edges": new_edges}, id_redirect

def normalize_final_graph(graph: dict) -> dict:
    """Normalize the final graph without inventing business steps.

    The function keeps one canonical Start and End node, removes malformed data,
    preserves optional phase metadata, deduplicates edges, and connects genuinely
    terminal outcomes to End. Semantic repairs are handled by the graph-lint pass.
    """
    allowed_types = {"start", "process", "decision", "input_output", "end"}
    raw_nodes = list(graph.get("nodes", []))
    nodes = []
    seen_ids = set()
    first_start_id = None
    first_end_id = None
    redirects = {}

    for raw in raw_nodes:
        node_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        if not node_id or not label or node_id in seen_ids:
            continue
        node_type = str(raw.get("type", "process")).strip().lower()
        if node_type not in allowed_types:
            node_type = "process"

        if node_type == "start":
            if first_start_id is None:
                first_start_id = node_id
            else:
                redirects[node_id] = first_start_id
                continue
        if node_type == "end":
            if first_end_id is None:
                first_end_id = node_id
            else:
                redirects[node_id] = first_end_id
                continue

        normalized = {"id": node_id, "label": label, "type": node_type}
        phase = str(raw.get("phase", "")).strip()
        if phase:
            normalized["phase"] = phase
        try:
            phase_order = int(raw.get("phase_order"))
            normalized["phase_order"] = phase_order
        except (TypeError, ValueError):
            pass
        nodes.append(normalized)
        seen_ids.add(node_id)

    valid_ids = {n["id"] for n in nodes}
    edges = []
    seen_edges = set()
    for raw in graph.get("edges", []):
        source = redirects.get(str(raw.get("from", "")).strip(), str(raw.get("from", "")).strip())
        target = redirects.get(str(raw.get("to", "")).strip(), str(raw.get("to", "")).strip())
        if source not in valid_ids or target not in valid_ids or source == target:
            continue
        label = str(raw.get("label", "")).strip()
        key = (source, target, label.lower())
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({
            "from": source,
            "to": target,
            "label": label,
            "primary": bool(raw.get("primary", True)),
        })

    semantic_nodes = [n for n in nodes if n["type"] not in {"start", "end"}]
    valid_ids = {n["id"] for n in nodes}

    if semantic_nodes:
        incoming = {node_id: 0 for node_id in valid_ids}
        outgoing = {node_id: 0 for node_id in valid_ids}
        for edge in edges:
            incoming[edge["to"]] = incoming.get(edge["to"], 0) + 1
            outgoing[edge["from"]] = outgoing.get(edge["from"], 0) + 1

        roots = [n for n in semantic_nodes if incoming.get(n["id"], 0) == 0]
        requested_start = str(graph.get("start_node_id", "")).strip()
        root_ids = {n["id"] for n in roots}
        if requested_start not in valid_ids or requested_start not in root_ids:
            requested_start = roots[0]["id"] if roots else semantic_nodes[0]["id"]

        existing_start = next((n for n in nodes if n["type"] == "start"), None)
        if existing_start is None:
            start_id = "__start__"
            while start_id in valid_ids:
                start_id += "_"
            existing_start = {"id": start_id, "label": "Start", "type": "start"}
            nodes.insert(0, existing_start)
            valid_ids.add(start_id)
        else:
            start_id = existing_start["id"]
            existing_start["label"] = "Start"

        # A start node must have exactly one outgoing edge into the real workflow.
        edges = [e for e in edges if e["from"] != start_id]
        edges.insert(0, {"from": start_id, "to": requested_start, "label": "", "primary": True})

        existing_end = next((n for n in nodes if n["type"] == "end"), None)
        if existing_end is None:
            end_id = "__end__"
            while end_id in valid_ids:
                end_id += "_"
            existing_end = {"id": end_id, "label": "End", "type": "end"}
            nodes.append(existing_end)
            valid_ids.add(end_id)
        else:
            end_id = existing_end["id"]
            existing_end["label"] = "End"

        # End cannot have outgoing edges.
        edges = [e for e in edges if e["from"] != end_id]

        current_outgoing = {node_id: 0 for node_id in valid_ids}
        for edge in edges:
            current_outgoing[edge["from"]] = current_outgoing.get(edge["from"], 0) + 1
        terminal_ids = [
            n["id"] for n in nodes
            if n["type"] not in {"start", "end"} and current_outgoing.get(n["id"], 0) == 0
        ]
        existing_to_end = {e["from"] for e in edges if e["to"] == end_id}
        for terminal_id in terminal_ids:
            if terminal_id not in existing_to_end:
                edges.append({"from": terminal_id, "to": end_id, "label": "", "primary": True})

    return {
        "nodes": nodes,
        "edges": edges,
        "start_node_id": next((n["id"] for n in nodes if n["type"] == "start"), None),
    }

def lint_graph(graph: dict) -> list:
    """Return deterministic structural warnings for the normalized graph."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_by_id = {n["id"]: n for n in nodes}
    starts = [n for n in nodes if n.get("type") == "start"]
    ends = [n for n in nodes if n.get("type") == "end"]
    outgoing = {n["id"]: [] for n in nodes}
    incoming = {n["id"]: [] for n in nodes}
    warnings = []

    for edge in edges:
        if edge.get("from") in outgoing:
            outgoing[edge["from"]].append(edge)
        if edge.get("to") in incoming:
            incoming[edge["to"]].append(edge)

    if len(starts) != 1:
        warnings.append(f"Graph must contain exactly one Start node; found {len(starts)}.")
    if len(ends) != 1:
        warnings.append(f"Graph must contain exactly one End node; found {len(ends)}.")

    for node in nodes:
        node_id = node["id"]
        node_type = node.get("type", "process")
        if node_type == "start" and len(outgoing[node_id]) != 1:
            warnings.append(f"Start node '{node['label']}' must have exactly one outgoing edge.")
        if node_type == "end" and outgoing[node_id]:
            warnings.append("End node must not have outgoing edges.")
        if node_type not in {"start"} and not incoming[node_id]:
            warnings.append(f"Node '{node['label']}' is disconnected from the preceding process.")
        if node_type not in {"end"} and not outgoing[node_id]:
            warnings.append(f"Node '{node['label']}' has no outcome or continuation.")
        if node_type == "decision":
            count = len(outgoing[node_id])
            if count != 2:
                warnings.append(f"Decision '{node['label']}' must have exactly two outgoing branches; found {count}.")
            labels = [e.get("label", "").strip().lower() for e in outgoing[node_id]]
            if any(not label for label in labels):
                warnings.append(f"Decision '{node['label']}' has an unlabeled branch.")
            if len(labels) != len(set(labels)):
                warnings.append(f"Decision '{node['label']}' has duplicate branch labels.")
        elif node_type not in {"start", "end"} and len(outgoing[node_id]) > 2:
            warnings.append(f"Process node '{node['label']}' has too many outgoing edges ({len(outgoing[node_id])}).")

    # Reachability from Start.
    if len(starts) == 1:
        start_id = starts[0]["id"]
        seen = set()
        stack = [start_id]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(e["to"] for e in outgoing.get(current, []))
        for node in nodes:
            if node["id"] not in seen:
                warnings.append(f"Node '{node['label']}' is unreachable from Start.")

    # Detect suspicious backward phase jumps. Secondary retry edges are allowed.
    for edge in edges:
        source = node_by_id.get(edge["from"], {})
        target = node_by_id.get(edge["to"], {})
        source_order = source.get("phase_order")
        target_order = target.get("phase_order")
        if isinstance(source_order, int) and isinstance(target_order, int):
            if target_order < source_order and edge.get("primary", True):
                warnings.append(
                    f"Primary edge from '{source.get('label')}' to '{target.get('label')}' moves backward across phases."
                )

    # De-duplicate while preserving order.
    return list(dict.fromkeys(warnings))

def prune_unreachable_nodes(graph: dict) -> dict:
    """Remove truly unreachable islands after the repair pass."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    start = next((n for n in nodes if n.get("type") == "start"), None)
    if not start:
        return graph
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])
    reachable = set()
    stack = [start["id"]]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(outgoing.get(current, []))
    return {
        **graph,
        "nodes": [n for n in nodes if n["id"] in reachable],
        "edges": [e for e in edges if e["from"] in reachable and e["to"] in reachable],
    }

def compute_start_node(graph: dict) -> str:
    """Fallback: if no start_node_id was provided, pick whichever node has no
    incoming edges (the first node added is used as a tiebreaker)."""
    if not graph.get("nodes"):
        return None
    targets = {e["to"] for e in graph.get("edges", [])}
    for node in graph["nodes"]:
        if node["id"] not in targets:
            return node["id"]
    return graph["nodes"][0]["id"]
