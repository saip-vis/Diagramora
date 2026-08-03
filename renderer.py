"""Renderer registry and Graphviz-based vector/raster exports.

The canonical Flowchart object is the source of truth. Renderers consume the
same layout plan and may output PDF, SVG, or PNG without changing graph logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from graphviz import Digraph

from config import COLOR_BLACK, COLOR_DARK_BLUE, COLOR_WHITE
from graph_model import Flowchart, ensure_flowchart
from graph_processing import compute_start_node
from layout_engine import build_layout_plan

SUPPORTED_FORMATS = {"pdf", "svg", "png"}


def sanitize_filename(title: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "" for c in title)
    safe = safe.strip().replace(" ", "_").lower()
    return safe if safe else "flowchart"


def _node_attrs(node_type: str) -> dict:
    if node_type in {"start", "end"}:
        return {"shape": "oval", "fillcolor": COLOR_DARK_BLUE, "fontcolor": COLOR_WHITE, "penwidth": "2.5"}
    if node_type == "decision":
        return {"shape": "diamond", "style": "filled", "fillcolor": COLOR_WHITE, "margin": "0.18,0.10"}
    if node_type == "input_output":
        return {"shape": "parallelogram", "style": "filled", "fillcolor": COLOR_WHITE}
    return {"shape": "box", "style": "rounded,filled", "fillcolor": COLOR_WHITE}


def _cluster_name(index: int) -> str:
    return f"cluster_phase_{index}"


def build_graphviz(flowchart_data, title: str | None = None, start_node_id: str | None = None, output_format: str = "pdf") -> Digraph:
    if output_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {output_format}")

    flowchart = build_layout_plan(ensure_flowchart(flowchart_data))
    graph_dict = flowchart.to_dict()
    dot = Digraph(comment="Generated Flowchart", format=output_format)
    dot.attr(
        rankdir="TB",
        margin="0.55,0.45",
        ranksep="0.62 equally",
        nodesep="0.46",
        splines="spline",
        compound="true",
        newrank="true",
        concentrate="false",
        outputorder="edgesfirst",
        bgcolor="transparent",
    )
    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor=COLOR_WHITE,
        color=COLOR_DARK_BLUE,
        fontcolor=COLOR_BLACK,
        fontname="Helvetica",
        fontsize="11",
        margin="0.17,0.10",
        penwidth="1.25",
    )
    dot.attr(
        "edge",
        fontname="Helvetica",
        fontsize="9.5",
        fontcolor=COLOR_DARK_BLUE,
        color=COLOR_DARK_BLUE,
        arrowsize="0.76",
    )

    if not start_node_id:
        start_node_id = flowchart.start_node_id or compute_start_node(graph_dict)

    if title:
        dot.attr(
            label=f"\n{title}\n\n",
            labelloc="t",
            fontsize="24",
            fontname="Helvetica-Bold",
            fontcolor=COLOR_DARK_BLUE,
        )

    special_nodes = [node for node in flowchart.nodes if node.type in {"start", "end"}]
    for node in special_nodes:
        dot.node(node.id, node.label, **_node_attrs(node.type))

    phase_nodes: Dict[str, list] = {}
    phase_cluster: Dict[str, str] = {}
    for node in flowchart.nodes:
        if node.type in {"start", "end"}:
            continue
        phase_nodes.setdefault(node.phase or "Workflow", []).append(node)

    phase_heads = []
    for index, phase in enumerate(flowchart.phases or phase_nodes.keys(), start=1):
        nodes = phase_nodes.get(phase, [])
        if not nodes:
            continue
        cluster_name = _cluster_name(index)
        phase_cluster[phase] = cluster_name
        with dot.subgraph(name=cluster_name) as cluster:
            cluster.attr(
                label=phase,
                labelloc="t",
                labeljust="l",
                fontsize="13",
                fontname="Helvetica-Bold",
                fontcolor=COLOR_DARK_BLUE,
                color="#B9C7D8",
                penwidth="1.0",
                style="rounded",
                margin="16",
            )
            for node in nodes:
                cluster.node(node.id, node.label, **_node_attrs(node.type))
        # The earliest-depth node acts as a phase anchor for deterministic order.
        head = min(nodes, key=lambda node: node.metadata.get("layout", {}).get("depth", 10_000))
        phase_heads.append(head.id)

    # Invisible low-impact scaffold preserves phase order without replacing the
    # true process edges. It is especially useful when retry loops exist.
    for left, right in zip(phase_heads, phase_heads[1:]):
        dot.edge(left, right, style="invis", weight="0.15", minlen="1", constraint="true")

    node_map = flowchart.node_by_id()
    for edge in flowchart.edges:
        layout = edge.metadata.get("layout", {}) if edge.metadata else {}
        role = layout.get("role", "forward")
        route = layout.get("route", "main")
        source = node_map.get(edge.source)
        target = node_map.get(edge.target)

        attrs = {"label": edge.label}
        if role == "retry":
            attrs.update(
                style="dashed",
                color="#5B7390",
                fontcolor="#415A77",
                penwidth="1.05",
                constraint="false",
                minlen="2",
                weight="0.1",
            )
            if route == "outer_left":
                attrs.update(tailport="w", headport="w")
            else:
                attrs.update(tailport="e", headport="e")
        elif role == "alternate":
            attrs.update(penwidth="1.35", weight="2", minlen="1")
            preferred = layout.get("preferred_side")
            if preferred == "right":
                attrs.update(tailport="e")
            elif preferred == "left":
                attrs.update(tailport="w")
        else:
            attrs.update(penwidth="2.25", weight="6", minlen="1")

        dot.edge(edge.source, edge.target, **attrs)

    return dot


def render_flowchart(flowchart_data, output_base: str, title: str = None, start_node_id: str = None, output_format: str = "pdf") -> str:
    """Render a canonical workflow to PDF, SVG, or PNG and return the file path."""
    dot = build_graphviz(flowchart_data, title=title, start_node_id=start_node_id, output_format=output_format)
    return dot.render(output_base, cleanup=True)


def render_pdf(flowchart_data, output_base: str, title: str = None, start_node_id: str = None) -> str:
    return render_flowchart(flowchart_data, output_base, title, start_node_id, "pdf")


def render_svg(flowchart_data, output_base: str, title: str = None, start_node_id: str = None) -> str:
    return render_flowchart(flowchart_data, output_base, title, start_node_id, "svg")


def render_png(flowchart_data, output_base: str, title: str = None, start_node_id: str = None) -> str:
    return render_flowchart(flowchart_data, output_base, title, start_node_id, "png")
