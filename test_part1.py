"""Fast offline smoke tests for the Part 1 workflow foundation."""

from graph_model import Flowchart
from layout_engine import build_layout_plan
from renderer import build_graphviz


def sample_graph():
    return {
        "start_node_id": "start",
        "nodes": [
            {"id": "start", "label": "Start", "type": "start"},
            {"id": "1", "label": "Submit request", "type": "process", "phase": "Submission", "phase_order": 1, "actor": "Employee"},
            {"id": "2", "label": "Request approved?", "type": "decision", "phase": "Review", "phase_order": 2, "actor": "Manager"},
            {"id": "3", "label": "Revise request", "type": "process", "phase": "Correction", "phase_order": 3, "actor": "Employee"},
            {"id": "4", "label": "Send confirmation", "type": "process", "phase": "Finalization", "phase_order": 4, "actor": "System"},
            {"id": "end", "label": "End", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "1", "label": "", "primary": True},
            {"from": "1", "to": "2", "label": "", "primary": True},
            {"from": "2", "to": "4", "label": "Yes", "primary": True},
            {"from": "2", "to": "3", "label": "No", "primary": False},
            {"from": "3", "to": "1", "label": "Retry", "primary": False},
            {"from": "4", "to": "end", "label": "", "primary": True},
        ],
    }


def run():
    planned = build_layout_plan(Flowchart.from_dict(sample_graph()))
    assert planned.metadata["layout"]["version"] == 2
    assert planned.metadata["layout"]["swimlane_ready"] is True
    assert planned.phases == ["Submission", "Review", "Correction", "Finalization"]
    retry = next(edge for edge in planned.edges if edge.label == "Retry")
    assert retry.metadata["layout"]["role"] == "retry"
    assert retry.metadata["layout"]["route"] in {"outer_left", "outer_right"}
    dot = build_graphviz(planned, title="Smoke Test", output_format="svg")
    source = dot.source
    assert "cluster_phase_1" in source
    assert "style=dashed" in source
    assert "Request approved?" in source
    print("Part 1 smoke tests passed")


if __name__ == "__main__":
    run()
