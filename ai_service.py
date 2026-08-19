"""OpenAI-backed transcription, graph generation, repair, and title generation."""

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from openai import OpenAI

from graph_model import Flowchart
from layout_engine import build_layout_plan
from graph_processing import (
    lint_graph,
    merge_similar_nodes,
    normalize_final_graph,
    prune_unreachable_nodes,
)
from prompts import CONTENT_POLICY, GESTURE_DEFINITIONS

client = OpenAI(timeout=45.0, max_retries=1)
AI_EDIT_MODEL = os.getenv("AI_EDIT_MODEL", "gpt-4.1-nano").strip() or "gpt-4.1-nano"
_usage_collector: ContextVar[list | None] = ContextVar("ai_usage_collector", default=None)


@contextmanager
def collect_ai_usage():
    """Collect per-request model usage without leaking data across Flask threads."""
    records = []
    token = _usage_collector.set(records)
    try:
        yield records
    finally:
        _usage_collector.reset(token)


def _capture_usage(response, requested_model: str) -> None:
    records = _usage_collector.get()
    usage = getattr(response, "usage", None)
    if records is None or usage is None:
        return
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    records.append({
        "model": getattr(response, "model", None) or requested_model,
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "request_id": getattr(response, "_request_id", None),
    })

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcript.text

def moderate_text(text: str) -> bool:
    """Returns True if the text is flagged by OpenAI's moderation endpoint."""
    if not text or not text.strip():
        return False
    try:
        result = client.moderations.create(input=text)
        return bool(result.results[0].flagged)
    except Exception:
        logging.exception("OpenAI moderation check failed")
        raise RuntimeError("Content safety check is temporarily unavailable.")

def extract_flowchart_json(transcript: str, gestures: list) -> dict:
    gesture_text = "None detected."
    if gestures:
        gesture_lines = [f"- At {g['time']}s: {g['gesture']}" for g in gestures]
        gesture_text = "\n".join(gesture_lines)

    prompt = f"""Extract a flowchart from this transcript. The speaker also made hand gestures while talking, listed below with timestamps.

{GESTURE_DEFINITIONS}

{CONTENT_POLICY}

Use gesture timing relative to when things are said in the transcript to help resolve ambiguous connections, but treat the spoken content as the primary source of truth. Ignore gestures that don't clearly correspond to anything said nearby.

For each edge, include "primary": true if it represents the main/expected/successful path through the process, or "primary": false if it represents a secondary path (an error branch, a rejection path, an aside, or an alternate route). Default to true for the normal sequential flow.

Return ONLY valid JSON, no other text, in this exact format:

{{
  "nodes": [
    {{"id": "1", "label": "short step description"}}
  ],
  "edges": [
    {{"from": "1", "to": "2", "label": "optional condition, e.g. 'if email exists'", "primary": true}}
  ]
}}

Keep node labels short (3-6 words).

Transcript:
{transcript}

Gestures detected during the explanation:
{gesture_text}"""

    model = "gpt-4o-mini"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _capture_usage(response, model)
    return Flowchart.from_dict(json.loads(response.choices[0].message.content.strip())).to_dict()

def update_graph_live(transcript: str, previous_graph: dict, gestures: list) -> dict:
    """Incrementally update a flowchart graph as the transcript grows. Asks the model
    for ONLY the new/updated nodes/edges since the last call (not the whole graph again),
    then merges that diff into the existing graph server-side."""
    gesture_text = "None detected."
    if gestures:
        gesture_lines = [f"- At {g['time']}s: {g['gesture']}" for g in gestures]
        gesture_text = "\n".join(gesture_lines)

    existing_ids = [node["id"] for node in previous_graph.get("nodes", [])]

    prompt = f"""You are incrementally building a flowchart from a growing transcript of someone explaining a process out loud, in real time. You will be called repeatedly as more speech becomes available.

Below is the FULL transcript so far, the graph that already exists from previous calls (for context only — do not repeat it), and any hand gestures detected so far.

{GESTURE_DEFINITIONS}

{CONTENT_POLICY}

CRITICAL RULES:
- Only output NODES AND EDGES THAT ARE NEW since the existing graph below. Do not repeat any node or edge that already exists.
- Existing node IDs already in use: {existing_ids if existing_ids else "none yet"}. Any new node you create must use a new ID that doesn't collide with these.
- If a new edge connects a NEW node to an EXISTING node, use the existing node's ID for that side of the edge.
- For each new edge, include "primary": true if it represents the main/expected/successful path through the process, or "primary": false for a secondary path (error branch, rejection, aside, alternate route). Default to true for the normal sequential flow.
- The transcript below may be incomplete or still in progress (the speaker may not have finished their sentence yet). SPEED AND LIVENESS ARE THE PRIORITY HERE: if a step is reasonably clear even from a partial or in-progress sentence, go ahead and create a node for it now rather than waiting for the sentence to finish. It's fine if this is occasionally wrong or gets revised later — there is a final cleanup pass after recording stops. Only hold off if there's truly nothing identifiable yet (e.g. just "so then the").
- HOWEVER: every node label must be directly traceable to specific words the speaker actually said. NEVER invent generic, placeholder, or "obviously implied" process steps that were not actually spoken — for example, do not create nodes like "Start the process," "Gather information," "Analyze the data," "Identify key issues," or "Develop solutions" unless the speaker used those specific words. If the transcript so far is empty, only filler words, or too vague/generic to point to a concrete real step, the correct behavior is to create NOTHING yet — an empty result is always better than a fabricated one.
- A valid node must describe a concrete ACTION or STEP in the process — something that happens or gets done (e.g. "uploads a file," "checks the file format," "sends a notification"). NEVER create a node from a bare sentence fragment, opener, or subject-only phrase that hasn't yet stated an action — for example "here's," "our video," "upload the video" (as a title/heading rather than an instruction), "the Creator," or "first" are NOT valid nodes on their own.
- NEVER create a node from narrative/meta commentary about the explanation itself, regardless of how it's phrased — this includes intro phrases like "here's how," "here's what," "so basically," "let's go over," "okay so," "the way this works is," and similar framing language.
- This also applies to PROCESS-LEVEL SUMMARY statements, not just intro phrases — for example "our video upload pipeline works," "this system handles uploads," or any sentence describing what the overall system/process does in general, is NOT a step and must never become a node, even though it may contain a verb.
- NEVER create a node from a single standalone transitional/ordering word alone, such as "first," "then," "next," "after that," or "finally," even if it appears as its own chunk of speech.
- If a step you're about to create is conceptually the same as, or very similar to, an existing node in the graph below, do NOT create a near-duplicate node. This applies EVEN IF the wording is a paraphrase rather than a more complete version. Judge sameness by the underlying real-world action being described, not by exact wording match. Put refinements in "updated_nodes" using the existing ID. Only use "new_nodes" for genuinely new, distinct actions.
- Use gesture timing to help resolve ambiguous connections, but treat spoken content as the primary source of truth.
- Gestures are META-INSTRUCTIONS about HOW to structure the graph. They are NEVER content to include in the graph itself. NEVER create a node or edge label using a gesture's name.

Return ONLY valid JSON, no other text, in this exact format:

{{
  "new_nodes": [
    {{"id": "some_new_id", "label": "short step description"}}
  ],
  "updated_nodes": [
    {{"id": "existing_id", "label": "refined, more complete label for this same step"}}
  ],
  "new_edges": [
    {{"from": "some_id", "to": "some_id", "label": "optional condition", "primary": true}}
  ]
}}

Existing graph (context only, do not repeat):
{json.dumps(previous_graph)}

Full transcript so far:
{transcript}

Gestures detected so far:
{gesture_text}"""

    model = "gpt-4.1-nano"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _capture_usage(response, model)
    diff = json.loads(response.choices[0].message.content.strip())

    merged_nodes = list(previous_graph.get("nodes", []))
    existing_id_set = {n["id"] for n in merged_nodes}
    for node in diff.get("new_nodes", []):
        if node["id"] not in existing_id_set:
            merged_nodes.append(node)
            existing_id_set.add(node["id"])

    for update in diff.get("updated_nodes", []):
        for node in merged_nodes:
            if node["id"] == update["id"]:
                node["label"] = update["label"]
                break

    merged_edges = list(previous_graph.get("edges", []))
    existing_edge_set = {(e["from"], e["to"]) for e in merged_edges}
    for edge in diff.get("new_edges", []):
        source = str(edge.get("from", "")).strip()
        target = str(edge.get("to", "")).strip()
        # Mermaid renders undeclared edge endpoints as ID-shaped placeholder
        # nodes (for example "node5"). Ignore malformed incremental edges until
        # the model supplies both real nodes in a later update.
        if source not in existing_id_set or target not in existing_id_set or source == target:
            continue
        key = (source, target)
        if key not in existing_edge_set:
            merged_edges.append({**edge, "from": source, "to": target})
            existing_edge_set.add(key)

    result, _ = merge_similar_nodes({"nodes": merged_nodes, "edges": merged_edges})
    return Flowchart.from_dict(result).to_dict()

def repair_graph_with_lint(graph: dict, transcript: str, warnings: list) -> dict:
    """Ask the reasoning model to repair only the concrete lint failures."""
    if not warnings:
        return graph
    prompt = f"""You are repairing a process flowchart that failed structural validation.

Use the transcript as the source of truth. Correct the graph without inventing business actions.

REPAIR REQUIREMENTS:
- Organize the workflow into chronological phases before connecting nodes.
- Every decision must have exactly two outgoing branches with distinct, meaningful labels.
- A decision may not directly control unrelated later phases.
- Error or retry branches must return to the exact corrective step.
- Successful branches should merge at the first shared next step.
- Primary edges may not move backward to an earlier phase.
- Keep exactly one Start and one End.
- Preserve every important explicit transcript step.
- Keep optional node fields phase and phase_order. phase_order starts at 1 and increases chronologically.
- Return ONLY valid JSON in the same schema as the supplied graph.

LINT FAILURES:
{chr(10).join('- ' + warning for warning in warnings)}

CURRENT GRAPH:
{json.dumps(graph)}

TRANSCRIPT:
{transcript}
"""
    model = "gpt-4o"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _capture_usage(response, model)
    repaired = Flowchart.from_dict(json.loads(response.choices[0].message.content.strip())).to_dict()
    repaired, _ = merge_similar_nodes(repaired)
    return normalize_final_graph(repaired)

def finalize_graph(graph: dict, transcript: str, gestures: list) -> dict:
    """Phase-aware reasoning followed by deterministic linting and one repair pass."""
    if not transcript.strip() and not graph.get("nodes"):
        return {"nodes": [], "edges": []}

    graph, _ = merge_similar_nodes(graph)
    gesture_text = "None detected."
    if gestures:
        gesture_text = "\n".join(f"- At {g['time']}s: {g['gesture']}" for g in gestures)

    prompt = f"""You are a senior business-process analyst. Reconstruct the logically correct final flowchart from the complete transcript and the rough live graph.

{GESTURE_DEFINITIONS}

{CONTENT_POLICY}

Before producing JSON, reason privately in this order:
1. Divide the workflow into logical chronological phases.
2. Order those phases from earliest to latest.
3. Build the local steps and decisions inside each phase.
4. Connect phases only at legitimate handoff points.
5. Validate all branches, merges, and retry loops.
6. Compare the finished node list against every explicit action and condition in the transcript.

MANDATORY RULES:
- Preserve chronological and causal order from the transcript.
- Include every important explicit step. Do not silently collapse distinct steps.
- Treat conditional language such as "if", "otherwise", "whether", "approved", "complete", "valid", or "successful" as a decision when it controls what happens next. Represent the check itself as a decision diamond rather than placing branch labels on an ordinary process node.
- Preserve explicit terminal actions that occur before End, including phrases such as "mark complete", "close the request", "notify the customer", or "record the result". End is structural and never replaces a stated business action.
- Before returning JSON, account for every sentence or clause in the full transcript. Each explicit action must map to a node, and each explicit condition must map to a decision plus correctly labeled outgoing edges.
- Merge only true semantic duplicates.
- Classify every node as start, process, decision, input_output, or end.
- Add "phase" and integer "phase_order" to every non-Start/End node.
- Add an "actor" to each non-Start/End node when the responsible participant is explicit or strongly implied (for example Customer, System, Warehouse, Carrier).
- A decision label must be a concise question ending in "?".
- Every decision must have exactly TWO outgoing edges unless the transcript explicitly names more than two mutually exclusive outcomes.
- For ordinary yes/no, valid/invalid, success/failure decisions, use exactly two branches.
- Decision edges must have distinct labels such as Yes/No, Valid/Invalid, Success/Failure, Available/In use.
- A decision must control only its immediate question. Do not connect one decision directly to unrelated later phases.
- Every branch must rejoin, retry the exact corrective step, or terminate at an explicit outcome.
- Error branches must stay contextually correct.
- Use primary=true for the expected successful path and primary=false for error, retry, rejection, or alternate paths.
- Successful alternatives should merge at the earliest shared next step.
- Do not create disconnected nodes.
- Do not invent business actions absent from the transcript. Structural Start and End are allowed.
- Keep labels concise, preferably 3-6 words.

Return ONLY valid JSON in this schema:
{{
  "start_node_id": "start",
  "nodes": [
    {{"id": "start", "label": "Start", "type": "start"}},
    {{"id": "1", "label": "Add items to cart", "type": "process", "phase": "Shopping", "phase_order": 1}},
    {{"id": "2", "label": "Account exists?", "type": "decision", "phase": "Authentication", "phase_order": 2}},
    {{"id": "end", "label": "End", "type": "end"}}
  ],
  "edges": [
    {{"from": "start", "to": "1", "label": "", "primary": true}},
    {{"from": "2", "to": "3", "label": "Yes", "primary": true}},
    {{"from": "2", "to": "4", "label": "No", "primary": false}}
  ]
}}

ROUGH LIVE GRAPH:
{json.dumps(graph)}

FULL TRANSCRIPT:
{transcript}

GESTURES:
{gesture_text}"""

    model = "gpt-4o"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _capture_usage(response, model)
    cleaned = Flowchart.from_dict(json.loads(response.choices[0].message.content.strip())).to_dict()
    cleaned, _ = merge_similar_nodes(cleaned)
    cleaned = normalize_final_graph(cleaned)

    first_warnings = lint_graph(cleaned)
    if first_warnings:
        cleaned = repair_graph_with_lint(cleaned, transcript, first_warnings)

    cleaned = prune_unreachable_nodes(cleaned)
    cleaned = normalize_final_graph(cleaned)
    cleaned["lint_warnings"] = lint_graph(cleaned)
    return build_layout_plan(Flowchart.from_dict(cleaned)).to_dict()

def generate_title(flowchart_data: dict) -> str:
    flowchart = Flowchart.from_dict(flowchart_data)
    node_labels = [node.label for node in flowchart.nodes]
    if not node_labels:
        return "Flowchart"

    prompt = f"""Based on these flowchart step labels, write a short, descriptive title (3-6 words) for what process this flowchart describes. Return ONLY the title text, nothing else — no quotes, no punctuation at the end, no explanation.

Steps:
{chr(10).join('- ' + label for label in node_labels)}"""

    try:
        model = "gpt-4.1-nano"
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        _capture_usage(response, model)
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        return title if title else "Flowchart"
    except Exception:
        return "Flowchart"


def edit_flowchart_with_prompt(graph: dict, instruction: str) -> dict:
    """Apply a natural-language edit to an existing canonical workflow.

    The model returns a complete graph, but is instructed to preserve IDs,
    phases, actors, layout metadata, and user style metadata unless the edit
    explicitly requires a change. The result is converted through Flowchart so
    malformed fields cannot leak into the rest of the application.
    """
    flowchart = Flowchart.from_dict(graph)
    prompt = f"""You are editing an existing flowchart, not regenerating it.

USER INSTRUCTION:
{instruction}

CURRENT FLOWCHART JSON:
{json.dumps(flowchart.to_dict())}

RULES:
- Return the complete updated flowchart as valid JSON only.
- Preserve every existing node ID and edge endpoint whenever possible.
- Preserve phases, actors, metadata, layout hints, and style metadata unless the user explicitly asks to change them.
- Make only the smallest changes needed to satisfy the instruction.
- Do not invent unrelated business steps.
- Node types must be one of: start, process, decision, input_output, end.
- If adding a node, give it a unique stable ID and connect it logically.
- If deleting a node, reconnect its incoming and outgoing path when that is unambiguous.
- Preserve exactly one Start and one End.
- Every decision should remain a concise question with meaningful branch labels.

Return this schema:
{{
  "nodes": [{{"id":"...","label":"...","type":"process","phase":"...","phase_order":1,"actor":"...","metadata":{{}}}}],
  "edges": [{{"from":"...","to":"...","label":"","primary":true,"metadata":{{}}}}],
  "start_node_id":"start",
  "phases":[],
  "actors":[]
}}"""
    response = client.chat.completions.create(
        model=AI_EDIT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    _capture_usage(response, AI_EDIT_MODEL)
    result = json.loads(response.choices[0].message.content.strip())
    return build_layout_plan(Flowchart.from_dict(result)).to_dict()
