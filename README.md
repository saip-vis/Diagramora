# Flowchart App

Voice-to-flowchart MVP with a canonical workflow model, AI reasoning, deterministic normalization/linting, phase-aware layout planning, live Mermaid preview, and vector exports.

## Part 1 architecture

```text
Transcript / speech
        ↓
AI workflow extraction
        ↓
Canonical Flowchart model
        ↓
Normalizer + linter + repair
        ↓
Layout Engine v2
        ↓
Live Mermaid preview / Graphviz PDF, SVG, PNG
```

## Main modules

- `app.py` — Flask routes
- `ai_service.py` — transcription, generation, finalization, repair, titles
- `graph_model.py` — canonical `Flowchart`, `Node`, and `Edge` objects
- `graph_processing.py` — normalization, deduplication, linting, reachability
- `layout_engine.py` — phase order, depth, actor lanes, branch and retry routing hints
- `renderer.py` — shared Graphviz renderer for PDF, SVG, and PNG
- `prompts.py` — shared AI prompt fragments
- `index.html` — recording UI and Mermaid live preview
- `test_part1.py` — offline Part 1 smoke tests

## Run

```bash
export OPENAI_API_KEY="your-key"
pip3 install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:5000`.

## Test the non-AI foundation

```bash
python3 test_part1.py
python3 -m unittest -v test_security.py
python3 -m py_compile *.py
```

## Export API

The existing `/export_pdf` route remains available. The renderer also supports:

- `POST /export/pdf`
- `POST /export/svg`
- `POST /export/png`

All export formats consume the same canonical workflow and shared layout plan.

## Part 2 editor MVP

After a recording is finalized, the editor opens below the diagram. You can:

- click a node to edit its text, shape, fill, border, font size, and scale;
- click a connection to edit its label, color, thickness, pattern, and arrowhead;
- add or delete nodes and connections;
- undo and redo edits;
- enter a natural-language AI edit such as “Add legal review after HR approval”;
- click **Finish editing** to rebuild the downloadable PDF from the edited SVG.

Editor styles are stored in each node or edge's canonical `metadata.editor_style` object, so they remain separate from workflow logic and are ready for future persistence.

## Editor and account system

The live editor now supports node text, shape, fill, border, font, size, and border thickness. Connections support labels, color, thickness, solid/dashed/dotted patterns, straight/smooth/elbow routing, and arrowheads. Changes are rendered in the live diagram with a short debounce for faster interaction.

The title can be renamed from the editor toolbar. The Download dialog supports PDF, PNG, and SVG, plus page size, orientation, scale, and margin controls.

Accounts use Supabase Auth. Saved designs, profiles, and manual version snapshots are stored in Supabase Postgres and protected by Row Level Security plus explicit user filters in the Flask API.

## Supabase authentication and saved designs

This build uses Supabase Auth and Postgres instead of the temporary local `host` account and `designs.json`. See `SUPABASE_SETUP.md` and run `supabase_schema.sql` in Supabase's SQL Editor.

## User platform features

The current build includes Supabase-backed signup/login/logout, password reset email requests, profile editing, saved designs, autosave for opened designs, design rename/duplicate/delete, and manual version snapshots.

During private testing, each account can save up to three designs. The database enforces this atomically; updates to an existing saved design do not consume another slot.

AI-backed routes are protected by per-operation throttles and a database-backed daily quota. The default is 200 AI calls per account per day; set `DAILY_AI_LIMIT` between 10 and 500 to tune it. The latest `supabase_schema.sql` must be installed for quota enforcement and atomic design/version operations.

Password recovery redirects to `/?reset=1`, reads the short-lived Supabase recovery session from the URL, removes it from browser history, and completes the password update through Flask. Add the local and eventual deployed reset URLs to Supabase Authentication → URL Configuration → Redirect URLs.

Focused natural-language editor changes use `gpt-4.1-nano` by default for lower latency. Override `AI_EDIT_MODEL` to evaluate a different quality/latency tradeoff without changing application code.

After updating from an earlier build, run the complete `supabase_schema.sql` in Supabase SQL Editor so the `design_versions` table and policies are created.
