# School Policy Assistant — Agent Guide

These directions apply to contributors and automation inside `app/`. The assistant now relies on OpenAI’s Agents SDK with GPT-5-nano and a Postgres/pgvector backing store.

## Core Principles
- Every response must be grounded in the retrieved policy chunks; cite the policy title and page number for each claim.
- Treat OpenAI credentials and Postgres URLs as secrets—load them from environment variables, never hard-code.
- Preserve conversational quality while avoiding storage of transcripts outside the active Gradio session.

## Agent Roles

### ManagerAgent
- Confirm that Postgres is reachable and populated before claiming the bot can answer questions.
- Surface the OpenAI models in use (`gpt-5-nano`, `text-embedding-large`) and the current deployment target (e.g., Render).

### DevAgent (Agents & UI)
- Maintain ingestion (`policy_processing.py`, `build_index.py`), the Postgres layer (`policy_store.py`), agent wiring (`policy_agent.py`), and the Gradio UI (`app.py`).
- Default models: `gpt-5-nano` for responses, `text-embedding-large` for embeddings.
- Guardrails:
  - Tools must stay registered via the OpenAI Agents SDK.
  - Retrieval queries must go through Postgres similarity search (`pgvector`).
  - Chunking remains sentence-aware with configurable overlap; table text should be preserved.

### DataPrepAgent
- Keep PDFs in `policies/` with descriptive filenames.
- After any document changes, run `python build_index.py --rebuild` to refresh embeddings in Postgres.
- Record ingestion changes in `CHANGELOG.md`.

### QADocsAgent
- Provide regression checks for ingestion (mocking OpenAI when offline) and agent responses.
- Keep documentation (`README.md`, `SETUP.md`, `EVAL.md`, `TODO.md`, `CHANGELOG.md`) aligned with the OpenAI + Postgres architecture.

### SecOpsAgent
- Monitor dependency and model updates (OpenAI SDK, Agents SDK, Postgres drivers).
- Ensure Render.com (or other deployment targets) store secrets via environment variables.
- Verify Postgres has `vector` extension enabled before deploying.

## Default Checks
```bash
ruff check .
black --check .
isort --check-only .
mypy .
pytest -q
```

## Instruction Priority
- Current user instructions override this file.
- If another `AGENTS.md` exists deeper in the tree, its guidance wins for files in that path.
