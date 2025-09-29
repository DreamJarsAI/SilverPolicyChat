# Developer Setup Guide

Use this guide to configure the School Policy Assistant locally.

## Prerequisites
- Python 3.10+
- Postgres 14+ with the `vector` extension (`CREATE EXTENSION IF NOT EXISTS vector;`)
- OpenAI API access for GPT-5 and embeddings

## 1. Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Environment Variables
1. Copy `.env.example` → `.env`.
2. Populate the following:
   - `OPENAI_API_KEY` (required)
   - Optional: `OPENAI_ORGANIZATION`, `OPENAI_PROJECT`
   - `DATABASE_URL` for Render/production deployments; local testing defaults to `sqlite:///app/policy_vectors.db` but you can override it with `SQLALCHEMY_DATABASE_URL` if needed (Postgres example: `postgresql://user:password@localhost:5432/policies`)
3. Additional tuning knobs (`OPENAI_COMPLETION_MODEL`, `CHUNK_SIZE`, etc.) are optional.

> **Git tip:** `.gitignore` already excludes `.env` and the SQLite cache (`policy_vectors.db*`), so you can push the project to GitHub without leaking credentials or large binaries.

## 3. Seed Policy Data
1. Place PDF files under `policies/` (filenames are used as document titles).
2. Run the ingestion script to populate Postgres:
   ```bash
   python build_index.py --rebuild
   ```
   Re-run this command whenever policy PDFs change.

## 4. Launch the App
```bash
python app.py
```
Open `http://localhost:7860` to chat. Greetings or catalog-style questions list the available policies; substantive queries trigger the OpenAI agent and cite retrieved passages.

## 5. Optional Quality Checks
```bash
ruff check .
black --check .
isort --check-only .
pytest -q
```
Add integration tests that stub OpenAI responses for CI environments without network access.

## Render.com Deployment Notes
1. Provision a Postgres instance and enable `pgvector`.
2. Add environment variables (`OPENAI_API_KEY`, `DATABASE_URL`, etc.) in Render’s dashboard.
3. Run `python build_index.py --rebuild` in a Render job to seed embeddings before starting the web service.
4. Configure the web service with `python app.py` as the start command.

## Codex Cloud Notes
- The included `codex.yaml` defines a `web` task that installs dependencies and launches the Gradio app on port 7860 from the repo root.
- Set the same environment variables you use locally (OpenAI keys, database URL) in Codex Cloud before starting the task.

## Troubleshooting
- **Missing vector extension**: Run `CREATE EXTENSION IF NOT EXISTS vector;` on the Postgres instance.
- **Ingestion errors**: Ensure the OpenAI key is valid and Postgres credentials allow write access.
- **Empty answers**: Verify `build_index.py` has been executed and the database contains documents (`SELECT COUNT(*) FROM documents;`).
