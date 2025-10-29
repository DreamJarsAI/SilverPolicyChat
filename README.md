# School Policy Assistant
A Gradio-powered chatbot that answers students’ questions about school policies using OpenAI’s Agents SDK, GPT-5-nano, and a Postgres/pgvector knowledge base.

## Features
- **OpenAI Agents SDK** orchestrates GPT-5-nano with structured tool calls for grounded, citation-backed answers.
- **OpenAI `text-embedding-3-large` embeddings** stored in Postgres with `pgvector` for scalable similarity search.
- **AI-assisted intent detection** routes greetings and catalog questions directly to a policy inventory.
- **PDF ingestion pipeline** cleans headers/footers and preserves table content via `pdfplumber` with sentence-aware chunking.
- **Render-ready deployment** using environment-driven configuration and a Postgres backing store.

## Requirements
- Python 3.10+
- Postgres 14+ with the `vector` extension enabled (for deployment)
- OpenAI API access (models: `gpt-5-nano`, `text-embedding-3-large`)

## Setup
1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment variables**
   - Copy `.env.example` → `.env`
   - Provide `OPENAI_API_KEY` (and optionally `OPENAI_ORGANIZATION` / `OPENAI_PROJECT`)
   - Leave `DATABASE_URL` unset for local development to fall back to a local SQLite file (`policy_vectors.db`). You can override the path via `SQLALCHEMY_DATABASE_URL` if desired, or point either variable to Postgres (`postgresql://...`) when running in Render/production
3. **Prepare Postgres**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. **Add policy PDFs** to `policies/` (filenames become document titles).
5. **Embed and store policies** (run against whichever database is configured)
   ```bash
   python build_index.py --rebuild
   ```
   Re-run after any policy changes to refresh embeddings in Postgres.

### Version Control Notes
- `.gitignore` excludes secrets, virtual environments, and the local SQLite cache (`policy_vectors.db*`). Double-check `git status` before committing to ensure no credential files or generated databases slip in.
- Keep `.env` and any credential files out of commits. For deployment, set environment variables via Render (or your host) rather than storing secrets in the repo.
- Commit the contents of `policies/` only if the PDFs are meant for distribution; otherwise add them to `.gitignore` or store them elsewhere.

### Codex Cloud Task
- The repo includes `codex.yaml`, defining a `web` task that installs dependencies and then runs `python app.py` from the repository root.
- Before launching the task, set the required environment variables (`OPENAI_API_KEY`, optional `OPENAI_ORGANIZATION`/`OPENAI_PROJECT`, and `DATABASE_URL` when targeting Postgres) in the Codex Cloud UI.
- Start the cloud run via the “Run in the cloud” button and select the `web` task; Gradio will be served on port 7860.

## Running Locally
```bash
python app.py
```
Visit `http://localhost:7860` to chat. Greetings or catalog questions yield the indexed policy list; substantive questions trigger the OpenAI agent and return cited excerpts.

## Deploying to Render.com
1. Provision a managed Postgres database, enable the `vector` extension, and note the `DATABASE_URL` Render provides.
2. In Render’s dashboard, set environment variables for the web service (at minimum `OPENAI_API_KEY` and `DATABASE_URL`; include SMTP settings if you’ll email verification codes).
3. Before the first deploy, run a one-off job (or shell) in Render that executes `python build_index.py --rebuild`. This seeds documents + embeddings into Postgres so the web app can query them on startup without rebuilding.
4. Configure the web service to run `python app.py`. On subsequent deploys the app will reuse the existing embeddings stored in Postgres.

## Architecture Notes
- `config.py` centralises settings (OpenAI models, Postgres URL, chunk sizes).
- `policy_processing.py` discovers PDFs and produces cleaned chunks.
- `policy_store.py` persists documents, chunks, and embeddings via SQLAlchemy (SQLite locally, Postgres in production).
- `policy_agent.py` defines the OpenAI Agent with retrieval/catalog tools and intent heuristics.
- `build_index.py` ingests PDFs and stores embeddings via OpenAI’s `text-embedding-large` model.
- `app.py` serves the Gradio UI and proxies requests through the agent runner.

## Data Refresh Checklist
1. Update PDFs in `policies/`.
2. Run `python build_index.py --rebuild`.
3. Restart the app (or redeploy).

## Testing Checklist
- Ensure `.env` contains valid OpenAI and Postgres credentials.
- Run ingestion and confirm chunk counts in the logs.
- Start `python app.py` and verify:
  - Greetings list all indexed policies.
  - Policy questions return cited answers.
  - Clearing the chat resets history.
