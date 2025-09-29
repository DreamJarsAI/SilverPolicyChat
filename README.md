# School Policy Assistant
A Gradio-powered chatbot that answers students’ questions about school policies using OpenAI’s Agents SDK, GPT-5-nano, and a Postgres/pgvector knowledge base.

## Features
- **OpenAI Agents SDK** orchestrates GPT-5-nano with structured tool calls for grounded, citation-backed answers.
- **OpenAI `text-embedding-large` embeddings** stored in Postgres with `pgvector` for scalable similarity search.
- **AI-assisted intent detection** routes greetings and catalog questions directly to a policy inventory.
- **PDF ingestion pipeline** cleans headers/footers and preserves table content via `pdfplumber` with sentence-aware chunking.
- **Render-ready deployment** using environment-driven configuration and a Postgres backing store.

## Requirements
- Python 3.10+
- Postgres 14+ with the `vector` extension enabled
- OpenAI API access (models: `gpt-5-nano`, `text-embedding-large`)

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
   - Set a database URL: production uses `DATABASE_URL` (Postgres), while local testing defaults to an on-disk SQLite database at `app/policy_vectors.db` but can be overridden with `SQLALCHEMY_DATABASE_URL`
3. **Prepare Postgres**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. **Add policy PDFs** to `policies/` (filenames become document titles).
5. **Embed and store policies**
   ```bash
   python build_index.py --rebuild
   ```
   Re-run after any policy changes to refresh embeddings in Postgres.

### Version Control Notes
- `.gitignore` excludes secrets, virtual environments, and the local SQLite cache (`policy_vectors.db*`) so the repository is safe to push to GitHub.
- Keep `.env` and any credential files out of commits; use Render/hosting dashboards to configure production secrets.
- Commit the contents of `policies/` only if the PDFs are meant for distribution; otherwise add them to `.gitignore` or store them elsewhere.

## Running Locally
```bash
python app.py
```
Visit `http://localhost:7860` to chat. Greetings or catalog questions yield the indexed policy list; substantive questions trigger the OpenAI agent and return cited excerpts.

## Deploying to Render.com
1. Provision Postgres with `pgvector` and capture the `DATABASE_URL`.
2. Configure environment variables (`OPENAI_API_KEY`, optional organization/project, `DATABASE_URL`).
3. Run `python build_index.py --rebuild` in a Render job to seed embeddings.
4. Launch the web service with `python app.py`.

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
