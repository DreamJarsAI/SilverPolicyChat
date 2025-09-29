# Changelog
All notable changes to this project are documented here. This project follows semantic versioning where practical.

## [Unreleased]
### Added
- OpenAI Agents SDK integration (`policy_agent.py`) with retrieval and catalog tools.
- Postgres/pgvector persistence layer (`policy_store.py`) and configuration helper (`config.py`).
- Ingestion CLI (`build_index.py`) that embeds chunks with OpenAI `text-embedding-large` and writes them to Postgres.

### Changed
- Default response model is `gpt-5-nano`; embeddings now call OpenAI directly instead of local models.
- Gradio front-end runs conversations through the OpenAI agent runner and surfaces catalog responses for greetings.
- Documentation refreshed for the OpenAI + Postgres deployment path (Render notes, env vars, testing guidance).
- Configuration now accepts `SQLALCHEMY_DATABASE_URL`/`SQLALCHEMY_DATABASE_URI` as a local testing fallback for `DATABASE_URL`.
- Policy persistence now uses SQLAlchemy with a default on-disk SQLite database (`app/policy_vectors.db`) for local workflows.
- Repository hygiene tightened for GitHub (expanded `.gitignore`, documentation callouts).
- Added Codex Cloud task definition (`codex.yaml`) and documentation for launching the hosted Gradio app.

### Removed
- Local ChromaDB vector store and sentence-transformer dependency.

## [0.2.0] - 2025-03-18
### Added
- Gradio chatbot entry point (`app.py`) with conversational memory and citation reporting.
- PDF ingestion and cleaning utilities (`policy_rag.py`) that strip headers/footers, chunk text, and build a ChromaDB index using sentence-transformer embeddings.
- Deployment-ready documentation, including local setup and Hugging Face Spaces guidance.

### Changed
- Updated requirements to include Gradio, ChromaDB, Sentence Transformers, and Hugging Face Hub packages.

### Fixed
- Warning messaging when no policy PDFs are present, preventing silent failures at startup.

## [0.1.0] - 2025-02-10
### Added
- Placeholder documentation files to scaffold project structure.
