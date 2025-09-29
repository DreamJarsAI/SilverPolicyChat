# Evaluation Plan

Validation guidance for the OpenAI Agents–powered School Policy Assistant.

## Test Matrix
| Area | Goal | Suggested Approach |
| --- | --- | --- |
| PDF ingestion | Verify headers/footers are stripped and tables preserved | Unit-test helpers in `policy_processing.py`; spot-check chunk output for fixture PDFs |
| Postgres embeddings | Ensure chunks embed and persist correctly | Integration test that runs `build_index` against a temporary pgvector database and asserts row counts |
| Retrieval quality | Confirm similarity search returns relevant passages | Mock embeddings and assert `PolicyStore.similar_chunks` ordering; manual checks against known Q&A |
| Agent responses | Produce grounded, cited answers | Stub OpenAI API responses (Agents + embeddings) and assert tool usage + citation format |
| Gradio UI | Validate conversation flow and catalog responses | Launch app locally, test greetings, follow-ups, and error states |

## Smoke Tests
1. **Database Connectivity**
   - Start the app with a valid `DATABASE_URL` but empty tables.
   - Expected: the app warns that no policies are indexed until ingestion runs.
2. **Embedding Ingestion**
   - Run `python build_index.py --rebuild` against sample PDFs.
   - Expected: logs report chunk and document counts; Postgres tables populated.
3. **Agent Tool Invocation**
   - Ask a policy question after ingestion.
   - Expected: the agent calls `retrieve_policy_context`, cites the correct policy/page, and avoids unsupported claims.
4. **Catalog Request**
   - Provide a greeting (“Hi”) or “How many documents do you have?”.
   - Expected: bot lists all indexed policy filenames without invoking retrieval.

## Automated Testing Hooks
```bash
pytest -q
# Example focused run once tests exist
pytest tests/test_policy_store.py -q
```
Tests should stub OpenAI network calls (embeddings, agent responses) and use a disposable Postgres database (e.g., `pytest-postgresql` or Docker) to validate pgvector queries.

## Manual QA Checklist
- [ ] `build_index.py --rebuild` succeeds against target Postgres instance.
- [ ] The Gradio UI loads at `http://localhost:7860` and lists policies on greeting.
- [ ] Policy questions receive cited answers referencing policy title + page.
- [ ] Clearing the chat resets both UI and server-side history.
- [ ] Render deployment instructions followed end-to-end (env vars, pgvector, ingestion job).

## Acceptance Criteria
A release is ready when:
- Automated tests (with OpenAI stubs) pass locally or in CI.
- Manual smoke tests for ingestion, retrieval, and UI behaviours succeed.
- Documentation (`README.md`, `SETUP.md`) reflects the OpenAI Agents + Postgres workflow.
