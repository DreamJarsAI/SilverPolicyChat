# School Policy Assistant
A Gradio-powered chatbot that answers questions about school policies using retrieval-augmented generation (RAG) over local PDF documents.

## Features
- Vector search over PDF policies using the `intfloat/e5-base-v2` embedding model.
- PDF parsing via `pdfplumber`, with table rows flattened into text for retrieval.
- Automatic cleaning of headers, footers, and page numbers before indexing.
- Sentence-aware chunking with word-overlap to preserve context across responses.
- Citation-backed responses that list the policy sources used.
- Conversational memory so students can ask follow-up questions without repeating context.
- Ready for Hugging Face Spaces deployment.

## Requirements
- Python 3.10+
- `pip install -r requirements.txt`
- Policy PDFs stored in `policies/`
- Hugging Face API token (for the text generation model)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Place your policy PDFs in `policies/` (filenames become document IDs; metadata titles are used when present).
4. Duplicate `.env.example` to `.env` and set your Hugging Face token. The app automatically loads `HF_API_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) on startup.
5. Build the local vector store so the app can reuse it at runtime:
   ```bash
   python build_index.py --rebuild
   ```
   The Chroma files are written to `vector_store/`; they are git-ignored by default, so rebuild them after cloning with `python build_index.py --rebuild`.

## Running Locally
```bash
python app.py
```
The Gradio UI launches at `http://localhost:7860`. Ask a policy question to start the conversation. Use the clear button to reset memory.
If you need to re-embed documents on the fly, run `FORCE_REBUILD_INDEX=1 python app.py` so the app rebuilds the Chroma store before starting.

## Deploying to Hugging Face Spaces
- Upload the entire `app/` folder (including `policies/`, `policy_rag.py`, `app.py`, and `requirements.txt`).
- Set the Space SDK to “Gradio” and the Python version to 3.10 or later.
- Configure a secret named `HF_API_TOKEN` with access to `Qwen/Qwen2.5-7B-Instruct` (or adjust `generator_model` in `app.py` to a model your token can reach).
- When preparing a deployment (e.g., to Hugging Face Spaces), generate the embeddings with `python build_index.py --rebuild` and upload the resulting `vector_store/` artifacts manually if the host expects them.

## Architecture Notes
- `policy_rag.py` handles PDF ingestion, cleaning, chunking, embedding, and ChromaDB indexing.
- Embeddings default to `intfloat/e5-base-v2`, a lightweight model that balances retrieval quality with quick CPU startup (override `embed_model` in `PolicyChatbot` if needed).
- Document titles in the vector metadata mirror the underlying PDF filenames, making citations easy to trace back to the original files.
- `app.py` instantiates the vector store, performs retrieval, formats prompts, calls the Hugging Face Inference API, and serves the Gradio UI.
- Responses use `Qwen/Qwen2.5-7B-Instruct` via the Hugging Face Inference API by default, and you can swap to any other hosted chat model through the `generator_model` parameter.
- Responses append a “Sources” block listing every cited policy title and page number so answers remain grounded.
- Conversation memory feeds the last few user questions back into retrieval, enabling coherent follow-up exchanges.
- The vector store persists under `vector_store/`; rebuild it locally as needed so retrieval stays warm on first request.
