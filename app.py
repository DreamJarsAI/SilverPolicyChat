"""Gradio front-end for the OpenAI Agents-based School Policy assistant."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from config import load_settings
from policy_agent import (
    PolicyAgentContext,
    build_agent,
    format_catalog_response,
    run_agent,
    should_list_documents,
)
from policy_store import PolicyStore
from agents.run import AgentRunner as OpenAIAgentRunner

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _bootstrap() -> tuple[PolicyAgentContext, object]:
    settings = load_settings()
    client = OpenAI(
        api_key=settings.openai_api_key,
        organization=settings.openai_organization,
        project=settings.openai_project,
    )
    store = PolicyStore(settings.database_url)
    try:
        store.ensure_schema()  # Ensures tables exist; dimension inferred from metadata.
    except RuntimeError as exc:
        logger.info("Schema check skipped: %s", exc)
    agent = build_agent(settings)
    context = PolicyAgentContext(store=store, client=client, settings=settings)
    return context, agent


def create_interface() -> gr.Blocks:
    context, agent = _bootstrap()
    runner = OpenAIAgentRunner()

    custom_css = """
    :root {
        --silver-indigo: #4b1d9b;
        --silver-violet: #7a3bff;
        --silver-lavender: #c9b6ff;
        --panel-surface: rgba(255, 255, 255, 0.92);
        --panel-border: rgba(75, 29, 155, 0.12);
        --text-primary: #23154b;
        --text-secondary: #4a3b6d;
    }

    body {
        min-height: 100vh;
        background: linear-gradient(140deg, #f7f4ff 0%, #efe8ff 45%, #f9f7ff 100%);
        color: var(--text-primary);
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background:
            radial-gradient(circle at 10% 18%, rgba(122, 59, 255, 0.28) 0%, transparent 55%),
            radial-gradient(circle at 85% 12%, rgba(75, 29, 155, 0.22) 0%, transparent 60%),
            radial-gradient(circle at 48% 85%, rgba(201, 182, 255, 0.3) 0%, transparent 55%);
        z-index: -1;
    }

    .gradio-container {
        background-color: transparent !important;
        max-width: 1180px;
        margin: 0 auto !important;
        padding: 2.75rem 1.75rem 3.5rem !important;
    }

    .hero-block {
        position: relative;
        overflow: hidden;
        border-radius: 1.75rem;
        padding: 2.75rem 3rem;
        margin-bottom: 2.25rem;
        background: linear-gradient(130deg, rgba(75, 29, 155, 0.92) 0%, rgba(122, 59, 255, 0.88) 45%, rgba(201, 182, 255, 0.82) 100%);
        color: #ffffff;
        box-shadow: 0 28px 48px rgba(42, 18, 98, 0.28);
        text-align: left;
    }

    .hero-block::after {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 75% 20%, rgba(255, 255, 255, 0.18) 0%, transparent 55%);
        opacity: 0.75;
    }

    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 640px;
    }

    .hero-content h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.85rem;
        letter-spacing: -0.01em;
    }

    .hero-content p {
        font-size: 1.1rem;
        line-height: 1.55;
        margin: 0;
        color: rgba(255, 255, 255, 0.88);
    }

    .layout-row {
        gap: 1.85rem;
        align-items: stretch;
    }

    .chat-column,
    .info-column {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .chatbot-panel .gradio-chatbot,
    .chatbot-panel .chatbot {
        background: var(--panel-surface);
        border-radius: 1.5rem;
        border: 1px solid var(--panel-border);
        box-shadow: 0 22px 45px rgba(35, 21, 75, 0.16);
    }

    .chatbot-panel .gradio-chatbot .message.user,
    .chatbot-panel .chatbot .message.user {
        background: linear-gradient(135deg, rgba(75, 29, 155, 0.95), rgba(122, 59, 255, 0.9));
        color: #ffffff;
    }

    .chatbot-panel .gradio-chatbot .message.bot,
    .chatbot-panel .chatbot .message.bot {
        background: rgba(246, 242, 255, 0.88);
        border: 1px solid rgba(75, 29, 155, 0.08);
        color: var(--text-primary);
    }

    .input-card {
        background: var(--panel-surface);
        border-radius: 1.5rem;
        padding: 1.75rem;
        border: 1px solid var(--panel-border);
        box-shadow: 0 18px 38px rgba(35, 21, 75, 0.18);
    }

    .input-card label {
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.6rem;
    }

    .question-box textarea {
        min-height: 7rem;
        font-size: 1.05rem;
        line-height: 1.55;
        border-radius: 1rem !important;
        border: 1px solid rgba(75, 29, 155, 0.18) !important;
        box-shadow: inset 0 1px 1px rgba(35, 21, 75, 0.05);
    }

    .question-box textarea:focus-visible {
        outline: 2px solid rgba(122, 59, 255, 0.45);
        box-shadow: 0 0 0 4px rgba(122, 59, 255, 0.12);
    }

    .send-row {
        display: flex;
        gap: 0.75rem;
        justify-content: flex-end;
        margin-top: 1rem;
    }

    .send-row .gr-button-primary {
        background: linear-gradient(135deg, #5f22d9, #7a3bff);
        border: none;
        box-shadow: 0 10px 20px rgba(95, 34, 217, 0.25);
    }

    .send-row .gr-button-secondary {
        border: 1px solid rgba(95, 34, 217, 0.22);
        background: rgba(122, 59, 255, 0.08);
        color: var(--silver-indigo);
    }

    .info-card {
        background: var(--panel-surface);
        border-radius: 1.5rem;
        padding: 2rem;
        border: 1px solid var(--panel-border);
        box-shadow: 0 20px 40px rgba(35, 21, 75, 0.16);
        color: var(--text-primary);
    }

    .info-card h3 {
        font-weight: 650;
        margin-bottom: 1rem;
    }

    .info-card ul {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 0.85rem;
    }

    .info-card li {
        display: grid;
        grid-template-columns: 18px 1fr;
        gap: 0.75rem;
        align-items: start;
        line-height: 1.5;
        color: var(--text-secondary);
    }

    .info-card li::before {
        content: "";
        display: inline-flex;
        width: 10px;
        height: 10px;
        border-radius: 999px;
        margin-top: 0.45rem;
        background: linear-gradient(135deg, rgba(95, 34, 217, 0.95), rgba(201, 182, 255, 0.95));
        box-shadow: 0 0 0 4px rgba(95, 34, 217, 0.12);
    }

    @media (max-width: 1024px) {
        .hero-block {
            padding: 2.25rem 2rem;
            text-align: center;
        }

        .hero-content {
            margin: 0 auto;
        }

        .layout-row {
            flex-direction: column;
        }

        .send-row {
            justify-content: stretch;
        }

        .info-card {
            padding: 1.75rem;
        }
    }
    """

    with gr.Blocks(
        title="NYU Silver Policy Assistant",
        theme=gr.themes.Soft(),
        css=custom_css,
    ) as demo:
        gr.Markdown(
            """
            <div class="hero-block">
                <div class="hero-content">
                    <h1>NYU Silver Policy Assistant</h1>
                    <p>Get grounded guidance on school policy decisions for the NYU Silver community.</p>
                </div>
            </div>
            """,
            elem_classes=["hero-block"],
        )

        with gr.Row(elem_classes=["layout-row"]):
            with gr.Column(scale=3, elem_classes=["chat-column"]):
                chat = gr.Chatbot(
                    label="Policy Assistant",
                    type="messages",
                    height=540,
                    show_copy_button=True,
                    elem_classes=["chatbot-panel"],
                )
                history_state = gr.State([])

                with gr.Group(elem_classes=["input-card"]):
                    message_box = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about attendance, grading, scholarships, ...",
                        lines=4,
                        max_lines=8,
                        elem_classes=["question-box"],
                    )
                    with gr.Row(elem_classes=["send-row"]):
                        send_button = gr.Button("Send", variant="primary", scale=0)
                        clear_button = gr.Button(
                            "Clear conversation",
                            variant="secondary",
                            scale=0,
                        )

            with gr.Column(scale=2, elem_classes=["info-column"]):
                gr.Markdown(
                    """
                    <div class="info-card">
                        <h3>How to get the most out of the assistant</h3>
                        <ul>
                            <li>Frame your scenario with key details like grade level, policy area, and stakeholders.</li>
                            <li>Ask follow-up questions to clarify interpretations or explore alternative actions.</li>
                            <li>Use the Clear button to start a new conversation when switching topics.</li>
                        </ul>
                    </div>
                    """,
                    elem_classes=["info-card"],
                )

        def respond(user_message: str, chat_history: List[dict], rag_history: List[Tuple[str, str]]):
            rag_history = list(rag_history or [])
            normalized = user_message.strip()
            if not normalized:
                return "", chat_history, rag_history

            if should_list_documents(normalized, context):
                answer = format_catalog_response(context)
            else:
                answer = run_agent(agent, runner, context, rag_history, normalized)

            rag_history.append((normalized, answer))
            updated_chat = list(chat_history or [])
            updated_chat.append({"role": "user", "content": normalized})
            updated_chat.append({"role": "assistant", "content": answer})
            return "", updated_chat, rag_history

        for trigger in (message_box.submit, send_button.click):
            trigger(
                respond,
                inputs=[message_box, chat, history_state],
                outputs=[message_box, chat, history_state],
            )

        clear_button.click(lambda: ([], []), outputs=[chat, history_state])

    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
