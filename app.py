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
    should_list_documents,
    stream_agent,
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

    with gr.Blocks(title="School Policy Assistant", theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
                # School Policy Assistant
                _Grounded answers to policy questions, powered by OpenAI Agents._
                """
            )

        chat = gr.Chatbot(
            label="Policy Assistant",
            type="messages",
            height=520,
            show_copy_button=True,
        )
        history_state = gr.State([])

        with gr.Row():
            message_box = gr.Textbox(
                label="Your question",
                placeholder="Ask about attendance, grading, scholarships, ...",
                scale=4,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        clear_button = gr.Button("Clear conversation", variant="secondary")

        async def respond(
            user_message: str,
            chat_history: List[dict],
            rag_history: List[Tuple[str, str]],
        ):
            rag_history = list(rag_history or [])
            normalized = user_message.strip()
            cleared_input = gr.update(value="")
            if not normalized:
                yield cleared_input, chat_history, rag_history
                return

            updated_chat = list(chat_history or [])
            updated_chat.append({"role": "user", "content": normalized})
            assistant_entry = {"role": "assistant", "content": ""}
            updated_chat.append(assistant_entry)

            if should_list_documents(normalized, context):
                answer = format_catalog_response(context)
                assistant_entry["content"] = answer
                rag_history.append((normalized, answer))
                yield cleared_input, updated_chat, rag_history
                return

            prior_history = list(rag_history)
            rag_history.append((normalized, ""))

            # Show the user turn and placeholder assistant message immediately.
            yield cleared_input, updated_chat, rag_history

            emitted_chunk = False
            async for chunk in stream_agent(agent, runner, context, prior_history, normalized):
                assistant_entry["content"] += chunk
                rag_history[-1] = (normalized, assistant_entry["content"])
                emitted_chunk = True
                yield cleared_input, updated_chat, rag_history

            rag_history[-1] = (normalized, assistant_entry["content"])
            if not emitted_chunk:
                # Ensure the final state is delivered even if no streaming chunks were emitted.
                yield cleared_input, updated_chat, rag_history

            return

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
