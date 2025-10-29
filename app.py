"""Gradio front-end for the OpenAI Agents-based School Policy assistant."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

from auth_service import AuthService, EmailSender, EmailSettings, SecretHasher
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
def _bootstrap() -> tuple[PolicyAgentContext, object, AuthService]:
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
    email_sender = EmailSender(EmailSettings.from_settings(settings))
    auth_service = AuthService(store, email_sender, SecretHasher())
    return context, agent, auth_service


def create_interface() -> gr.Blocks:
    context, agent, auth_service = _bootstrap()
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

    .chat-column {
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

    @media (max-width: 1024px) {
        .hero-block {
            padding: 2.25rem 2rem;
            text-align: center;
        }

        .hero-content {
            margin: 0 auto;
        }

        .send-row {
            justify-content: stretch;
        }
    }

    .disclaimer-text {
        margin-top: 2.5rem;
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.95rem;
    }

    .gradio-container header,
    .gradio-container footer,
    .gradio-container .footer,
    .gradio-container [data-testid="share-btn"],
    .gradio-container [data-testid="share-button"],
    .gradio-container [data-testid="settings-button"],
    .gradio-container button[aria-label="Settings"],
    .gradio-container [aria-label="Use via API"],
    .gradio-container a[href*="gradio.app"],
    .gradio-container a[href*="gradio.space"] {
        display: none !important;
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
                    <p>Answer your questions on school policies and precedures 24/7.</p>
                </div>
            </div>
            """,
            elem_classes=["hero-block"],
        )

        session_state = gr.State({"authenticated": False, "username": None, "email": None})
        history_state = gr.State([])
        registration_state = gr.State({"username": None, "email": None, "code_sent": False})
        reset_state = gr.State({"email": None, "code_sent": False})

        with gr.Row():
            status_bar = gr.Markdown("Not signed in.")
            logout_button = gr.Button("Log out", variant="secondary", visible=False)

        with gr.Tabs() as auth_tabs:
            with gr.Tab("Login"):
                login_username = gr.Textbox(label="Username", placeholder="nyustudent")
                login_password = gr.Textbox(
                    label="Password",
                    type="password",
                    placeholder="Enter your password",
                )
                login_button = gr.Button("Log in", variant="primary")
                login_feedback = gr.Markdown(visible=False)

            with gr.Tab("Register"):
                register_username = gr.Textbox(
                    label="Choose a username",
                    placeholder="nyustudent",
                )
                register_email = gr.Textbox(
                    label="NYU email",
                    placeholder="netid@nyu.edu",
                )
                send_registration_code = gr.Button("Send verification code", variant="primary")
                registration_feedback = gr.Markdown(visible=False)
                register_code = gr.Textbox(
                    label="6-digit verification code",
                    placeholder="Enter the code from your email",
                    visible=False,
                )
                register_password = gr.Textbox(
                    label="Create a strong password",
                    placeholder="At least 12 characters with mixed case, numbers, symbols",
                    type="password",
                    visible=False,
                )
                complete_registration = gr.Button(
                    "Complete registration", variant="primary", visible=False
                )

            with gr.Tab("Forgot username/password"):
                reset_email = gr.Textbox(
                    label="NYU email",
                    placeholder="netid@nyu.edu",
                )
                send_reset_code = gr.Button("Send reset email", variant="primary")
                reset_feedback = gr.Markdown(visible=False)
                reset_code = gr.Textbox(
                    label="6-digit verification code",
                    placeholder="Enter the code from your email",
                    visible=False,
                )
                reset_password = gr.Textbox(
                    label="New password",
                    type="password",
                    placeholder="At least 12 characters with mixed case, numbers, symbols",
                    visible=False,
                )
                complete_reset = gr.Button("Reset password", variant="primary", visible=False)

        with gr.Group(visible=False) as chat_container:
            with gr.Column(elem_classes=["chat-column"]):
                chat = gr.Chatbot(
                    label="Policy Assistant",
                    type="messages",
                    height=540,
                    show_copy_button=True,
                    elem_classes=["chatbot-panel"],
                )

                with gr.Group(elem_classes=["input-card"]):
                    message_box = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about attendance, grading, scholarships, ...",
                        lines=4,
                        max_lines=8,
                        elem_classes=["question-box"],
                        interactive=False,
                    )
                    with gr.Row(elem_classes=["send-row"]):
                        send_button = gr.Button("Send", variant="primary", scale=0, interactive=False)
                        clear_button = gr.Button(
                            "Clear conversation",
                            variant="secondary",
                            scale=0,
                        )

        gr.Markdown(
            "AI can make mistakes and you should check with official documents for most accurate answers",
            elem_classes=["disclaimer-text"],
        )

        def handle_login(
            username: str,
            password: str,
            current_session: Dict[str, Any],
        ) -> tuple[Any, ...]:
            success, message, payload = auth_service.authenticate(username, password)
            if success and payload:
                session = {"authenticated": True, **payload}
                return (
                    gr.update(value=f"✅ {message}", visible=True),
                    session,
                    gr.update(value=f"Signed in as **{payload['username']}**."),
                    gr.update(visible=True),
                    gr.update(visible=False, selected=0),
                    gr.update(visible=True),
                    gr.update(value="", interactive=True),
                    gr.update(interactive=True),
                    gr.update(value=""),
                    gr.update(value=""),
                )

            session = {"authenticated": False, "username": None, "email": None}
            return (
                gr.update(value=f"❌ {message}", visible=True),
                session,
                gr.update(value="Not signed in."),
                gr.update(visible=False),
                gr.update(visible=True, selected=0),
                gr.update(visible=False),
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                gr.update(value=username),
                gr.update(value=""),
            )

        login_button.click(
            handle_login,
            inputs=[login_username, login_password, session_state],
            outputs=[
                login_feedback,
                session_state,
                status_bar,
                logout_button,
                auth_tabs,
                chat_container,
                message_box,
                send_button,
                login_username,
                login_password,
            ],
        )

        def handle_logout(current_session: Dict[str, Any]) -> tuple[Any, ...]:
            _ = current_session
            return (
                gr.update(value="You have been signed out.", visible=True),
                {"authenticated": False, "username": None, "email": None},
                gr.update(value="Not signed in."),
                gr.update(visible=False),
                gr.update(visible=True, selected=0),
                gr.update(visible=False),
                gr.update(value="", interactive=False),
                gr.update(interactive=False),
                [],
                [],
            )

        logout_button.click(
            handle_logout,
            inputs=[session_state],
            outputs=[
                login_feedback,
                session_state,
                status_bar,
                logout_button,
                auth_tabs,
                chat_container,
                message_box,
                send_button,
                chat,
                history_state,
            ],
        )

        def start_registration(
            username: str,
            email: str,
            state: Dict[str, Any],
        ) -> tuple[Any, ...]:
            success, message, payload = auth_service.initiate_registration(username, email)
            if success and payload:
                new_state = {"code_sent": True, **payload}
                return (
                    gr.update(value=f"✅ {message}", visible=True),
                    new_state,
                    gr.update(value="", visible=True),
                    gr.update(value="", visible=True),
                    gr.update(visible=True),
                    gr.update(value=payload["username"]),
                    gr.update(value=payload["email"]),
                )

            keep_state = state or {"code_sent": False, "username": None, "email": None}
            active = bool(keep_state.get("code_sent"))
            return (
                gr.update(value=f"❌ {message}", visible=True),
                keep_state,
                gr.update(value="", visible=active),
                gr.update(value="", visible=active),
                gr.update(visible=active),
                gr.update(value=username),
                gr.update(value=email),
            )

        send_registration_code.click(
            start_registration,
            inputs=[register_username, register_email, registration_state],
            outputs=[
                registration_feedback,
                registration_state,
                register_code,
                register_password,
                complete_registration,
                register_username,
                register_email,
            ],
        )

        def finish_registration(
            code: str,
            password: str,
            state: Dict[str, Any],
        ) -> tuple[Any, ...]:
            success, message = auth_service.complete_registration(state or {}, code, password)
            if success:
                return (
                    gr.update(value="", visible=False),
                    {"username": None, "email": None, "code_sent": False},
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=""),
                    gr.update(value=f"✅ {message}", visible=True),
                    gr.update(visible=True, selected=0),
                )

            keep_state = state or {"username": None, "email": None, "code_sent": False}
            return (
                gr.update(value=f"❌ {message}", visible=True),
                keep_state,
                gr.update(value=code, visible=True),
                gr.update(value="", visible=True),
                gr.update(visible=True),
                gr.update(value=keep_state.get("username") or ""),
                gr.update(value=keep_state.get("email") or ""),
                gr.update(),
                gr.update(visible=True, selected=1),
            )

        complete_registration.click(
            finish_registration,
            inputs=[register_code, register_password, registration_state],
            outputs=[
                registration_feedback,
                registration_state,
                register_code,
                register_password,
                complete_registration,
                register_username,
                register_email,
                login_feedback,
                auth_tabs,
            ],
        )

        def start_reset(email: str, state: Dict[str, Any]) -> tuple[Any, ...]:
            success, message, payload = auth_service.initiate_password_reset(email)
            if success and payload:
                new_state = {"code_sent": True, **payload}
                return (
                    gr.update(value=f"✅ {message}", visible=True),
                    new_state,
                    gr.update(value="", visible=True),
                    gr.update(value="", visible=True),
                    gr.update(visible=True),
                    gr.update(value=payload["email"]),
                )

            keep_state = state or {"email": None, "code_sent": False}
            active = bool(keep_state.get("code_sent"))
            return (
                gr.update(value=f"❌ {message}", visible=True),
                keep_state,
                gr.update(value="", visible=active),
                gr.update(value="", visible=active),
                gr.update(visible=active),
                gr.update(value=email),
            )

        send_reset_code.click(
            start_reset,
            inputs=[reset_email, reset_state],
            outputs=[
                reset_feedback,
                reset_state,
                reset_code,
                reset_password,
                complete_reset,
                reset_email,
            ],
        )

        def finish_reset(
            code: str,
            password: str,
            state: Dict[str, Any],
        ) -> tuple[Any, ...]:
            success, message = auth_service.complete_password_reset(state or {}, code, password)
            if success:
                return (
                    gr.update(value="", visible=False),
                    {"email": None, "code_sent": False},
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=f"✅ {message}", visible=True),
                    gr.update(visible=True, selected=0),
                )

            keep_state = state or {"email": None, "code_sent": False}
            return (
                gr.update(value=f"❌ {message}", visible=True),
                keep_state,
                gr.update(value=code, visible=True),
                gr.update(value="", visible=True),
                gr.update(visible=True),
                gr.update(value=keep_state.get("email") or ""),
                gr.update(),
                gr.update(visible=True, selected=2),
            )

        complete_reset.click(
            finish_reset,
            inputs=[reset_code, reset_password, reset_state],
            outputs=[
                reset_feedback,
                reset_state,
                reset_code,
                reset_password,
                complete_reset,
                reset_email,
                login_feedback,
                auth_tabs,
            ],
        )

        async def respond(
            user_message: str,
            chat_history: List[dict],
            rag_history: List[Tuple[str, str]],
            session: Dict[str, Any],
        ):
            rag_history = list(rag_history or [])
            normalized = user_message.strip()
            cleared_input = gr.update(value="")

            if not session or not session.get("authenticated"):
                yield cleared_input, chat_history, rag_history
                return

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

            yield cleared_input, updated_chat, rag_history

            emitted_chunk = False
            async for chunk in stream_agent(agent, runner, context, prior_history, normalized):
                assistant_entry["content"] += chunk
                rag_history[-1] = (normalized, assistant_entry["content"])
                emitted_chunk = True
                yield cleared_input, updated_chat, rag_history

            rag_history[-1] = (normalized, assistant_entry["content"])
            if not emitted_chunk:
                yield cleared_input, updated_chat, rag_history

        for trigger in (message_box.submit, send_button.click):
            trigger(
                respond,
                inputs=[message_box, chat, history_state, session_state],
                outputs=[message_box, chat, history_state],
            )

        clear_button.click(lambda: ([], []), outputs=[chat, history_state])

    return demo


if __name__ == "__main__":
    interface = create_interface()
    port = int(os.environ.get("PORT", "7860"))
    interface.launch(server_name="0.0.0.0", server_port=port, share=False)
