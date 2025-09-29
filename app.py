"""Gradio front-end for the OpenAI Agents-based School Policy assistant."""
from __future__ import annotations

import logging
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
    run_agent,
    should_list_documents,
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

    with gr.Blocks(title="School Policy Assistant", theme=gr.themes.Soft()) as demo:
        with gr.Column():
            gr.Markdown(
                """
                # School Policy Assistant
                _Grounded answers to policy questions, powered by OpenAI Agents._
                """
            )

        session_state = gr.State({"authenticated": False, "username": None, "email": None})
        history_state = gr.State([])
        registration_state = gr.State({"username": None, "email": None, "code_sent": False})
        reset_state = gr.State({"email": None, "code_sent": False})

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
            chat = gr.Chatbot(
                label="Policy Assistant",
                type="messages",
                height=520,
                show_copy_button=True,
            )
            with gr.Row():
                message_box = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about attendance, grading, scholarships, ...",
                    scale=4,
                    interactive=False,
                )
                send_button = gr.Button("Send", variant="primary", scale=1, interactive=False)

            clear_button = gr.Button("Clear conversation", variant="secondary")

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
                chat_container,
                message_box,
                send_button,
                login_username,
                login_password,
            ],
        )

        def handle_logout(current_session: Dict[str, Any]) -> tuple[Any, ...]:
            _ = current_session  # not used beyond signature
            return (
                gr.update(value="You have been signed out.", visible=True),
                {"authenticated": False, "username": None, "email": None},
                gr.update(value="Not signed in."),
                gr.update(visible=False),
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
                    gr.update(value=f"✅ {message}", visible=True),
                    {"username": None, "email": None, "code_sent": False},
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(value=""),
                    gr.update(value=""),
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
                    gr.update(value=f"✅ {message}", visible=True),
                    {"email": None, "code_sent": False},
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False),
                    gr.update(visible=False),
                    gr.update(value=""),
                )

            keep_state = state or {"email": None, "code_sent": False}
            return (
                gr.update(value=f"❌ {message}", visible=True),
                keep_state,
                gr.update(value=code, visible=True),
                gr.update(value="", visible=True),
                gr.update(visible=True),
                gr.update(value=keep_state.get("email") or ""),
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
            ],
        )

        def respond(
            user_message: str,
            chat_history: List[dict],
            rag_history: List[Tuple[str, str]],
            session: Dict[str, Any],
        ):
            rag_history = list(rag_history or [])
            normalized = user_message.strip()
            if not session or not session.get("authenticated"):
                return "", chat_history, rag_history
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
                inputs=[message_box, chat, history_state, session_state],
                outputs=[message_box, chat, history_state],
            )

        clear_button.click(lambda: ([], []), outputs=[chat, history_state])

    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
