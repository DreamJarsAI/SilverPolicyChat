"""Factory helpers to assemble the OpenAI Agent and retrieval tools."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import List

from agents import Agent
from agents.run import AgentRunner
from agents.run_context import RunContextWrapper
from agents.tool import function_tool
from openai import OpenAI

from config import Settings
from policy_store import PolicyStore

logger = logging.getLogger(__name__)


@dataclass
class PolicyAgentContext:
    """Dependencies that tools need access to during an agent run."""

    store: PolicyStore
    client: OpenAI
    settings: Settings


def _embedding(client: OpenAI, settings: Settings, text: str) -> List[float]:
    response = client.embeddings.create(model=settings.embedding_model, input=[text])
    return response.data[0].embedding


@function_tool
def retrieve_policy_context(
    run_context: RunContextWrapper["PolicyAgentContext"],
    question: str,
    top_k: int = 4,
) -> dict:
    """Look up the most relevant policy passages for the given question."""

    context: PolicyAgentContext = run_context.context
    embedding = _embedding(context.client, context.settings, question)
    results = context.store.similar_chunks(embedding, top_k=top_k)
    payload = [
        {
            "title": chunk.title,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "excerpt": chunk.text,
        }
        for chunk in results
    ]
    return {"chunks": payload}


@function_tool
def list_available_policies(run_context: RunContextWrapper["PolicyAgentContext"]) -> dict:
    """Return every policy document currently indexed for this assistant."""

    context: PolicyAgentContext = run_context.context
    documents = context.store.list_documents()
    return {"documents": documents, "count": len(documents)}


def should_list_documents(message: str, context: PolicyAgentContext) -> bool:
    """Use heuristics and a lightweight LLM classification to detect catalog requests."""

    lowered = message.lower()
    greetings = {
        "hi",
        "hello",
        "hey",
        "yo",
        "yo there",
        "how are you",
        "how are you doing",
        "what's up",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if lowered in greetings:
        return True
    if "how many" in lowered and any(token in lowered for token in ("doc", "document", "policy", "policies")):
        return True
    if lowered.startswith("list") and any(token in lowered for token in ("doc", "policy")):
        return True
    catalog_prompts = {
        "what documents do you have",
        "what policies do you have",
        "which documents",
        "which policies",
        "what documents exist",
        "what policies exist",
    }
    if lowered in catalog_prompts:
        return True

    tokens = lowered.split()
    if len(tokens) <= 4 and not any(
        token in lowered
        for token in (
            "policy",
            "document",
            "handbook",
            "scholarship",
            "curriculum",
            "registration",
            "attendance",
            "grade",
            "tuition",
            "requirement",
        )
    ):
        # Very short social phrases default to catalog response so the user sees available policies.
        return True

    if not context.settings.openai_api_key:
        return False

    try:
        classification = context.client.chat.completions.create(
            model=context.settings.gpt_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's intent. Respond with exactly one word: CATALOG if the user is greeting "
                        "or asking about available documents/policies, otherwise respond NORMAL."
                    ),
                },
                {"role": "user", "content": message},
            ],
            max_completion_tokens=3,
        )
    except Exception as exc:
        logger.warning("Intent classification fallback failed: %s", exc)
        return False

    content = classification.choices[0].message.content or ""
    return "catalog" in content.lower()


def format_catalog_response(context: PolicyAgentContext) -> str:
    titles = context.store.list_documents()
    titles.sort()
    count = len(titles)
    bullet_list = "\n".join(f"- {title}" for title in titles) or "(no documents indexed yet)"
    heading = f"I currently have {count} policy document{'s' if count != 1 else ''} in my knowledge base:"
    return (
        f"{heading}\n\n{bullet_list}\n\n"
        "Ask about any of these documents and I'll cite the relevant sections."
    )


def build_agent(settings: Settings) -> Agent[PolicyAgentContext]:
    instructions = (
        "You are a school policy assistant."
        " Always begin by calling `retrieve_policy_context` with the student's question to gather"
        " relevant passages. If the student asks about available policies or greets you, you may"
        " call `list_available_policies` instead."
        " Always cite sources using the exact policy title and page number from the tool outputs."
        " If the tools do not provide enough information, say you do not know."
    )

    return Agent(
        name="policy_assistant",
        instructions=instructions,
        model=settings.gpt_model,
        tools=[retrieve_policy_context, list_available_policies],
    )


def run_agent(
    agent: Agent[PolicyAgentContext],
    runner: AgentRunner,
    context: PolicyAgentContext,
    history: List[tuple[str, str]],
    message: str,
) -> str:
    conversation_lines: List[str] = []
    for user, assistant in history:
        conversation_lines.append(f"Student: {user}")
        conversation_lines.append(f"Assistant: {assistant}")
    conversation_lines.append(f"Student: {message}")
    prompt = "\n".join(conversation_lines)

    async def _run() -> str:
        run_result = await runner.run(agent, prompt, context=context)
        return run_result.final_output_as(str)

    return asyncio.run(_run())
