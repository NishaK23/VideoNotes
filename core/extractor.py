
"""Extract meeting/video action items, decisions and unresolved questions."""

import os
import re
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        temperature=0,
    )


def clean_items(response: str) -> list[str]:
    """Convert LLM output into a clean list of meaningful items."""

    if not response:
        return []

    items = []

    for line in response.splitlines():
        line = line.strip()

        # Remove markdown formatting and common bullet characters.
        line = re.sub(r"^[-*•]+\s*", "", line)

        # Remove numbering such as 1. or 1) or 1.1
        line = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", line)

        # Remove markdown bold markers.
        line = line.replace("**", "").replace("__", "")

        line = line.strip()

        if not line:
            continue

        # Ignore empty or number-only responses.
        if not re.search(r"[A-Za-z]", line):
            continue

        # Ignore common no-result responses.
        if line.lower() in {
            "none",
            "none found",
            "no questions found",
            "no action items found",
            "no decisions found",
            "no unresolved questions",
            "no open questions",
            "no transcript available",
        }:
            continue

        # Prevent duplicate entries.
        if line.lower() not in [item.lower() for item in items]:
            items.append(line)

    return items


def _extract(transcript: str, instruction: str) -> list[str]:
    """Run the extraction prompt and return clean items."""

    if not transcript or not transcript.strip():
        return []

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a precise meeting and video analyst.

            Follow these rules:
            - Use only information explicitly supported by the transcript.
            - Do not invent information.
            - Do not repeat the same point.
            - Return one complete item per line.
            - Do not use section headings.
            - Do not use numbering.
            - Do not return explanations or introductions.
            - If nothing is found, return exactly: NONE

            {instruction}
            """,
        ),
        (
            "human",
            "Analyse this transcript:\n\n{text}",
        ),
    ])

    response = (
        prompt
        | get_llm()
        | StrOutputParser()
    ).invoke({
        "text": transcript,
        "instruction": instruction,
    })

    return clean_items(response)


def extract_action_items(transcript: str) -> list[str]:
    return _extract(
        transcript,
        """
        Extract actionable tasks mentioned in the transcript.

        For every task, include:
        - The task to be completed.
        - The responsible person, if mentioned.
        - The deadline, if mentioned.

        If the owner or deadline is not mentioned, write
        'Owner not specified' or 'Deadline not specified'.
        Extract only actual tasks, not general discussion.
        """,
    )


def extract_key_decisions(transcript: str) -> list[str]:
    return _extract(
        transcript,
        """
        Extract important decisions explicitly made in the transcript.

        Include the decision and its relevant context.
        Do not mistake suggestions, opinions or possibilities
        for confirmed decisions.
        """,
    )


def extract_questions(transcript: str) -> list[str]:
    return _extract(
        transcript,
        """
        Extract unresolved questions and topics that require
        further discussion, clarification or follow-up.

        Include only questions or unresolved matters supported
        by the transcript. Do not create questions yourself.
        """,
    )
