"""LLM-powered title generation and long-transcript summarisation."""
import os
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        api_key=api_key,
        temperature=0.2,
    )


def split_transcript(transcript: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError("Cannot summarise an empty transcript.")

    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise this portion of a video transcript. Preserve key facts and concepts."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    # Kept sequential for now to avoid unexpected Groq rate-limit failures.
    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in split_transcript(transcript)
    ]

    combine_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Combine the partial summaries into one clear, structured summary. "
            "Use concise headings and bullet points. Do not add facts that are absent.",
        ),
        ("human", "{text}"),
    ])
    return (combine_prompt | llm | StrOutputParser()).invoke(
        {"text": "\n\n".join(chunk_summaries)}
    )


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled video"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a clear, professional video title of no more than 8 words. "
            "Return only the title.",
        ),
        ("human", "{text}"),
    ])
    return (prompt | get_llm() | StrOutputParser()).invoke(
        {"text": transcript[:4000]}
    ).strip().strip('"')
