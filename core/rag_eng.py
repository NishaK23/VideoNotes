
"""RAG question answering for one selected video."""

import os
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


@lru_cache(maxsize=1)
def get_llm():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return ChatGroq(
        model=os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b",
        ),
        api_key=api_key,
        temperature=0.2,
    )


def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def make_chain(retriever):

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an accurate video assistant. Answer using only "
            "the supplied transcript context. If the answer is not "
            "supported by the context, say that you could not find it "
            "in this video. Do not invent details.\n\n"
            "Transcript context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def build_rag_chain(
    transcript,
    collection_name,
    video_id=None,
):

    vector_store = build_vector_store(
        transcript=transcript,
        collection_name=collection_name,
        video_id=video_id,
    )

    retriever = get_retriever(
        vector_store,
        k=4,
    )

    return make_chain(retriever)


def load_rag_chain(collection_name):

    vector_store = load_vector_store(collection_name)

    retriever = get_retriever(
        vector_store,
        k=4,
    )

    return make_chain(retriever)


def ask_question(rag_chain, question):

    question = question.strip()

    if not question:
        raise ValueError("Please enter a question.")

    return rag_chain.invoke(question)
