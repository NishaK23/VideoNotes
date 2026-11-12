"""Command-line entry point for the AI Video Assistant."""
import uuid
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_eng import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    if not source or not source.strip():
        raise ValueError("A YouTube URL or local video path is required.")

    video_id = uuid.uuid4().hex[:16]
    collection_name = f"video_{video_id}"

    print("Starting VideoNotes...")
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    if not transcript.strip():
        raise RuntimeError("Transcription returned no text. Check the audio input.")

    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(
        transcript,
        collection_name=collection_name,
        video_id=video_id,
    )

    return {
        "video_id": video_id,
        "collection_name": collection_name,
        "source": source,
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip().lower() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print("                 VideoNotes")
    print("          AI-Powered Video Assistant")
    print("=" * 60)

    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction items:\n{result['action_items']}")
    print(f"\nKey decisions:\n{result['key_decisions']}")
    print(f"\nOpen questions:\n{result['open_questions']}")
    print("=" * 60)

    print("\nVideoNotes Chat – Ask questions about your video")
    print("(Type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if question:
            print(f"\nAssistant: {ask_question(result['rag_chain'], question)}\n")
