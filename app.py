
import os
import re
import html
import uuid
import time
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input, cleanup_audio_chunks
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_eng import build_rag_chain, load_rag_chain, ask_question
from core.video_library import (
    list_videos,
    get_video,
    save_video,
    delete_video_record,
)
from core.vector_store import delete_vector_collection


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="VideoNotes | AI Video Assistant",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DARK FOREST GREEN DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #F4F5EF;
    --surface: #FFFFFF;
    --surface-soft: #E9EEE7;
    --green: #183D2D;
    --green-dark: #102C20;
    --green-mid: #426B51;
    --green-light: #DCE8DC;
    --gold: #B79A58;
    --ink: #1D2B22;
    --text: #35483A;
    --muted: #647267;
    --border: #D8E0D5;
    --danger: #A8443D;
}

/* BASE */

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--ink) !important;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.block-container {
    background: var(--bg) !important;
    color: var(--ink) !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    display: none !important;
}

.block-container {
    max-width: 1280px;
    padding: 1.5rem 3rem 3rem;
}

/* GLOBAL TEXT CONTRAST */

h1, h2, h3, h4, h5, h6,
p, li, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: var(--ink) !important;
    opacity: 1 !important;
}

.stCaption,
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* BRAND */

.brand-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.1rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}

.brand-name {
    font-family: 'Manrope', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.06em;
    color: var(--green) !important;
}

.brand-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: var(--green);
    color: white !important;
    height: 34px;
    width: 34px;
    border-radius: 9px;
    margin-right: 8px;
}

.brand-caption {
    color: var(--muted) !important;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* HERO */

.eyebrow {
    color: var(--green-mid) !important;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-family: 'Manrope', sans-serif;
    color: var(--green-dark) !important;
    font-size: clamp(2.5rem, 5vw, 4.3rem);
    line-height: 1.06;
    letter-spacing: -0.075em;
    font-weight: 800;
    margin-bottom: 1rem;
}

.hero-accent {
    color: var(--green-mid) !important;
}

.hero-description {
    max-width: 680px;
    color: var(--muted) !important;
    font-size: 1rem;
    line-height: 1.75;
}

/* HEADINGS */

.section-title {
    font-family: 'Manrope', sans-serif;
    color: var(--green-dark) !important;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.045em;
    margin-bottom: 0.3rem;
}

.section-subtitle {
    color: var(--muted) !important;
    font-size: 0.84rem;
    line-height: 1.6;
    margin-bottom: 1.1rem;
}

/* PANELS */

.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

.panel-label {
    color: var(--green-mid) !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.13em;
    margin-bottom: 0.65rem;
}

.panel-title {
    font-family: 'Manrope', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--green-dark) !important;
    margin-bottom: 0.65rem;
}

.panel-text {
    color: var(--text) !important;
    font-size: 0.9rem;
    line-height: 1.7;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
}

/* SUMMARY SPACING */

.summary-content {
    color: var(--text) !important;
    font-size: 0.93rem;
    line-height: 1.65;
}

.summary-content p {
    color: var(--text) !important;
    margin-top: 0.35rem !important;
    margin-bottom: 0.55rem !important;
}

.summary-content h1,
.summary-content h2,
.summary-content h3,
.summary-content h4 {
    color: var(--green) !important;
    margin-top: 1rem !important;
    margin-bottom: 0.4rem !important;
    line-height: 1.35 !important;
}

.summary-content ul,
.summary-content ol {
    margin-top: 0.25rem !important;
    margin-bottom: 0.55rem !important;
    padding-left: 1.4rem !important;
}

.summary-content li {
    color: var(--text) !important;
    margin-bottom: 0.3rem !important;
    line-height: 1.6 !important;
}

.summary-content hr {
    border-color: var(--border) !important;
    margin: 0.8rem 0 !important;
}

/* METRICS */

.metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
}

.metric-number {
    font-family: 'Manrope', sans-serif;
    color: var(--green) !important;
    font-size: 1.9rem;
    font-weight: 800;
}

.metric-label {
    color: var(--muted) !important;
    font-size: 0.76rem;
    margin-top: 0.2rem;
}

/* LIBRARY */

.library-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 1.3rem;
    min-height: 145px;
}

.library-label {
    color: var(--gold) !important;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    margin-bottom: 0.7rem;
}

.library-title {
    font-family: 'Manrope', sans-serif;
    color: var(--green-dark) !important;
    font-size: 1.05rem;
    font-weight: 800;
    line-height: 1.45;
    margin-bottom: 0.6rem;
}

.library-meta {
    color: var(--muted) !important;
    font-size: 0.74rem;
}

/* FEATURE CARDS */

.feature-card {
    border-top: 3px solid var(--green-mid);
    padding: 1rem 0.5rem;
}

.feature-number {
    color: var(--gold) !important;
    font-size: 0.75rem;
    font-weight: 800;
    margin-bottom: 0.7rem;
}

.feature-title {
    font-family: 'Manrope', sans-serif;
    color: var(--green-dark) !important;
    font-size: 1.05rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.feature-description {
    color: var(--muted) !important;
    font-size: 0.8rem;
    line-height: 1.7;
}

/* TRANSCRIPT TEXT AREA */

.stTextArea textarea,
[data-testid="stTextArea"] textarea,
textarea:disabled {
    background: #FFFFFF !important;
    color: #263B2D !important;
    -webkit-text-fill-color: #263B2D !important;
    opacity: 1 !important;
    border: 1px solid #C7D3C7 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.75 !important;
    padding: 1rem !important;
}

.stTextArea textarea:disabled {
    cursor: text !important;
}

/* INPUTS */

.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #1D2B22 !important;
    border: 1px solid #C7D3C7 !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder {
    color: #78847A !important;
    opacity: 1 !important;
}

.stTextInput label,
.stSelectbox label {
    color: var(--text) !important;
}

/* SELECT MENU */

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background: #FFFFFF !important;
}

[data-baseweb="menu"] li,
[data-baseweb="option"] {
    color: #1D2B22 !important;
}

/* BUTTONS */

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: var(--green) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--green) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: var(--green-dark) !important;
    color: #FFFFFF !important;
    border-color: var(--green-dark) !important;
}

.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--green) !important;
    border: 1px solid #B8C9B9 !important;
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    gap: 1.1rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--muted) !important;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    color: var(--green) !important;
}

/* ITEM CONTAINERS */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.item-number {
    color: var(--gold) !important;
    font-weight: 800;
}

.item-text {
    color: var(--text) !important;
    line-height: 1.7;
}

/* CHAT */

.chat-user {
    background: #E5EEE4;
    border: 1px solid #D2E0D1;
    border-radius: 14px 14px 3px 14px;
    padding: 1rem;
    margin: 0.8rem 0 0.8rem auto;
    max-width: 85%;
    color: #203B29 !important;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.chat-assistant {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 3px;
    padding: 1rem;
    margin: 0.8rem auto 0.8rem 0;
    max-width: 90%;
    color: #263B2D !important;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

/* STATUS / ALERTS */

[data-testid="stAlert"] {
    background: #E8EEE7 !important;
    border: 1px solid #CBD9CA !important;
    color: #203B29 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #203B29 !important;
}

/* PROGRESS */

.stProgress > div > div > div {
    background: var(--gold) !important;
}

[data-testid="stProgressBar"] {
    background: #DFE7DC !important;
}

/* DIVIDERS */

hr {
    border-color: var(--border) !important;
}

/* FOOTER */

.footer {
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    padding-top: 1.1rem;
    display: flex;
    justify-content: space-between;
    color: var(--muted) !important;
    font-size: 0.72rem;
}

/* MOBILE */

@media (max-width: 800px) {
    .block-container {
        padding: 1rem 1rem 3rem;
    }

    .hero-title {
        font-size: 2.7rem;
    }

    .brand-caption {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Home",
    "result": None,
    "chat_history": [],
    "selected_video_id": None,
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_html(value):
    return html.escape(str(value or ""))


def get_value(record, key, default=None):
    if not isinstance(record, dict):
        return default
    return record.get(key, default)


def collection_name_for(video_id):
    clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(video_id))
    return f"video_{clean_id}"


def clean_summary(value):
    """Remove excessive blank lines without destroying summary structure."""
    text = str(value or "").strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text


def normalise_items(value):
    """Accept list-based and older string-based extractor results."""

    if value is None:
        return []

    raw_items = value if isinstance(value, (list, tuple)) else str(value).splitlines()
    cleaned = []

    for item in raw_items:
        item = str(item).strip()
        item = re.sub(r"^[-*•]+\s*", "", item)
        item = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", item)
        item = item.replace("**", "").replace("__", "").strip()

        if not item or not re.search(r"[A-Za-z]", item):
            continue

        if item.lower() in {
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

        if item.lower() not in [x.lower() for x in cleaned]:
            cleaned.append(item)

    return cleaned


def render_metric(number, label):
    st.markdown(f"""
    <div class="metric">
        <div class="metric-number">{safe_html(number)}</div>
        <div class="metric-label">{safe_html(label)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_panel(label, title, content):
    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">{safe_html(label)}</div>
        <div class="panel-title">{safe_html(title)}</div>
        <div class="panel-text">{safe_html(content)}</div>
    </div>
    """, unsafe_allow_html=True)


def render_summary(summary):
    summary = clean_summary(summary)

    if not summary:
        st.info("No summary available.")
        return

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">THE BIG PICTURE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Video summary</div>',
                unsafe_allow_html=True)

    # Render markdown so headings and bullets are formatted properly.
    # Compact CSS above prevents oversized paragraph gaps.
    st.markdown(
        f'<div class="summary-content">{summary}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_items(items, empty_message):
    items = normalise_items(items)

    if not items:
        st.info(empty_message)
        return

    for index, item in enumerate(items, start=1):
        with st.container(border=True):
            left, right = st.columns([0.1, 0.9])

            with left:
                st.markdown(
                    f'<div class="item-number">{index:02d}</div>',
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    f'<div class="item-text">{safe_html(item)}</div>',
                    unsafe_allow_html=True,
                )


def load_library():
    try:
        return list_videos() or []
    except Exception as exc:
        st.error(f"Could not load your video library: {exc}")
        return []


def open_saved_video(video_id):
    record = get_video(video_id)

    if not record:
        st.error("Video record not found.")
        return

    collection_name = get_value(
        record,
        "collection_name",
        collection_name_for(video_id),
    )

    record["rag_chain"] = load_rag_chain(collection_name)

    st.session_state.result = record
    st.session_state.selected_video_id = video_id
    st.session_state.chat_history = []
    st.session_state.page = "Analysis"


def delete_saved_video(video):
    video_id = get_value(video, "video_id")
    collection_name = get_value(
        video,
        "collection_name",
        collection_name_for(video_id),
    )

    delete_vector_collection(collection_name)
    delete_video_record(video_id)

    if st.session_state.selected_video_id == video_id:
        st.session_state.result = None
        st.session_state.selected_video_id = None
        st.session_state.chat_history = []


# ============================================================
# NAVIGATION
# ============================================================

st.markdown("""
<div class="brand-bar">
    <div class="brand-name">
        <span class="brand-icon">▶</span>VideoNotes
    </div>
    <div class="brand-caption">AI VIDEO ASSISTANT</div>
</div>
""", unsafe_allow_html=True)

nav1, nav2, nav3 = st.columns(3)

with nav1:
    if st.button("⌂  Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()

with nav2:
    if st.button("▤  My Library", use_container_width=True):
        st.session_state.page = "Library"
        st.rerun()

with nav3:
    if st.button("＋  New Analysis", use_container_width=True):
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.page = "Home"
        st.rerun()


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="eyebrow">YOUR PERSONAL VIDEO RESEARCH SPACE</div>

    <div class="hero-title">
        Understand videos<br>
        <span class="hero-accent">faster.</span>
    </div>

    <div class="hero-description">
        Turn long videos into searchable knowledge, useful summaries
        and clear answers. One workspace for everything you need
        to understand from a video.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="panel">
        <div class="panel-label">01 / ADD YOUR SOURCE</div>
        <div class="panel-title">What would you like to understand?</div>
        <div class="panel-text">
            Paste a YouTube link or enter the path to a local audio/video file.
        </div>
    </div>
    """, unsafe_allow_html=True)

    source = st.text_input(
        "Video URL or local file path",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="video_source",
    )

    col1, col2 = st.columns(2)

    with col1:
        language = st.selectbox(
            "Transcription language",
            ["english", "hinglish"],
            index=0,
        )

    with col2:
        st.markdown("""
        <div style="padding-top:2rem;color:#647267;font-size:0.8rem">
            Supported: YouTube URLs and local audio/video files.
        </div>
        """, unsafe_allow_html=True)

    analyse_button = st.button(
        "Analyse video  →",
        use_container_width=True,
    )

    if analyse_button:

        if not source.strip():
            st.warning("Please enter a YouTube URL or local file path.")

        else:
            progress = st.progress(0)
            status = st.empty()

            try:
                # 1. Audio processing
                status.info("1/6 · Processing audio...")
                chunks = process_input(source.strip())
                progress.progress(15)

                # 2. Transcription
                status.info(
                    "2/6 · Transcribing audio. Please wait..."
                )

                transcript = transcribe_all(chunks, language)

                if not transcript or not str(transcript).strip():
                    raise ValueError("Transcription returned no text.")

                # Remove temporary audio only after successful transcription.
                try:
                    cleanup_audio_chunks(chunks)
                except Exception as cleanup_error:
                    st.warning(f"Audio cleanup warning: {cleanup_error}")

                progress.progress(35)

                # 3. Title
                status.info("3/6 · Generating video title...")
                title = generate_title(transcript)
                progress.progress(45)

                # 4. Summary
                status.info("4/6 · Creating video summary...")
                summary = summarize(transcript)
                progress.progress(60)

                # 5. Extraction
                status.info(
                    "5/6 · Extracting action items, decisions and questions..."
                )

                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                progress.progress(78)

                # 6. Vector database and RAG
                status.info("6/6 · Building searchable video knowledge base...")

                video_id = uuid.uuid4().hex[:12]
                collection_name = collection_name_for(video_id)

                rag_chain = build_rag_chain(
                    transcript,
                    collection_name=collection_name,
                    video_id=video_id,
                )

                record = {
                    "video_id": video_id,
                    "collection_name": collection_name,
                    "title": title,
                    "source": source.strip(),
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": normalise_items(action_items),
                    "key_decisions": normalise_items(decisions),
                    "open_questions": normalise_items(questions),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }

                save_video(record)

                record["rag_chain"] = rag_chain

                st.session_state.result = record
                st.session_state.selected_video_id = video_id
                st.session_state.chat_history = []
                st.session_state.page = "Analysis"

                progress.progress(100)
                status.success("Video analysis completed!")

                time.sleep(0.4)
                st.rerun()

            except Exception as exc:
                status.error(f"Analysis failed: {exc}")
                st.exception(exc)

    # Features
    st.markdown("<div style='height:3rem'></div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">From video to insight</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A complete workflow to understand video content.'
        '</div>',
        unsafe_allow_html=True,
    )

    features = [
        ("01", "Transcribe", "Convert spoken content into readable text."),
        ("02", "Summarise", "Understand the main ideas quickly."),
        ("03", "Extract", "Identify tasks, decisions and follow-ups."),
        ("04", "Ask anything", "Chat with your video using RAG."),
    ]

    feature_cols = st.columns(4)

    for col, (number, title, description) in zip(feature_cols, features):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-number">{number}</div>
                <div class="feature-title">{safe_html(title)}</div>
                <div class="feature-description">{safe_html(description)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Recently analysed
    videos = load_library()

    if videos:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Recently analysed</div>',
            unsafe_allow_html=True,
        )

        recent_cols = st.columns(min(3, len(videos)))

        for col, video in zip(recent_cols, videos[:3]):
            with col:
                video_id = get_value(video, "video_id", "")
                title = get_value(video, "title", "Untitled video")

                st.markdown(f"""
                <div class="library-card">
                    <div class="library-label">VIDEO LIBRARY</div>
                    <div class="library-title">{safe_html(title)}</div>
                    <div class="library-meta">
                        {safe_html(get_value(video, "created_at", ""))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(
                    "Open video →",
                    key=f"recent_{video_id}",
                    use_container_width=True,
                ):
                    try:
                        open_saved_video(video_id)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not open video: {exc}")


# ============================================================
# LIBRARY PAGE
# ============================================================

elif st.session_state.page == "Library":

    st.markdown('<div class="eyebrow">YOUR COLLECTION</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="hero-title">
        My <span class="hero-accent">Library.</span>
    </div>

    <div class="hero-description">
        All your analysed videos in one place.
        Reopen an analysis without downloading or transcribing it again.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    videos = load_library()

    metric_col, _ = st.columns([1, 3])

    with metric_col:
        render_metric(len(videos), "Videos analysed")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if not videos:
        st.info("Your video library is empty.")

        if st.button("Analyse your first video →"):
            st.session_state.page = "Home"
            st.rerun()

    else:
        search = st.text_input(
            "Search videos",
            placeholder="Search by video title...",
        )

        filtered = [
            video for video in videos
            if search.lower() in str(
                get_value(video, "title", "")
            ).lower()
        ]

        if not filtered:
            st.warning("No videos found.")

        for start in range(0, len(filtered), 2):
            columns = st.columns(2)

            for col, video in zip(columns, filtered[start:start + 2]):
                with col:
                    video_id = get_value(video, "video_id", "")
                    title = get_value(video, "title", "Untitled video")
                    created = get_value(video, "created_at", "")

                    st.markdown(f"""
                    <div class="library-card">
                        <div class="library-label">▶ ANALYSED VIDEO</div>
                        <div class="library-title">{safe_html(title)}</div>
                        <div class="library-meta">
                            {safe_html(created)} · ID: {safe_html(video_id)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    open_col, delete_col = st.columns([2, 1])

                    with open_col:
                        if st.button(
                            "Open analysis →",
                            key=f"open_{video_id}",
                            use_container_width=True,
                        ):
                            try:
                                open_saved_video(video_id)
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Could not open video: {exc}")

                    with delete_col:
                        if st.button(
                            "Delete",
                            key=f"delete_{video_id}",
                            type="secondary",
                            use_container_width=True,
                        ):
                            st.session_state[
                                f"confirm_delete_{video_id}"
                            ] = True

                    if st.session_state.get(
                        f"confirm_delete_{video_id}", False
                    ):
                        st.warning(f"Delete '{title}' permanently?")

                        yes_col, no_col = st.columns(2)

                        with yes_col:
                            if st.button(
                                "Yes, delete",
                                key=f"yes_{video_id}",
                            ):
                                try:
                                    delete_saved_video(video)
                                    st.session_state[
                                        f"confirm_delete_{video_id}"
                                    ] = False
                                    st.success("Video deleted.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Deletion failed: {exc}")

                        with no_col:
                            if st.button(
                                "Cancel",
                                key=f"cancel_{video_id}",
                            ):
                                st.session_state[
                                    f"confirm_delete_{video_id}"
                                ] = False
                                st.rerun()


# ============================================================
# ANALYSIS PAGE
# ============================================================

elif st.session_state.page == "Analysis":

    record = st.session_state.result

    if not record:
        st.warning("No video is currently open.")

        if st.button("Go to My Library"):
            st.session_state.page = "Library"
            st.rerun()

        st.stop()

    title = get_value(record, "title", "Video analysis")
    transcript = get_value(record, "transcript", "")
    summary = get_value(record, "summary", "")

    st.markdown(
        '<div class="eyebrow">VIDEO INTELLIGENCE / ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="hero-title">{safe_html(title)}</div>',
        unsafe_allow_html=True,
    )

    st.caption(f"Source: {get_value(record, 'source', 'Not available')}")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    metric_cols = st.columns(3)

    with metric_cols[0]:
        render_metric(len(str(transcript).split()), "Transcript words")

    with metric_cols[1]:
        render_metric(
            len(normalise_items(get_value(record, "action_items"))),
            "Action items",
        )

    with metric_cols[2]:
        render_metric(
            len(normalise_items(get_value(record, "open_questions"))),
            "Open questions",
        )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    overview_tab, takeaways_tab, transcript_tab, chat_tab = st.tabs([
        "Overview",
        "Key Takeaways",
        "Transcript",
        "Ask the Video",
    ])

    # OVERVIEW
    with overview_tab:

        render_summary(summary)

        left, right = st.columns(2)

        with left:
            st.markdown("""
            <div class="panel">
                <div class="panel-label">01 / UNDERSTANDING</div>
                <div class="panel-title">Structured insights</div>
                <div class="panel-text">
                    Review key decisions, action items and unresolved
                    questions extracted from the video.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class="panel">
                <div class="panel-label">02 / RETRIEVAL</div>
                <div class="panel-title">Context-aware answers</div>
                <div class="panel-text">
                    Ask questions about the video. RAG retrieves
                    relevant transcript sections to prepare an answer.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # KEY TAKEAWAYS
    with takeaways_tab:

        action_col, decision_col = st.columns(2, gap="large")

        with action_col:
            st.markdown(
                '<div class="section-title">Action items</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Tasks that need to be completed.</div>',
                unsafe_allow_html=True,
            )

            render_items(
                get_value(record, "action_items", []),
                "No action items identified.",
            )

        with decision_col:
            st.markdown(
                '<div class="section-title">Key decisions</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-subtitle">Important decisions made in the video.</div>',
                unsafe_allow_html=True,
            )

            render_items(
                get_value(record, "key_decisions", []),
                "No key decisions identified.",
            )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Open questions & follow-ups</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Unresolved topics requiring clarification or follow-up.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_items(
            get_value(record, "open_questions", []),
            "No open questions identified.",
        )

    # TRANSCRIPT
    with transcript_tab:

        st.markdown(
            '<div class="section-title">Full transcript</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'The complete text generated from the video audio.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.text_area(
            "Transcript",
            value=str(transcript),
            height=500,
            disabled=True,
            label_visibility="collapsed",
        )

        st.download_button(
            "Download transcript",
            data=str(transcript),
            file_name="video_transcript.txt",
            mime="text/plain",
        )

    # ASK THE VIDEO
    with chat_tab:

        st.markdown(
            '<div class="section-title">Ask anything.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Explore the information contained in this video.'
            '</div>',
            unsafe_allow_html=True,
        )

        for message in st.session_state.chat_history:

            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{safe_html(message["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-assistant">{safe_html(message["content"])}</div>',
                    unsafe_allow_html=True,
                )

        with st.form("video_chat_form", clear_on_submit=True):

            question = st.text_input(
                "Your question",
                placeholder="What are the most important points?",
            )

            send = st.form_submit_button(
                "Send question →",
                use_container_width=True,
            )

        if send and question.strip():

            st.session_state.chat_history.append({
                "role": "user",
                "content": question.strip(),
            })

            try:
                with st.spinner("Searching the transcript and preparing your answer..."):
                    answer = ask_question(
                        record["rag_chain"],
                        question.strip(),
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": str(answer),
                })

            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
                st.session_state.chat_history.pop()

            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    # PAGE NAVIGATION
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    back_col, new_col = st.columns(2)

    with back_col:
        if st.button("← Back to library", use_container_width=True):
            st.session_state.page = "Library"
            st.rerun()

    with new_col:
        if st.button("＋ Analyse another video", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.page = "Home"
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <span>VIDEONOTES · AI VIDEO ASSISTANT</span>
    <span>Transcribe. Summarise. Ask.</span>
</div>
""", unsafe_allow_html=True)
