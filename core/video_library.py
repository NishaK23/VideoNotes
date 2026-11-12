
"""Persistent metadata and transcript library for analysed videos."""

import json
import os
import re
import tempfile

from datetime import datetime, timezone
from pathlib import Path


LIBRARY_DIR = Path("video_library")
LIBRARY_FILE = LIBRARY_DIR / "videos.json"
TRANSCRIPTS_DIR = LIBRARY_DIR / "transcripts"


def _safe_video_id(video_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,80}", video_id or ""):
        raise ValueError("Invalid video ID.")

    return video_id


def _read_library():

    if not LIBRARY_FILE.exists():
        return {}

    try:
        data = json.loads(
            LIBRARY_FILE.read_text(encoding="utf-8")
        )

        return data if isinstance(data, dict) else {}

    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            "Could not read the video library JSON file."
        ) from exc


def list_videos():

    videos = list(_read_library().values())

    return sorted(
        videos,
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )


def get_video(video_id):

    video_id = _safe_video_id(video_id)

    record = _read_library().get(video_id)

    if record is None:
        return None

    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.txt"

    if not transcript_file.is_file():
        raise FileNotFoundError(
            f"Transcript missing for video {video_id}."
        )

    record = dict(record)

    record["transcript"] = transcript_file.read_text(
        encoding="utf-8"
    )

    return record


def save_video(video):

    required = {
        "video_id",
        "title",
        "collection_name",
        "transcript",
    }

    missing = required.difference(video)

    if missing:
        raise ValueError(
            f"Missing fields: {', '.join(sorted(missing))}"
        )

    video_id = _safe_video_id(str(video["video_id"]))

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.txt"

    transcript_file.write_text(
        video["transcript"],
        encoding="utf-8",
    )

    library = _read_library()

    record = {
        key: value
        for key, value in video.items()
        if key not in {"transcript", "rag_chain"}
    }

    record["transcript_file"] = str(
        Path("transcripts") / transcript_file.name
    )

    record.setdefault(
        "created_at",
        datetime.now(timezone.utc).isoformat(),
    )

    library[video_id] = record

    fd, temporary_name = tempfile.mkstemp(
        prefix="videos_",
        suffix=".tmp",
        dir=LIBRARY_DIR,
    )

    try:

        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(
                library,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temporary_name, LIBRARY_FILE)

    finally:

        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def delete_video_record(video_id):

    video_id = _safe_video_id(video_id)

    library = _read_library()

    if video_id not in library:
        return False

    del library[video_id]

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = LIBRARY_FILE.with_suffix(".tmp")

    temporary_file.write_text(
        json.dumps(library, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary_file, LIBRARY_FILE)

    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.txt"

    if transcript_file.exists():
        transcript_file.unlink()

    return True
