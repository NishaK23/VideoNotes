
"""Download, convert, chunk and clean up audio files."""

import os
import re
import shutil
import uuid
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio into a unique temporary folder."""

    run_id = uuid.uuid4().hex
    run_dir = DOWNLOAD_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(run_dir / "%(id)s.%(ext)s")

    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "video")

    wav_path = run_dir / f"{video_id}.wav"

    if not wav_path.exists():
        candidates = list(run_dir.glob("*.wav"))
        if not candidates:
            raise FileNotFoundError(
                "YouTube audio download did not produce a WAV file."
            )
        wav_path = candidates[0]

    return str(wav_path)


def convert_to_wav(input_path: str) -> str:
    """Convert a local audio/video file to mono 16 kHz WAV."""

    source = Path(input_path)

    if not source.is_file():
        raise FileNotFoundError(f"File not found: {input_path}")

    output_path = source.with_name(
        f"{source.stem}_converted.wav"
    )

    audio = AudioSegment.from_file(str(source))
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(str(output_path), format="wav")

    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list[str]:
    """Split WAV audio into chunks of the requested duration."""

    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be greater than zero.")

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for index, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{index}.wav"

        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list[str]:
    """Download or convert input, then return audio chunk paths."""

    source = source.strip()

    if not source:
        raise ValueError("Please provide a YouTube URL or local file path.")

    if source.startswith(("http://", "https://")):
        print("Downloading YouTube audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Converting local media...")
        wav_path = convert_to_wav(source)

    print("Splitting audio into 10-minute chunks...")
    chunks = chunk_audio(wav_path)

    print(f"Created {len(chunks)} audio chunk(s).")

    return chunks


def cleanup_audio_chunks(chunk_paths: list[str]) -> None:
    """
    Delete temporary downloaded audio and its chunks.

    Only removes folders created inside the application's downloads folder.
    Never deletes user-provided local source files.
    """

    if not chunk_paths:
        return

    downloads_root = DOWNLOAD_DIR.resolve()
    run_dirs = {
        Path(path).resolve().parent
        for path in chunk_paths
    }

    for run_dir in run_dirs:

        try:
            run_dir.relative_to(downloads_root)
        except ValueError:
            continue

        if run_dir == downloads_root:
            continue

        if run_dir.exists() and run_dir.is_dir():
            shutil.rmtree(run_dir)
            print(f"Cleaned temporary audio files: {run_dir}")
