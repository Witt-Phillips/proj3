#!/usr/bin/env python3
"""
Part 1: Speech-to-Text Transcription
Law and LLMs - Project 3

Uses OpenAI's Whisper model to transcribe audio from the Depp v. Heard video.
Supports both: (1) local audio file, (2) YouTube URL (downloads and extracts audio).
Whisper automatically handles long audio via a sliding 30-second window.
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Assignment video: https://www.youtube.com/watch?v=IfN9F93xr_g
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=IfN9F93xr_g"


def _is_url(url_or_path: str) -> bool:
    """Check if input looks like a URL."""
    return bool(re.match(r"^https?://", url_or_path.strip()))


def _download_youtube_audio(url: str, output_dir: Optional[Path] = None) -> Path:
    """
    Download audio from YouTube URL to a file.
    Preprocessing per assignment: cannot hand URL to Whisper; must extract audio first.
    Downloads bestaudio (mp4/m4a); audioread loads it without ffmpeg on macOS.
    """
    import yt_dlp

    output_dir = output_dir or Path(tempfile.gettempdir())
    output_tmpl = str(output_dir / "youtube_audio.%(ext)s")

    # Download best audio; skip ffmpeg postprocessor (audioread handles mp4/m4a on macOS)
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_tmpl,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


def _load_audio(path: str):
    """Load audio file to float32 mono 16kHz numpy array (Whisper format)."""
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample

    # Try standard WAV first
    with open(path, "rb") as f:
        header = f.read(4)
    if header in (b"RIFF", b"RIFX"):
        sample_rate, audio = wavfile.read(path)
    else:
        # Use audioread for M4A, MP3, etc. (uses Core Audio on macOS, no ffmpeg)
        import audioread
        with audioread.audio_open(path) as f:
            sample_rate = f.samplerate
            channels = f.channels
            raw = b"".join(buf for buf in f)
        audio = np.frombuffer(raw, dtype=np.int16)
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if sample_rate != 16000:
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples).astype(np.float32)
    # Clip to valid range (resampling can produce slight overshoot) and ensure contiguous
    audio = np.ascontiguousarray(np.clip(audio, -1.0, 1.0))
    return audio, 16000


def transcribe_audio(
    audio_path_or_url: str,
    output_path: Optional[str] = None,
    model_name: str = "base",
) -> str:
    """
    Transcribe audio using OpenAI Whisper.

    Args:
        audio_path_or_url: Path to a local audio file (.wav, .mp3, etc.) OR a
            YouTube URL. For URLs, downloads and extracts audio first (preprocessing).
        output_path: Optional path to save the transcription. If None, uses
            {source}_transcription.txt
        model_name: Whisper model. Use 'base.en' for English (default), or
            'small', 'medium' for better accuracy.

    Returns:
        The transcription text.
    """
    import whisper

    # YouTube URL: download and extract audio (preprocessing per assignment)
    if _is_url(audio_path_or_url):
        print(f"Downloading and extracting audio from: {audio_path_or_url}")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = _download_youtube_audio(audio_path_or_url, Path(tmpdir))
            print(f"Preprocessed audio ready")
            # Load while file still exists (tmpdir is deleted when we exit)
            print(f"Loading audio: {audio_path}")
            audio, sample_rate = _load_audio(str(audio_path))
        audio_path = Path("youtube_audio")  # For output filename
    else:
        audio_path = Path(audio_path_or_url)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        print(f"Loading audio: {audio_path}")
        audio, sample_rate = _load_audio(str(audio_path))

    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)

    print(f"Transcribing: {audio_path}")
    # Whisper handles long audio internally with a sliding 30-second window
    # Force English for legal/court dialogue (Depp v. Heard)
    # condition_on_previous_text=False reduces repetition hallucinations in long audio
    result = model.transcribe(
        audio, fp16=False, language="en", condition_on_previous_text=False
    )

    transcription = result["text"].strip()

    # Save to file
    if output_path is None:
        output_path = Path.cwd() / f"{audio_path.stem}_transcription.txt"
    else:
        output_path = Path(output_path)

    output_path.write_text(transcription, encoding="utf-8")
    print(f"Transcription saved to: {output_path}")

    return transcription


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using OpenAI Whisper (Part 1 of Project 3)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_YOUTUBE_URL,
        help="YouTube URL or path to audio file (default: assignment video URL)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: {audio_stem}_transcription.txt)",
    )
    parser.add_argument(
        "-m", "--model",
        default="base.en",
        choices=[
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en", "large",
        ],
        help="Whisper model (default: base.en for English)",
    )
    args = parser.parse_args()

    try:
        text = transcribe_audio(
            audio_path_or_url=args.input,
            output_path=args.output,
            model_name=args.model,
        )
        print("\n--- Transcription ---")
        print(text)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
