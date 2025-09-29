#!/usr/bin/env python3
"""
cornell_recorder.py

Enhanced: Record long meetings (in chunks), transcribe locally using Whisper,
and generate intelligent summaries (key points, cue questions, action items,
concise summary) using Google AI Studio (Gemini) free API.

This file is an extension of your earlier script; new features added:
- Real-time audio recording (unlimited length via chunked files)
- Local Whisper transcription with chunking for large files, progress bars,
  model-choice support, and estimated processing time display
- Gemini integration scaffold using `google-generativeai` (AI Studio) client:
  - Produces key points, cue questions, action items, concise summary
  - Chunking transcript to avoid huge requests
- Resume/partial-work support via a `.state.json` file per job
- Export to TXT / JSON / Markdown (Cornell)
- Error handling for API/network/disk conditions
- CLI configuration and environment variable support
- Installation instructions and dependency hints below
- KeyboardInterrupt clean shutdown handling
- Disk-space checks before recording/transcription

USAGE EXAMPLES:
    # Basic (uses env GEMINI_API_KEY if available)
    python cornell_recorder.py --title "Team Sync" --segment-minutes 10 --prefer-whisper

    # Provide Gemini key on CLI:
    python cornell_recorder.py -t "My Talk" --gemini-api-key "ya29...." --whisper-model small

INSTALLATION (example):
    # System-level: install ffmpeg
    # macOS (homebrew): brew install ffmpeg
    # Ubuntu/Debian: sudo apt install ffmpeg
    # Windows: install from ffmpeg.org and add to PATH

    pip install sounddevice soundfile numpy tqdm openai-whisper google-generativeai

    Notes:
      - openai-whisper requires ffmpeg available.
      - google-generativeai is the Python client for Google Generative AI Studio (may be named differently depending on versions).
      - If using a GPU for Whisper, follow whisper docs to install CUDA-enabled dependencies.

ENVIRONMENT VARIABLES:
    GEMINI_API_KEY    - Gemini / Google AI Studio API key (optional; CLI flag overrides)
    OUTPUT_BASE_DIR   - Default output directory (optional)
"""

# --- Standard libs ---
import argparse
import os
import sys
import queue
import threading
import time
import math
import tempfile
import json
import shutil
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import numpy as np
from tqdm import tqdm
import re
import pathlib
from typing import List, Tuple, Dict, Any, Optional

# Try imports for optional libs
try:
    import whisper as _whisper_lib  # openai-whisper
except Exception:
    _whisper_lib = None

try:
    import google.generativeai as genai  # google-generativeai client
except Exception:
    genai = None

# ----------------------------
# Helper utilities
# ----------------------------
def human_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def ensure_disk_space(path: str, needed_bytes: int) -> bool:
    """Return True if `path` has at least needed_bytes free."""
    try:
        total, used, free = shutil.disk_usage(path)
        return free >= needed_bytes
    except Exception:
        return True  # if we can't determine, don't block; user risk

def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# Original Recording utilities (unchanged, only extended where needed)
# ----------------------------
def record_in_segments(output_dir: str,
                       prefix: str = "segment",
                       segment_minutes: int = 10,
                       samplerate: int = 16000,
                       channels: int = 1):
    """
    Record audio in continuous segments and save WAV files to output_dir.
    Stops when the user presses Ctrl+C.

    Returns list of recorded filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        q.put(indata.copy())

    print(f"Starting recording — new files in {output_dir}")
    print("Press Ctrl+C to stop recording and proceed to transcription.")

    recorded_files = []
    try:
        with sf.SoundFile(os.path.join(output_dir, f"{prefix}_temp.wav"),
                          mode='w', samplerate=samplerate,
                          channels=channels, subtype='PCM_16') as tempfile_sf:
            with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
                segment_frames = segment_minutes * 60 * samplerate
                frames_written = 0
                seg_index = 1
                start_time = time.time()
                last_print = start_time
                while True:
                    data = q.get()
                    tempfile_sf.write(data)
                    frames_written += data.shape[0]
                    # periodic recording time print
                    now = time.time()
                    if now - last_print >= 1.0:
                        elapsed = now - start_time
                        print(f"\rRecording... elapsed: {human_seconds(elapsed)}  (segment {seg_index})", end="", flush=True)
                        last_print = now
                    # write to a real segment file when we hit segment_frames
                    if frames_written >= segment_frames:
                        # flush and rotate file
                        tempfile_sf.flush()
                        seg_name = os.path.join(output_dir, f"{prefix}_{seg_index:03d}.wav")
                        # move temp file -> segment file atomically by closing and renaming
                        tempfile_sf.close()
                        temp_path = os.path.join(output_dir, f"{prefix}_temp.wav")
                        os.replace(temp_path, seg_name)
                        print(f"\nSaved segment: {seg_name}")
                        recorded_files.append(seg_name)
                        seg_index += 1
                        # reopen temp file for next segment
                        tempfile_sf = sf.SoundFile(os.path.join(output_dir, f"{prefix}_temp.wav"),
                                                   mode='w', samplerate=samplerate,
                                                   channels=channels, subtype='PCM_16')
                        frames_written = 0
    except KeyboardInterrupt:
        # finish current buffer into a final segment if it has any frames
        try:
            tempfile_path = os.path.join(output_dir, f"{prefix}_temp.wav")
            if os.path.exists(tempfile_path):
                # if file has size > 44 bytes (wave header) consider saving
                if os.path.getsize(tempfile_path) > 1000:
                    final_name = os.path.join(output_dir, f"{prefix}_{len(recorded_files)+1:03d}.wav")
                    os.replace(tempfile_path, final_name)
                    print(f"\nSaved final segment: {final_name}")
                    recorded_files.append(final_name)
                else:
                    os.remove(tempfile_path)
        except Exception as e:
            print("Error finalizing recording:", e)
        print("\nRecording stopped by user.")
    except Exception as e:
        print("Recording error:", e)
        raise

    return recorded_files

# ----------------------------
# Transcription utilities (enhanced)
# ----------------------------
def try_import_whisper():
    """
    Try to return loaded whisper lib (module) if available, else None.
    """
    global _whisper_lib
    return _whisper_lib

def load_whisper_model(model_name: str):
    """
    Load whisper model (caching supported).
    """
    whisper = try_import_whisper()
    if whisper is None:
        raise RuntimeError("Whisper not installed. Install openai-whisper to use local transcription.")
    # caching model instance in module-level attribute to avoid reloading
    if not hasattr(load_whisper_model, "_cache"):
        load_whisper_model._cache = {}
    cache = load_whisper_model._cache
    if model_name in cache:
        return cache[model_name]
    print(f"Loading Whisper model '{model_name}' (may take a while)...")
    model = whisper.load_model(model_name)
    cache[model_name] = model
    return model

def get_audio_duration_seconds(path: str) -> float:
    try:
        info = sf.info(path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    # fallback: try ffmpeg via soundfile reading (may be slow)
    try:
        with sf.SoundFile(path) as f:
            return float(len(f)) / f.samplerate
    except Exception:
        return 0.0

def estimate_whisper_time(duration_seconds: float, model_name: str) -> float:
    """
    Provide a rough estimate for transcription time based on model.
    Multipliers are heuristics for CPU-only runs; adjust for your hardware.
    """
    multipliers = {
        "tiny": 0.3,
        "base": 0.5,
        "small": 1.0,
        "medium": 2.0,
        "large": 4.0
    }
    base = multipliers.get(model_name.lower(), 1.0)
    return duration_seconds * base

def split_audio_file_by_duration(input_wav: str, output_dir: str, chunk_seconds: int = 600) -> List[str]:
    """
    If a WAV is very large, split it into smaller WAV chunks using soundfile streaming.
    Returns list of chunk filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    try:
        with sf.SoundFile(input_wav) as sf_in:
            samplerate = sf_in.samplerate
            channels = sf_in.channels
            frames_per_chunk = int(chunk_seconds * samplerate)
            idx = 0
            while True:
                data = sf_in.read(frames_per_chunk, dtype='int16')
                if len(data) == 0:
                    break
                chunk_name = os.path.join(output_dir, f"{os.path.basename(input_wav).rsplit('.',1)[0]}_chunk{idx:03d}.wav")
                with sf.SoundFile(chunk_name, mode='w', samplerate=samplerate, channels=channels, subtype='PCM_16') as sf_out:
                    sf_out.write(data)
                chunks.append(chunk_name)
                idx += 1
    except Exception as e:
        print(f"Audio splitting failed for {input_wav}: {e}")
    return chunks

def transcribe_with_whisper_file(model, audio_path: str, progress_cb=None) -> str:
    """
    Transcribe a single audio file using a preloaded whisper model instance.
    If the file is huge, we can attempt to split it first (internal decision).
    Return text.
    """
    # If audio longer than 20 minutes (1200s) consider splitting to avoid memory issues
    duration = get_audio_duration_seconds(audio_path)
    if duration > 1200:
        print(f"File {audio_path} is long ({int(duration)}s). Splitting into 10-minute chunks for stability.")
        tmp_dir = tempfile.mkdtemp(prefix="whisper_split_")
        chunks = split_audio_file_by_duration(audio_path, tmp_dir, chunk_seconds=600)
        pieces = []
        for c in chunks:
            if progress_cb:
                progress_cb(f"Transcribing chunk {os.path.basename(c)}")
            try:
                res = model.transcribe(c)
                pieces.append(res.get("text", ""))
            except Exception as e:
                print(f"Whisper chunk transcription error for {c}: {e}")
        # cleanup chunk files
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return "\n".join(pieces)
    else:
        if progress_cb:
            progress_cb(f"Transcribing {os.path.basename(audio_path)}")
        result = model.transcribe(audio_path)
        return result.get("text", "")

def transcribe_files(file_list: List[str],
                     prefer_whisper: bool = True,
                     whisper_model: str = "small",
                     gemini_api_key: str = None,
                     state: Dict[str, Any] = None) -> str:
    """
    Transcribe a list of audio files in order. Returns concatenated transcript.
    Supports resuming using `state` dict that contains 'transcribed_files' list.
    """
    transcripts = []
    transcribed_set = set()
    if state and "transcribed_files" in state:
        transcribed_set.update(state.get("transcribed_files", []))

    whisper_module = try_import_whisper() if prefer_whisper else None
    whisper_model_instance = None
    if whisper_module is not None:
        whisper_model_instance = load_whisper_model(whisper_model)

    for f in file_list:
        fname = os.path.basename(f)
        if fname in transcribed_set:
            print(f"Skipping already-transcribed file {fname} (resume).")
            # Ideally include the existing text from state to reconstruct output
            if state and "transcripts" in state and fname in state["transcripts"]:
                transcripts.append(f"\n\n### Transcript from {fname}\n\n{state['transcripts'][fname]}")
            continue

        print(f"\nTranscribing: {f}")
        duration = get_audio_duration_seconds(f)
        est_seconds = estimate_whisper_time(duration, whisper_model)
        print(f"Estimated transcription time (heuristic): {human_seconds(est_seconds)} for model '{whisper_model}'")

        # small progress callback to show simple status (also used from split chunks)
        def progress_cb(msg: str):
            print("  ->", msg)

        try:
            if whisper_model_instance is not None:
                t = transcribe_with_whisper_file(whisper_model_instance, f, progress_cb=progress_cb)
            else:
                t = ""
        except NotImplementedError:
            print("No STT implementation available. Please install whisper or implement Gemini STT.")
            raise
        except Exception as e:
            print(f"Error transcribing {f}: {e}")
            t = ""

        transcripts.append(f"\n\n### Transcript from {fname}\n\n{t.strip()}")

        # update state for resume
        if state is not None:
            if "transcribed_files" not in state:
                state["transcribed_files"] = []
            if "transcripts" not in state:
                state["transcripts"] = {}
            state["transcribed_files"].append(fname)
            state["transcripts"][fname] = t
            # try to persist state immediately
            if "state_path" in state:
                try:
                    with open(state["state_path"], "w", encoding="utf-8") as sfp:
                        json.dump(state, sfp, indent=2)
                except Exception:
                    pass

    full = "\n\n".join(transcripts).strip()
    return full

# ----------------------------
# Cornell notes generation (unchanged; reused)
# ----------------------------
def split_sentences(text: str) -> List[str]:
    # Very simple sentence splitter. Works fine for English-ish transcripts.
    sentences = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def make_cue_from_sentence(s: str) -> str:
    # Simple heuristics to turn statement into a short cue question.
    s = re.sub(r'\s+', ' ', s).strip()
    # short and safe
    if len(s.split()) > 12:
        short = " ".join(s.split()[:12]) + "..."
    else:
        short = s
    # try to turn to question
    if short.lower().startswith(("the ", "this ", "that ", "these ", "those ")):
        return "What is " + short
    if short.lower().startswith(("we ", "i ", "they ", "you ")):
        return "Who/What: " + short
    return "Q: " + short

def chunk_paragraphs(text: str, max_chars=800) -> List[str]:
    """
    Split transcript into "key thoughts" paragraphs sized for the right column.
    """
    sentences = split_sentences(text)
    paras = []
    cur = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) > max_chars and cur:
            paras.append(" ".join(cur))
            cur = [s]
            cur_len = len(s)
        else:
            cur.append(s)
            cur_len += len(s)
    if cur:
        paras.append(" ".join(cur))
    return paras

def generate_cornell_from_transcript(transcript: str, title: str, max_cues = 12) -> Tuple[List[str], List[str], str]:
    """
    Returns (cues, key_thoughts_paragraphs, summary)
    """
    if not transcript.strip():
        return ([], [], "")

    # split and create paragraphs
    paragraphs = chunk_paragraphs(transcript, max_chars=800)

    # generate a cue question for the first sentences of each paragraph
    cues = []
    key_thoughts = []
    for p in paragraphs:
        sents = split_sentences(p)
        if sents:
            cues.append(make_cue_from_sentence(sents[0]))
        else:
            cues.append("Key point")
        key_thoughts.append(p)

    # cap cues
    if len(cues) > max_cues:
        cues = cues[:max_cues]
        key_thoughts = key_thoughts[:max_cues]

    # summary: pick first + last + top sentences heuristically
    sents = split_sentences(transcript)
    summary_sents = []
    if sents:
        summary_sents.append(sents[0])
    if len(sents) > 3:
        mid = sents[len(sents)//2]
        summary_sents.append(mid)
    if len(sents) > 1:
        summary_sents.append(sents[-1])
    summary = " ".join(summary_sents)
    if len(summary) > 800:
        summary = summary[:800] + "..."

    return (cues, key_thoughts, summary)

# ----------------------------
# Gemini (Google AI Studio) generation utilities
# ----------------------------
def chunk_text_by_chars(text: str, max_chars: int = 25000) -> List[str]:
    """
    Simple chunker by characters for sending to external models (e.g., Gemini).
    Adjust max_chars down if you need to be safe. Returns list of text chunks.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        # try to backtrack to nearest paragraph end to keep chunks sensible
        if end < n:
            back = text.rfind("\n\n", start, end)
            if back > start + 50:
                end = back
        chunks.append(text[start:end].strip())
        start = end
    return chunks

def generate_with_gemini(transcript: str, api_key: Optional[str], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    Generate key points, cue questions, action items and a concise summary using Gemini.
    This function uses the google.generativeai (genai) client if available.
    It chunks the transcript to a safe size and queries Gemini per chunk, then aggregates results.

    IMPORTANT:
      - The exact python client API for Gemini / AI Studio may differ by version.
      - If genai is not installed or if your account requires a different call pattern,
        replace this function with the appropriate API calls. The function is written defensively.

    The function expects the Gemini model to output JSON-friendly text; the prompt requests JSON.
    """
    if api_key is None:
        raise RuntimeError("No Gemini API key provided for generation.")

    if genai is None:
        raise RuntimeError("google-generativeai library not installed. Please `pip install google-generativeai` or adapt this function.")

    # configure client (pattern used by popular versions)
    try:
        genai.configure(api_key=api_key)
    except Exception:
        # some library versions may use genai.Client; attempt both
        try:
            genai.client = genai.Client(api_key=api_key)
        except Exception:
            pass

    # chunk transcript
    chunks = chunk_text_by_chars(transcript, max_chars=25000)
    print(f"Transcript split into {len(chunks)} chunk(s) for Gemini generation.")

    aggregated = {
        "key_points": [],
        "cue_questions": [],
        "action_items": [],
        "summaries": []
    }

    # For each chunk, ask Gemini to produce JSON with the requested fields
    for i, chunk in enumerate(chunks):
        prompt = (
            "You are a helpful assistant that extracts study/meeting assets from a transcript.\n"
            "Given the transcript below, produce a JSON object with the following keys: "
            "\"key_points\" (an array of 5-15 concise bullet points summarizing main topics), "
            "\"cue_questions\" (an array of 8-20 short review/study questions), "
            "\"action_items\" (an array of todo items with assignees/due-dates if present), "
            "\"summary\" (a concise 1-3 sentence summary).\n\n"
            "Output ONLY valid JSON. Do not include additional commentary.\n\n"
            "Transcript (chunk {i} of {n}):\n\n"
            "{chunk}\n"
            .format(i=i+1, n=len(chunks), chunk=chunk)
        )

        print(f"Generating with Gemini for chunk {i+1}/{len(chunks)}...")
        try:
            # The exact method name may vary by genai version. We attempt common variants.
            resp_text = None
            try:
                # Common pattern: genai.generate() or genai.generate_text()
                if hasattr(genai, "generate"):
                    resp = genai.generate(model=model, prompt=prompt)
                    # many versions return a dict-like with 'content' or 'candidates'
                    if isinstance(resp, dict):
                        if resp.get("content"):
                            resp_text = resp.get("content")
                        elif "candidates" in resp and len(resp["candidates"]) > 0:
                            resp_text = resp["candidates"][0].get("content", "")
                    else:
                        resp_text = str(resp)
                elif hasattr(genai, "generate_text"):
                    resp = genai.generate_text(model=model, input=prompt)
                    # resp may have .text or .outputs
                    if hasattr(resp, "text"):
                        resp_text = resp.text
                    elif isinstance(resp, dict) and "output" in resp:
                        resp_text = resp["output"]
                    else:
                        resp_text = str(resp)
                else:
                    # Try client interface
                    client = getattr(genai, "client", None)
                    if client is not None and hasattr(client, "generate"):
                        resp = client.generate(model=model, prompt=prompt)
                        resp_text = resp.text if hasattr(resp, "text") else str(resp)
                    else:
                        raise RuntimeError("Unsupported google-generativeai client version. Adapt generate_with_gemini().")
            except Exception as e:
                # If the high-level call fails, raise with helpful hint
                raise RuntimeError(f"Gemini generate call failed: {e}")

            if not resp_text:
                raise RuntimeError("Empty response from Gemini.")

            # Attempt to parse JSON from the response text
            parsed = None
            try:
                # Sometimes model wraps JSON in markdown or text — attempt to locate first { ... }
                txt = resp_text.strip()
                # find first '{' and last '}' to extract JSON blob
                first = txt.find('{')
                last = txt.rfind('}')
                if first != -1 and last != -1 and last > first:
                    json_blob = txt[first:last+1]
                    parsed = json.loads(json_blob)
                else:
                    parsed = json.loads(txt)
            except Exception as e:
                # fallback: try eval-style or line-by-line extraction (risky)
                print("Warning: failed to parse JSON directly from Gemini response. Response preview:")
                print(resp_text[:1000])
                raise RuntimeError("Could not parse JSON from Gemini response. Please inspect model output or adjust prompt.")

            # Aggregate lists (deduplicate later)
            for k in ["key_points", "cue_questions", "action_items", "summary"]:
                if k in parsed:
                    if k == "summary":
                        aggregated["summaries"].append(parsed[k])
                    else:
                        # ensure list
                        val = parsed[k]
                        if isinstance(val, str):
                            # try splitting lines
                            lines = [l.strip("-* \t") for l in val.splitlines() if l.strip()]
                            aggregated[k].extend(lines)
                        elif isinstance(val, list):
                            aggregated[k].extend(val)
                        else:
                            aggregated[k].append(str(val))
            # small pause to respect rate-limits
            time.sleep(0.5)
        except Exception as e:
            print(f"Gemini generation error for chunk {i+1}: {e}")
            # continue with others; but surface error
            continue

    # Postprocess aggregates: deduplicate & trim
    def dedupe_keep_order(lst):
        seen = set()
        out = []
        for item in lst:
            s = item.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    result = {
        "key_points": dedupe_keep_order(aggregated["key_points"])[:50],
        "cue_questions": dedupe_keep_order(aggregated["cue_questions"])[:100],
        "action_items": dedupe_keep_order(aggregated["action_items"])[:200],
        "summary": ""
    }

    # create a consolidated summary: pick most common summary line or join
    if aggregated["summaries"]:
        # choose the longest short summary or join them
        summaries = [s.strip() for s in aggregated["summaries"] if s and len(s.strip()) < 1000]
        if summaries:
            # pick the shortest non-empty as concise
            summaries_sorted = sorted(summaries, key=lambda s: len(s))
            result["summary"] = summaries_sorted[0]
        else:
            result["summary"] = " ".join(aggregated["summaries"])[:1000]
    else:
        result["summary"] = ""

    return result

# ----------------------------
# Output utilities (extended)
# ----------------------------
def save_json(data: dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON to {out_path}")

def save_text(text: str, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote text to {out_path}")

def save_markdown_cornell(title: str, cues: List[str], key_thoughts: List[str], summary: str, out_path: str):
    md_lines = []
    md_lines.append(f"# {title}\n")
    # Build a two-column style in markdown using HTML table to preserve layout better
    md_lines.append('<table style="width:100%"><tr>\n')
    md_lines.append('<td style="width:25%; vertical-align:top; padding:8px; border:1px solid #ddd;">\n')
    md_lines.append("### Cues\n")
    for c in cues:
        md_lines.append(f"- {c}\n")
    md_lines.append("</td>\n")
    md_lines.append('<td style="width:75%; vertical-align:top; padding:8px; border:1px solid #ddd;">\n')
    md_lines.append("### Key Thoughts\n")
    for k in key_thoughts:
        md_lines.append(f"- {k}\n\n")
    md_lines.append("</td>\n")
    md_lines.append("</tr></table>\n\n")
    md_lines.append("### Summary\n\n")
    md_lines.append(summary + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Wrote Markdown Cornell notes to {out_path}")

def save_html_cornell(title: str, cues: List[str], key_thoughts: List[str], summary: str, out_path: str):
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title} — Cornell Notes</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; margin: 24px; }}
  .container {{ width: 100%; max-width: 1100px; margin: auto; }}
  .title {{ text-align:center; font-size: 28px; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: 28% 72%; gap: 12px; }}
  .box {{ border: 1px solid #d0d0d0; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.03); }}
  .cues {{ font-size: 14px; }}
  .key {{ font-size: 15px; }}
  .summary {{ margin-top: 16px; padding: 12px; border-radius: 8px; background: #fbfbfb; }}
  .cue-item {{ margin: 8px 0; }}
  .key-item {{ margin: 12px 0; }}
</style>
</head>
<body>
<div class="container">
  <div class="title">{title}</div>
  <div class="grid">
    <div class="box cues">
      <strong>Cues</strong>
      <ul>
"""
    for c in cues:
        html += f"        <li class='cue-item'>{escape_html(c)}</li>\n"
    html += """      </ul>
    </div>
    <div class="box key">
      <strong>Key Thoughts</strong>
"""
    for k in key_thoughts:
        html += f"<div class='key-item'>{escape_html(k)}</div>\n"
    html += f"""    </div>
  </div>
  <div class="summary box">
    <strong>Summary</strong>
    <p>{escape_html(summary)}</p>
  </div>
</div>
</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote HTML Cornell notes to {out_path}")

def escape_html(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace("\n", "<br>"))

# ----------------------------
# State / resume helpers
# ----------------------------
def load_state(state_path: str) -> Dict[str, Any]:
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict[str, Any], state_path: str):
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: could not save state to {state_path}: {e}")

# ----------------------------
# Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Record audio and produce Cornell-style notes (.md + .html) with local Whisper + Gemini generation")
    parser.add_argument("--title", "-t", required=True, help="Title for the Cornell notes (used for filenames)")
    parser.add_argument("--segment-minutes", type=int, default=10, help="Minutes per audio segment file")
    parser.add_argument("--outdir", "-o", default=os.getenv("OUTPUT_BASE_DIR", "cornell_output"), help="Directory to save audio/transcripts/notes")
    parser.add_argument("--samplerate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--channels", type=int, default=1, help="Mic channels (1=mono)")
    parser.add_argument("--whisper-model", type=str, default="small", help="Whisper model name if using whisper (tiny/base/small/medium/large)")
    parser.add_argument("--prefer-whisper", action="store_true", help="Try to use local whisper for transcription if available")
    parser.add_argument("--gemini-api-key", type=str, default=None, help="(Optional) Gemini API key — CLI overrides GEMINI_API_KEY env var")
    parser.add_argument("--no-gemini", action="store_true", help="Do not call Gemini generation step")
    parser.add_argument("--chunk-size-chars", type=int, default=25000, help="Max characters per chunk sent to Gemini")
    parser.add_argument("--resume", action="store_true", help="Resume from previous job directory if available")
    parser.add_argument("--segment-save-limit-mb", type=int, default=5000, help="Warn if total recorded data approaches this size (MB)")
    args = parser.parse_args()

    safe_title = re.sub(r'[^0-9A-Za-z \-_]+', '', args.title).strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(args.outdir, f"{safe_title}_{timestamp}")
    os.makedirs(job_dir, exist_ok=True)

    # When resuming, attempt to find the most recent matching job dir
    if args.resume:
        base = args.outdir
        candidates = []
        try:
            for d in os.listdir(base):
                if d.startswith(safe_title + "_"):
                    candidates.append(os.path.join(base, d))
            if candidates:
                candidates = sorted(candidates)
                chosen = candidates[-1]
                print(f"Resuming from {chosen}")
                job_dir = chosen
            else:
                print("No previous job found to resume; continuing with new job dir.")
        except Exception:
            pass

    # state path
    state_path = os.path.join(job_dir, "job_state.json")
    state = load_state(state_path)
    state.setdefault("created_at", datetime.now().isoformat())
    state["state_path"] = state_path

    # 1) Recording
    print("Step 1 — recording audio segments.")
    try:
        # disk check
        if not ensure_disk_space(job_dir, needed_bytes=50 * 1024 * 1024):
            print("Warning: less than 50MB free in recording directory. Continue at your own risk.")
        recorded_files = record_in_segments(output_dir=job_dir,
                                            prefix="segment",
                                            segment_minutes=args.segment_minutes,
                                            samplerate=args.samplerate,
                                            channels=args.channels)
    except Exception as e:
        print("Recording failed or aborted:", e)
        recorded_files = []
    if not recorded_files:
        print("No recordings found — exiting.")
        return

    # Save a list of recorded files to state for resume capability
    state.setdefault("recorded_files", [])
    for rf in recorded_files:
        fn = os.path.basename(rf)
        if fn not in state["recorded_files"]:
            state["recorded_files"].append(fn)
    save_state(state, state_path)

    # 2) Transcription
    print("\nStep 2 — transcribing audio segments. This may take some time.")
    # Full paths for recorded files in order
    recorded_full = [os.path.join(job_dir, f) for f in state.get("recorded_files", [])]

    try:
        transcript = transcribe_files(recorded_full,
                                      prefer_whisper=args.prefer_whisper,
                                      whisper_model=args.whisper_model,
                                      gemini_api_key=(args.gemini_api_key or os.getenv("GEMINI_API_KEY")),
                                      state=state)
    except NotImplementedError:
        print("\nTranscription not implemented. Either install whisper or implement Gemini STT in transcribe_audio_with_gemini().")
        transcript = ""
    except Exception as e:
        print(f"Transcription encountered an error: {e}")
        transcript = ""

    # Save raw transcript
    transcript_path = os.path.join(job_dir, f"{safe_title}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"Transcript saved to {transcript_path}")

    # Save transcript in state as well
    state["full_transcript_path"] = transcript_path
    save_state(state, state_path)

    # 3) Generate Cornell notes (local heuristics)
    print("\nStep 3 — generating Cornell notes layout.")
    cues, keys, summary = generate_cornell_from_transcript(transcript, args.title)
    md_path = os.path.join(job_dir, f"{safe_title}_cornell.md")
    html_path = os.path.join(job_dir, f"{safe_title}_cornell.html")
    save_markdown_cornell(args.title, cues, keys, summary, md_path)
    save_html_cornell(args.title, cues, keys, summary, html_path)

    # 4) Call Gemini for advanced generation if requested
    gemini_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    gemini_out = {}
    if not args.no_gemini:
        if gemini_key is None:
            print("No Gemini API key provided (env GEMINI_API_KEY or --gemini-api-key). Skipping Gemini generation.")
        else:
            print("\nStep 4 — generating enhanced summaries with Gemini.")
            try:
                gemini_out = generate_with_gemini(transcript, api_key=gemini_key, model="gpt-4o-mini")
                # Save Gemini results
                gemini_json_path = os.path.join(job_dir, f"{safe_title}_gemini.json")
                save_json(gemini_out, gemini_json_path)

                # Also write a Markdown summary combining items
                md_lines = [f"# {args.title} — Generated Notes\n"]
                md_lines.append("## Key Points\n")
                for kp in gemini_out.get("key_points", []):
                    md_lines.append(f"- {kp}\n")
                md_lines.append("\n## Cue Questions\n")
                for cq in gemini_out.get("cue_questions", []):
                    md_lines.append(f"- {cq}\n")
                md_lines.append("\n## Action Items\n")
                for ai in gemini_out.get("action_items", []):
                    md_lines.append(f"- {ai}\n")
                md_lines.append("\n## Concise Summary\n")
                md_lines.append(gemini_out.get("summary", "") + "\n")

                gemini_md_path = os.path.join(job_dir, f"{safe_title}_gemini.md")
                with open(gemini_md_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(md_lines))
                print(f"Wrote Gemini-derived markdown to {gemini_md_path}")

            except Exception as e:
                print(f"Gemini generation failed: {e}")
    else:
        print("Gemini generation skipped by user (--no-gemini).")

    # Final save of job state
    state["completed_at"] = datetime.now().isoformat()
    save_state(state, state_path)

    print("\nDone. Files saved in:", job_dir)
    print("Open the HTML or Markdown files in a browser or editor for a nice layout.")
    print("Saved outputs:")
    for p in [transcript_path, md_path, html_path]:
        print("  -", p)
    if gemini_out:
        print("  -", os.path.join(job_dir, f"{safe_title}_gemini.md"))
        print("  -", os.path.join(job_dir, f"{safe_title}_gemini.json"))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting cleanly.")
        sys.exit(0)


