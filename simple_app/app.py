from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from faster_whisper import WhisperModel

BASE_DIR = Path(os.getenv("DATA_DIR", "/data"))
UPLOAD_DIR = BASE_DIR / "uploads"
JOB_DIR = BASE_DIR / "jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "5"))
MIN_CLIP_SECONDS = int(os.getenv("MIN_CLIP_SECONDS", "25"))
MAX_CLIP_SECONDS = int(os.getenv("MAX_CLIP_SECONDS", "55"))
LLM_API_URL = os.getenv("LLM_API_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

app = FastAPI(title="igedits Simple")
jobs: dict[str, dict[str, Any]] = {}
_model: WhisperModel | None = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=max(1, int(os.getenv("WHISPER_CPU_THREADS", "2"))),
                num_workers=1,
            )
    return _model


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:])


def sec_to_srt(value: float) -> str:
    ms = max(0, int(value * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def transcribe(video: Path) -> tuple[str, list[dict[str, Any]]]:
    model = get_model()
    segments, _ = model.transcribe(
        str(video),
        language=WHISPER_LANGUAGE,
        beam_size=1,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    output: list[dict[str, Any]] = []
    full_text: list[str] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        words = []
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            words.append({"start": float(word.start), "end": float(word.end), "text": word.word.strip()})
        output.append({"start": float(segment.start), "end": float(segment.end), "text": text, "words": words})
        full_text.append(text)
    return " ".join(full_text), output


def heuristic_clips(segments: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not segments:
        return []
    candidates: list[dict[str, Any]] = []
    start_index = 0
    while start_index < len(segments):
        start = segments[start_index]["start"]
        end_index = start_index
        while end_index + 1 < len(segments) and segments[end_index]["end"] - start < MIN_CLIP_SECONDS:
            end_index += 1
        while end_index + 1 < len(segments) and segments[end_index + 1]["end"] - start <= MAX_CLIP_SECONDS:
            end_index += 1
        end = segments[end_index]["end"]
        text = " ".join(item["text"] for item in segments[start_index : end_index + 1])
        score = len(text)
        score += 80 * len(re.findall(r"[!?]", text))
        score += 50 * len(re.findall(r"\b(важно|главн|ошибк|секрет|почему|как|нельзя|нужно|результат|деньги)\w*", text.lower()))
        candidates.append({"start": start, "end": end, "title": text[:80], "score": score})
        start_index = max(start_index + 1, end_index)
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(not (candidate["end"] <= item["start"] or candidate["start"] >= item["end"]) for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    return sorted(selected, key=lambda item: item["start"])


def llm_clips(segments: list[dict[str, Any]], count: int) -> list[dict[str, Any]] | None:
    if not (LLM_API_URL and LLM_MODEL):
        return None
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    prompt = (
        "Ты редактор коротких вертикальных видео. Выбери самые сильные законченные фрагменты из русской расшифровки. "
        f"Верни строго JSON-массив из максимум {count} объектов: "
        '{"start": число секунд, "end": число секунд, "title": "короткий заголовок"}. '
        f"Длительность каждого фрагмента от {MIN_CLIP_SECONDS} до {MAX_CLIP_SECONDS} секунд. "
        "Не добавляй markdown. Расшифровка:\n" + transcript
    )
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    try:
        response = httpx.post(LLM_API_URL.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        data = json.loads(content)
        valid = []
        max_end = segments[-1]["end"] if segments else 0
        for item in data:
            start = max(0.0, float(item["start"]))
            end = min(max_end, float(item["end"]))
            if MIN_CLIP_SECONDS <= end - start <= MAX_CLIP_SECONDS + 5:
                valid.append({"start": start, "end": end, "title": str(item.get("title", "Клип"))[:100]})
        return valid[:count] or None
    except Exception:
        return None


def write_srt(path: Path, segments: list[dict[str, Any]], clip_start: float, clip_end: float) -> None:
    cues = []
    index = 1
    for segment in segments:
        if segment["end"] <= clip_start or segment["start"] >= clip_end:
            continue
        words = [w for w in segment["words"] if w["end"] > clip_start and w["start"] < clip_end]
        if not words:
            cues.append((max(0, segment["start"] - clip_start), min(clip_end - clip_start, segment["end"] - clip_start), segment["text"]))
            continue
        for offset in range(0, len(words), 5):
            group = words[offset : offset + 5]
            cues.append((max(0, group[0]["start"] - clip_start), min(clip_end - clip_start, group[-1]["end"] - clip_start), " ".join(w["text"] for w in group)))
    with path.open("w", encoding="utf-8") as handle:
        for start, end, text in cues:
            handle.write(f"{index}\n{sec_to_srt(start)} --> {sec_to_srt(end)}\n{text.strip()}\n\n")
            index += 1


def render_clip(video: Path, out: Path, srt: Path, start: float, end: float) -> None:
    duration = end - start
    subtitle_path = str(srt).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    vf = (
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        f"subtitles='{subtitle_path}':force_style='FontName=DejaVu Sans,FontSize=22,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,"
        "Alignment=2,MarginV=150'"
    )
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
        "-vf", vf, "-c:v", "libx264", "-preset", os.getenv("FFMPEG_PRESET", "veryfast"),
        "-crf", os.getenv("FFMPEG_CRF", "23"), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ])


def process_job(job_id: str, video: Path, clip_count: int) -> None:
    job = jobs[job_id]
    folder = JOB_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    try:
        job.update(status="transcribing", progress=10, message="Распознаю русскую речь")
        full_text, segments = transcribe(video)
        (folder / "transcript.txt").write_text(full_text, encoding="utf-8")
        (folder / "transcript.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
        job.update(status="analyzing", progress=45, message="Выбираю лучшие моменты")
        clips = llm_clips(segments, clip_count) or heuristic_clips(segments, clip_count)
        if not clips:
            raise RuntimeError("Не удалось найти подходящие фрагменты")
        outputs = []
        for index, clip in enumerate(clips, start=1):
            job.update(status="rendering", progress=45 + int(45 * index / len(clips)), message=f"Рендерю клип {index} из {len(clips)}")
            srt = folder / f"clip_{index:02}.srt"
            out = folder / f"clip_{index:02}.mp4"
            write_srt(srt, segments, clip["start"], clip["end"])
            render_clip(video, out, srt, clip["start"], clip["end"])
            outputs.append({"name": out.name, "title": clip.get("title", f"Клип {index}"), "url": f"/jobs/{job_id}/files/{out.name}"})
        archive = folder / "clips.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in outputs:
                zf.write(folder / item["name"], item["name"])
            zf.write(folder / "transcript.txt", "transcript.txt")
        job.update(status="done", progress=100, message="Готово", clips=outputs, zip_url=f"/jobs/{job_id}/files/clips.zip")
    except Exception as exc:
        job.update(status="error", progress=100, message=str(exc))


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html lang='ru'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>igedits Simple</title><style>body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 18px;background:#111;color:#eee}main{background:#1b1b1b;padding:28px;border-radius:18px}input,button{font:inherit;margin:8px 0}button{padding:12px 18px;border:0;border-radius:10px;cursor:pointer}.bar{height:12px;background:#333;border-radius:9px;overflow:hidden}.fill{height:100%;background:#eee;width:0}.clip{padding:12px 0;border-top:1px solid #333}a{color:#fff}</style></head><body><main><h1>Видео → короткие клипы</h1><p>Локальная русская расшифровка, выбор моментов, вертикальный формат и субтитры.</p><form id='f'><input type='file' name='video' accept='video/*' required><br><label>Количество клипов <input type='number' name='clip_count' value='5' min='1' max='10'></label><br><button>Загрузить и обработать</button></form><p id='msg'></p><div class='bar'><div id='fill' class='fill'></div></div><div id='results'></div></main><script>const f=document.getElementById('f'),msg=document.getElementById('msg'),fill=document.getElementById('fill'),results=document.getElementById('results');f.onsubmit=async e=>{e.preventDefault();results.innerHTML='';msg.textContent='Загрузка...';const r=await fetch('/jobs',{method:'POST',body:new FormData(f)});const d=await r.json();if(!r.ok){msg.textContent=d.detail||'Ошибка';return}poll(d.id)};async function poll(id){const r=await fetch('/jobs/'+id),d=await r.json();msg.textContent=d.message;fill.style.width=d.progress+'%';if(d.status==='done'){results.innerHTML='<p><a href="'+d.zip_url+'">Скачать все клипы ZIP</a></p>'+d.clips.map(x=>'<div class="clip"><b>'+x.title+'</b><br><a href="'+x.url+'">Скачать MP4</a></div>').join('');return}if(d.status==='error')return;setTimeout(()=>poll(id),2000)}</script></body></html>"""


@app.post("/jobs")
def create_job(video: UploadFile = File(...), clip_count: int = Form(5)) -> dict[str, str]:
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(400, "Неподдерживаемый формат видео")
    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    with path.open("wb") as handle:
        shutil.copyfileobj(video.file, handle)
    count = min(max(1, clip_count), min(10, MAX_CLIPS))
    jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0, "message": "В очереди", "clips": []}
    threading.Thread(target=process_job, args=(job_id, path, count), daemon=True).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(404, "Задание не найдено")
    return jobs[job_id]


@app.get("/jobs/{job_id}/files/{filename}")
def get_file(job_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    path = JOB_DIR / job_id / safe_name
    if not path.is_file():
        raise HTTPException(404, "Файл не найден")
    return FileResponse(path, filename=safe_name)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
