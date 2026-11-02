"""FastAPI server. Run:  uvicorn crag_app.api:app --reload"""
import hashlib
import io
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from . import llm, vectorstore, video
from .graph import run_crag

app = FastAPI(title="CRAG API", version="1.0")

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_OCR_PAGES = 1500  # OCR takes a second or two per page
FRONTEND = Path(__file__).parent / "static" / "index.html"
NAMESPACE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _check_namespace(ns: str) -> str:
    if not NAMESPACE_RE.match(ns):
        raise HTTPException(400, "Namespace may only contain letters, numbers, - and _ (max 64 characters)")
    return ns


def _slug(filename: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(filename).stem).strip("-")[:64]
    return s or "documents"


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _duplicate_message(dup: dict) -> str:
    note = (" Only part of it was stored, because that upload was stopped early. "
            f"To upload it again in full, first delete namespace '{dup['namespace']}' in Pinecone."
            if dup.get("partial") else "")
    return (f"This PDF is already in the database: '{dup['source']}' is stored in "
            f"namespace '{dup['namespace']}' ({dup['match']}). Nothing was added.{note}")


@lru_cache
def _ocr_engine():
    from rapidocr_onnxruntime import RapidOCR  # imported lazily: loads the OCR models
    return RapidOCR()


def _ocr_pages(data: bytes, page_numbers: List[int], progress=lambda stage, done, total: None,
               should_stop=lambda: False) -> tuple:
    """OCR the given 1-based pages in order.

    Returns ({page: text} for pages where text was found, number of pages processed).
    Stops early, keeping what it has, as soon as should_stop() is true.
    """
    import numpy as np
    import pymupdf

    engine = _ocr_engine()
    found = {}
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for i, n in enumerate(page_numbers):
            if should_stop():
                return found, i
            progress("Reading scanned pages with OCR", i, len(page_numbers))
            pix = doc[n - 1].get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            result, _ = engine(img)
            text = "\n".join(line[1] for line in (result or [])).strip()
            if text:
                found[n] = text
    return found, len(page_numbers)


def _ingest_pdf(data: bytes, filename: str, namespace: str, file_hash: str,
                progress=lambda stage, done, total: None, should_stop=lambda: False) -> dict:
    """Extract text page by page and store it in a namespace (page number kept as metadata)."""
    progress("Reading text from the PDF", 0, 0)
    reader = PdfReader(io.BytesIO(data))
    page_texts = {n: (p.extract_text() or "").strip() for n, p in enumerate(reader.pages, start=1)}

    # Pages with no text layer are scans: read them with OCR.
    blank = [n for n, t in page_texts.items() if not t]
    if len(blank) > MAX_OCR_PAGES:
        raise ValueError(f"{len(blank)} pages need OCR, which is more than the {MAX_OCR_PAGES}-page limit")
    ocr_done, processed = _ocr_pages(data, blank, progress, should_stop) if blank else ({}, 0)
    not_read = len(blank) - processed  # scanned pages skipped because the user pressed Stop
    stopped = not_read > 0
    page_texts.update(ocr_done)

    texts, metadatas = [], []
    for n, text in page_texts.items():
        if text:
            texts.append(text)
            meta = {"source": filename, "page": n, "file_hash": file_hash}
            if n in ocr_done:
                meta["ocr"] = True
            if stopped:
                meta["partial"] = True
            metadatas.append(meta)
    if not texts:
        raise ValueError("Stopped before any text was read, so nothing was stored." if stopped
                         else "No text found, even with OCR (the pages may be blank or too low quality)")
    progress("Storing in Pinecone (embedding the text)", 0, 0)
    return {
        "filename": filename,
        "namespace": namespace,
        "pages": len(reader.pages),
        "pages_with_text": len(texts),
        "pages_ocr": len(ocr_done),
        "stopped": stopped,
        "scanned_pages_not_read": not_read,
        "chunks_stored": vectorstore.add_texts(texts, metadatas, namespace),
    }


# Uploads run as background jobs so a long OCR job doesn't hold a browser request open.
# Jobs live in memory: they are lost if the server restarts.
JOBS: dict = {}


def _run_job(job_id: str, data: bytes, filename: str, namespace: str, file_hash: str) -> None:
    job = JOBS[job_id]

    def progress(stage: str, done: int, total: int) -> None:
        job.update(stage=stage, done=done, total=total)

    try:
        job["result"] = _ingest_pdf(data, filename, namespace, file_hash, progress,
                                    lambda: job["stop_requested"])
        job["status"] = "done"
    except Exception as e:
        job.update(status="error", error=str(e))


def _start_job(data: bytes, filename: str, namespace: str, file_hash: str) -> str:
    for old in list(JOBS)[:-49]:  # keep the 50 most recent
        JOBS.pop(old, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "stage": "Starting", "done": 0, "total": 0,
                    "filename": filename, "namespace": namespace, "stop_requested": False}
    threading.Thread(target=_run_job, args=(job_id, data, filename, namespace, file_hash),
                     daemon=True).start()
    return job_id


async def _read_pdf(file: UploadFile) -> tuple:
    name = file.filename or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are supported")
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF is larger than 50 MB")
    return name, data


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    namespace: Optional[str] = None  # None / empty = search every namespace


class IngestRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[dict]] = None


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(FRONTEND)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return await run_in_threadpool(vectorstore.stats)


@app.get("/documents")
async def documents():
    try:
        return await run_in_threadpool(vectorstore.documents)
    except Exception as e:
        raise HTTPException(500, f"Could not read Pinecone: {e}")


@app.post("/summary")
async def summary():
    try:
        data = await run_in_threadpool(vectorstore.sample_for_summary)
        if not data["documents"]:
            return {**data, "summary": "The knowledge base is empty. Upload a PDF first."}
        text = "\n\n".join(
            f"Document: {d['name']} (namespace {d['namespace']}, {d['chunks']} chunks)\n"
            + "\n---\n".join(d["snippets"])
            for d in data["documents"]
        )
        msg = await llm.kb_summary_chain().ainvoke({"documents": text})
        return {**data, "summary": msg.content}
    except Exception as e:
        raise HTTPException(500, f"Could not summarise: {e}")


@app.post("/ingest")
async def ingest(req: IngestRequest):
    if req.metadatas and len(req.metadatas) != len(req.texts):
        raise HTTPException(400, "metadatas must match texts length")
    n = await run_in_threadpool(vectorstore.add_texts, req.texts, req.metadatas)
    return {"chunks_stored": n}


@app.post("/ingest/pdf/check")
async def check_pdf(file: UploadFile = File(...)):
    """Step 1 of an upload: is this PDF already stored? Nothing is written."""
    name, data = await _read_pdf(file)
    try:
        dup = await run_in_threadpool(vectorstore.find_duplicate, _hash(data), name)
        existing = await run_in_threadpool(vectorstore.namespaces)
    except Exception as e:
        raise HTTPException(500, f"Could not check Pinecone: {e}")
    return {
        "duplicate": dup is not None,
        "message": _duplicate_message(dup) if dup else "",
        "suggested_namespace": _slug(name),
        "namespaces": existing,
    }


@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...), namespace: str = Form(...)):
    """Step 2: re-check for duplicates, then start storing the PDF in the chosen namespace.

    Returns a job id straight away; poll GET /ingest/jobs/{job_id} for progress and the result.
    """
    _check_namespace(namespace)
    name, data = await _read_pdf(file)
    file_hash = _hash(data)
    try:
        dup = await run_in_threadpool(vectorstore.find_duplicate, file_hash, name)
    except Exception as e:
        raise HTTPException(500, f"Could not check Pinecone: {e}")
    if dup:
        raise HTTPException(409, _duplicate_message(dup))
    return {"job_id": _start_job(data, name, namespace, file_hash)}


@app.post("/ingest/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop reading scanned pages; whatever has been read so far is still stored."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    if job["status"] == "running":
        job["stop_requested"] = True
    return job


@app.get("/ingest/jobs/{job_id}")
async def ingest_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (the server may have restarted). Check the Knowledge base list.")
    return job


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)  # an idea, or a whole question + answer
    frames: int = Field(6, ge=2, le=12)                  # number of Stable Diffusion images
    fps: int = Field(12, ge=4, le=30)
    seconds_per_frame: float = Field(1.5, ge=0.5, le=5)  # how long each image stays on screen


def _run_video_job(job_id: str, req: VideoRequest) -> None:
    job = JOBS[job_id]

    def progress(stage: str, done: int, total: int) -> None:
        job.update(stage=stage, done=done, total=total)

    try:
        job["result"] = video.make_video(req.prompt, req.frames, req.fps, req.seconds_per_frame,
                                         job_id, progress)
        job["status"] = "done"
    except Exception as e:
        job.update(status="error", error=str(e))


@app.post("/video")
async def create_video(req: VideoRequest):
    """Text -> storyboard -> Stable Diffusion frames -> MP4. Returns a job id to poll."""
    missing = video.missing_credentials()
    if missing:
        raise HTTPException(400, missing)
    for old in list(JOBS)[:-49]:
        JOBS.pop(old, None)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "running", "stage": "Starting", "done": 0, "total": req.frames}
    threading.Thread(target=_run_video_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/video/jobs/{job_id}")
async def video_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job (the server may have restarted)")
    return job


@app.get("/video/files/{job_id}/{name}")
async def video_file(job_id: str, name: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id) or not video.FILE_RE.match(name):
        raise HTTPException(404, "Not found")
    path = video.OUTPUT_DIR / job_id / name
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path)


@app.post("/ask")
async def ask(req: AskRequest):
    ns = _check_namespace(req.namespace) if req.namespace else None
    try:
        return await run_crag(req.question, ns)
    except Exception as e:
        raise HTTPException(500, f"CRAG pipeline failed: {e}")
