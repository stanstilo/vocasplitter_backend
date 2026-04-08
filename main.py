# 🔥 MUST BE AT TOP LEVEL FOR WINDOWS MULTIPROCESSING
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    import sys
    sys.exit(0)

# === Imports ===
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
from pydantic import BaseModel
import asyncio
import shutil
import os
import uuid
import traceback
import numpy as np
import soundfile as sf
import librosa
import zipfile
import threading
import time
from datetime import datetime, timedelta
from collections import defaultdict


# === Models ===
class StemResponse(BaseModel):
    name: str
    filename: str
    job_id: str

class SeparateResponse(BaseModel):
    success: bool
    job_id: str
    stems: List[StemResponse]
    stem_mode: int

class StatusResponse(BaseModel):
    status: str
    stems: Optional[List[StemResponse]] = None
    stem_mode: Optional[int] = None
    error: Optional[str] = None


# === App ===
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.vocasplitter.com",
        "https://vocasplitter.com",
    ],
    allow_credentials=False,       # must be False when using wildcard methods/headers
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

executor = ThreadPoolExecutor(max_workers=2)
job_store = {}

# Separate directory for zips so they are NOT inside output_dir
ZIP_DIR = "temp_zips"
os.makedirs(ZIP_DIR, exist_ok=True)

STEM_CONFIGS = {
    2: {
        "model": "spleeter:2stems",
        "stems": ["vocals", "accompaniment"],
        "display_names": {
            "vocals": "Vocals",
            "accompaniment": "Instrumental",
        }
    },
    3: {
        "model": "spleeter:4stems",  # 4-stem model outputs vocals/drums/bass/other
        "stems": ["vocals", "drums", "bass", "other"],
        "merge_into_other": ["bass", "other"],  # merge bass+other into one instrumental stem
        "display_names": {
            "vocals": "Vocals",
            "drums": "Drums",
            "other": "Instrumental",
        }
    },
    5: {
        "model": "spleeter:5stems",
        "stems": ["vocals", "drums", "bass", "piano", "other"],
        "display_names": {
            "vocals": "Vocals",
            "drums": "Drums",
            "bass": "Bass",
            "piano": "Piano",
            "other": "Other",
        }
    },
}


# ============================================================
# === Cleanup Functions ===
# ============================================================

def cleanup_job_files(job_id: str):
    """Delete all files and directories associated with a job."""
    try:
        job = job_store.get(job_id, {})

        # Delete temp directory (original uploaded file)
        temp_dir = f"temp_{job_id}"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ Deleted temp directory: {temp_dir}")

        # Delete output directory (processed stem files)
        output_dir = job.get("output_dir", f"processed_{job_id}")
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"✅ Deleted output directory: {output_dir}")

        # Delete zip file if it exists (stored separately in ZIP_DIR)
        zip_path = os.path.join(ZIP_DIR, f"{job_id}_stems.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print(f"✅ Deleted zip file: {zip_path}")

        # Remove job from store if present
        if job_id in job_store:
            del job_store[job_id]
            print(f"✅ Removed job {job_id} from store")

    except Exception as e:
        print(f"❌ Error cleaning up job {job_id}: {str(e)}")


def schedule_cleanup(job_id: str, delay_minutes: int = 10):
    """
    Schedule a job for cleanup after `delay_minutes`.
    Used as the primary cleanup trigger after zip download,
    and as a fallback timer after job completion.
    """
    def delayed_cleanup():
        time.sleep(delay_minutes * 60)
        print(f"🗑️ Running scheduled cleanup for job {job_id} after {delay_minutes} min")
        cleanup_job_files(job_id)

    t = threading.Thread(target=delayed_cleanup, daemon=True)
    t.start()
    print(f"📅 Scheduled cleanup for job {job_id} in {delay_minutes} minutes")


async def cleanup_after_download(job_id: str, delay_seconds: int = 600):
    """
    Schedule cleanup after a download completes.
    The delay ensures FileResponse has finished streaming
    before we delete the files.
    """
    await asyncio.sleep(delay_seconds)
    print(f"🗑️ Cleaning up job {job_id} after download")
    cleanup_job_files(job_id)


def cleanup_all_expired_jobs(max_age_minutes: int = 10):
    """Fallback: clean up any jobs older than max_age_minutes."""
    try:
        now = datetime.now()
        cleaned = 0
        for job_id, job in list(job_store.items()):
            if "created_at" in job:
                age = now - job["created_at"]
                if age > timedelta(minutes=max_age_minutes):
                    print(f"🧹 Cleaning up expired job: {job_id} (age: {age})")
                    cleanup_job_files(job_id)
                    cleaned += 1
        if cleaned:
            print(f"🧹 Cleaned up {cleaned} expired jobs")
        else:
            print("🧹 No expired jobs found")
    except Exception as e:
        print(f"❌ Error in cleanup_all_expired_jobs: {str(e)}")


def cleanup_orphaned_processed_folders():
    """Clean up any processed_* and temp_* folders that don't have corresponding jobs."""
    try:
        cleaned = 0
        if os.path.exists("."):
            for item in os.listdir("."):
                if (item.startswith("processed_") or item.startswith("temp_")) and os.path.isdir(item):
                    job_id = item.replace("processed_", "").replace("temp_", "")
                    if job_id not in job_store:
                        print(f"🧹 Cleaning up orphaned folder: {item}")
                        shutil.rmtree(item)
                        cleaned += 1
        if cleaned:
            print(f"🧹 Cleaned up {cleaned} orphaned folders")
        else:
            print("🧹 No orphaned folders found")
    except Exception as e:
        print(f"❌ Error in cleanup_orphaned_processed_folders: {str(e)}")


def start_periodic_cleanup(interval_minutes: int = 10, max_age_minutes: int = 10):
    """Background thread: fallback cleanup for abandoned jobs that were never downloaded."""
    def periodic_cleanup():
        while True:
            time.sleep(interval_minutes * 60)
            print(f"\n🕐 Running periodic cleanup at {datetime.now()}")
            cleanup_all_expired_jobs(max_age_minutes)
            cleanup_orphaned_processed_folders()

    t = threading.Thread(target=periodic_cleanup, daemon=True)
    t.start()
    print(f"🕐 Started periodic cleanup every {interval_minutes} min (max age: {max_age_minutes} min)")


@app.on_event("startup")
async def startup_event():
    cleanup_orphaned_processed_folders()
    start_periodic_cleanup(interval_minutes=10, max_age_minutes=10)


# ============================================================
# === Vocal Post-Processing ===
# ============================================================

def process_vocals(vocals_path: str, sr):
    y, sr = librosa.load(vocals_path, sr=sr, mono=False)

    if y.ndim == 1:
        y = np.vstack([y, y])

    processed_channels = []
    for ch in range(2):
        D = librosa.stft(y[ch])
        freqs = librosa.fft_frequencies(sr=sr)
        mask = freqs <= 10000
        D_filtered = D * mask[:, None]
        y_filtered = librosa.istft(D_filtered, length=y[ch].shape[0])
        processed_channels.append(y_filtered)

    y = np.vstack(processed_channels)
    min_len = min(y[0].shape[0], y[1].shape[0])
    y = y[:, :min_len]

    mid = (y[0] + y[1]) / 2
    side = (y[0] - y[1]) / 2

    y_processed = np.vstack([
        mid + side * 0.3,
        mid - side * 0.3
    ])

    sf.write(vocals_path, y_processed.T, sr)
    return y_processed, sr


# ============================================================
# === Core Spleeter Process ===
# ============================================================

def run_spleeter_sync(audio_path: str, output_dir: str, job_id: str, stem_mode: int):
    try:
        from spleeter.separator import Separator

        config = STEM_CONFIGS[stem_mode]
        separator = Separator(config["model"])
        separator.separate_to_file(
            audio_path,
            output_dir,
            filename_format='{instrument}.wav'
        )

        # --- 3-stem: merge bass + other into single "other" stem ---
        if stem_mode == 3:
            merge_sources = config.get("merge_into_other", [])
            merge_arrays = []
            sr_ref = None

            for stem_name in merge_sources:
                p = os.path.join(output_dir, f"{stem_name}.wav")
                if os.path.exists(p):
                    y_s, sr_s = librosa.load(p, sr=None, mono=False)
                    if y_s.ndim == 1:
                        y_s = np.vstack([y_s, y_s])
                    merge_arrays.append(y_s)
                    sr_ref = sr_s
                    os.remove(p)

            if merge_arrays and sr_ref:
                min_len = min(a.shape[1] for a in merge_arrays)
                merged = np.sum([a[:, :min_len] for a in merge_arrays], axis=0)
                merged /= (np.max(np.abs(merged)) + 1e-8)
                sf.write(os.path.join(output_dir, "other.wav"), merged.T, sr_ref)

        # --- Vocals post-processing ---
        vocals_path = os.path.join(output_dir, "vocals.wav")
        y_processed = None
        sr_vocals = None

        if os.path.exists(vocals_path):
            y_processed, sr_vocals = process_vocals(vocals_path, sr=None)

        # --- Drums + vocals final mix (5-stem and 3-stem only) ---
        if stem_mode in (3, 5) and y_processed is not None:
            drums_path = os.path.join(output_dir, "drums.wav")
            if os.path.exists(drums_path):
                drums, _ = librosa.load(drums_path, sr=sr_vocals, mono=False)
                if drums.ndim == 1:
                    drums = np.vstack([drums, drums])

                min_len = min(drums.shape[1], y_processed.shape[1])
                drums = drums[:, :min_len]
                y_v = y_processed[:, :min_len]

                final_mix = drums + y_v
                final_mix /= (np.max(np.abs(final_mix)) + 1e-8)
                sf.write(os.path.join(output_dir, "final_mix.wav"), final_mix.T, sr_vocals)

        # --- Collect stems ---
        display_names = config["display_names"]
        stems = []

        for f in sorted(os.listdir(output_dir)):
            if f.endswith(".wav"):
                stem_key = f.replace(".wav", "")
                display = display_names.get(stem_key, stem_key.capitalize())
                stems.append({
                    "name": display,
                    "filename": f,
                    "job_id": job_id
                })

        return {"success": True, "stems": stems}

    except Exception:
        print(traceback.format_exc())
        raise


# ============================================================
# === Routes ===
# ============================================================

@app.get("/")
async def root():
    return {"message": "VocaSplitter API is running!"}


@app.post("/api/separate", response_model=SeparateResponse)
async def separate_audio(
    file: UploadFile = File(...),
    stems: int = Query(default=5, description="Number of stems: 2, 3, or 5")
):
    if stems not in (2, 3, 5):
        raise HTTPException(400, "stems must be 2, 3, or 5")

    if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
        raise HTTPException(400, "Unsupported format")

    job_id = uuid.uuid4().hex
    temp_dir = f"temp_{job_id}"
    output_dir = f"processed_{job_id}"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    audio_path = os.path.join(temp_dir, file.filename)

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # created_at is set once here and NEVER overwritten
    job_store[job_id] = {
        "status": "processing",
        "output_dir": output_dir,
        "stem_mode": stems,
        "created_at": datetime.now(),
    }

    async def process():
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                run_spleeter_sync,
                audio_path,
                output_dir,
                job_id,
                stems,
            )

            # Only update status and stems — do NOT touch created_at
            job_store[job_id].update({
                "status": "completed",
                "stems": result["stems"],
            })

            # Fallback cleanup in case the user never downloads
            schedule_cleanup(job_id, delay_minutes=10)
            print(f"✅ Job {job_id} completed. Fallback cleanup in 10 minutes.")

        except Exception as e:
            print(f"❌ Job {job_id} failed: {str(e)}")
            job_store[job_id].update({
                "status": "failed",
                "error": str(e),
            })
            # Clean up failed jobs sooner
            schedule_cleanup(job_id, delay_minutes=5)

    asyncio.create_task(process())

    return {
        "success": True,
        "job_id": job_id,
        "stems": [],
        "stem_mode": stems,
    }


@app.get("/api/status/{job_id}", response_model=StatusResponse)
async def status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/download/{job_id}/{filename}")
async def download(job_id: str, filename: str, background: BackgroundTasks):
    """
    Download an individual stem file.
    Triggers cleanup 10 minutes after download to clean up processed wav files and temp files.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    path = os.path.join(job["output_dir"], filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")

    # Schedule cleanup after individual download as well
    background.add_task(cleanup_after_download, job_id, delay_seconds=600)  # 10 minutes

    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/api/download-zip/{job_id}")
async def download_zip(job_id: str, background: BackgroundTasks):
    """
    Download all stems as a zip file.
    Triggers cleanup 10 minutes after this call returns,
    giving FileResponse time to finish streaming.
    Also cancels the 10-minute fallback by cleaning up sooner.
    """
    job = job_store.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Job not found or not completed")

    output_dir = job.get("output_dir")

    # Zip lives in ZIP_DIR (separate from output_dir) so output_dir
    # can be safely deleted without corrupting an in-progress zip write.
    zip_path = os.path.join(ZIP_DIR, f"{job_id}_stems.zip")

    if not os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            for f in os.listdir(output_dir):
                if f.endswith(".wav"):
                    zipf.write(os.path.join(output_dir, f), arcname=f)

    # Schedule cleanup AFTER zip is fully written and response has streamed
    background.add_task(cleanup_after_download, job_id, delay_seconds=600)

    return FileResponse(
        path=zip_path,
        filename="stems.zip",
        media_type="application/zip"
    )