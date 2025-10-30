import os
from pathlib import Path
from typing import Dict, Any, Optional, List

# Ensure PyTorch-only for transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
"""
Important: keep the Hugging Face cache outside the project directory to
avoid uvicorn StatReload watching large model download updates and
restarting the server mid-run.
"""
_hf_home = os.environ.get("HF_HOME")
if not _hf_home:
    import pathlib as _p
    os.environ["HF_HOME"] = str(_p.Path.home() / ".cache" / "huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"]) 

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import tempfile

from PIL import Image
import torch
from ultralytics import YOLO
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # optional

# Directories
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / ".cache"  # app-local misc cache (not HF cache)
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model config (SLM-first)
# Allow overriding via environment variables for accuracy/speed tradeoffs
YOLO_VARIANT = os.environ.get("YOLO_VARIANT", "yolov8n.pt")
CAPTION_CKPT = os.environ.get("CAPTION_CKPT", "Salesforce/blip-image-captioning-base")
VQA_CKPT = os.environ.get("VQA_CKPT", "Salesforce/blip-vqa-base")

# Small language model for scene reasoning (generates richer narrative)
REASONER_CKPT = os.environ.get("REASONER_CKPT", "google/flan-t5-small")

# Gemini config (optional)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY_ALT = os.environ.get("GEMINI_API_KEY_ALT", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Device selection
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Load models
_yolo: Optional[YOLO] = None
_caption_processor: Optional[BlipProcessor] = None
_caption_model: Optional[BlipForConditionalGeneration] = None
_vqa_processor: Optional[BlipProcessor] = None
_vqa_model: Optional[BlipForQuestionAnswering] = None
_reasoner_tokenizer: Optional[AutoTokenizer] = None
_reasoner_model: Optional[AutoModelForSeq2SeqLM] = None
_gemini_model = None
_gemini_model_alt = None

# Simple in-memory vitals store
LAST_VITALS: Dict[str, Any] = {}


def _lazy_load():
    global _yolo, _caption_processor, _caption_model, _vqa_processor, _vqa_model, _reasoner_tokenizer, _reasoner_model
    if _yolo is None:
        _yolo = YOLO(YOLO_VARIANT)
    if _caption_processor is None:
        # Use HF_HOME/TRANSFORMERS_CACHE; avoid writing to project .cache to prevent reload loops
        _caption_processor = BlipProcessor.from_pretrained(CAPTION_CKPT)
    if _caption_model is None:
        _caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_CKPT).to(DEVICE)
    if _vqa_processor is None:
        _vqa_processor = BlipProcessor.from_pretrained(VQA_CKPT)
    if _vqa_model is None:
        _vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_CKPT).to(DEVICE)
    if _reasoner_tokenizer is None or _reasoner_model is None:
        try:
            _reasoner_tokenizer = AutoTokenizer.from_pretrained(REASONER_CKPT)
            _reasoner_model = AutoModelForSeq2SeqLM.from_pretrained(REASONER_CKPT).to(DEVICE)
        except Exception:
            # Reasoner is optional; continue without it
            _reasoner_tokenizer = None
            _reasoner_model = None
    # Configure Gemini once if key is present
    global _gemini_model, _gemini_model_alt
    if _gemini_model is None and GEMINI_API_KEY and genai is not None:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        except Exception:
            _gemini_model = None
    if _gemini_model_alt is None and GEMINI_API_KEY_ALT and genai is not None:
        try:
            # Configure with alternate key just-in-time during call; keep handle
            _gemini_model_alt = (GEMINI_MODEL, GEMINI_API_KEY_ALT)
        except Exception:
            _gemini_model_alt = None


def detect(image_path: Path, conf: float = 0.15):
    _lazy_load()
    results = _yolo.predict(source=str(image_path), conf=conf, verbose=False)
    return results[0]


def caption(image_path: Path, max_new_tokens: int = 30) -> str:
    _lazy_load()
    image = Image.open(image_path).convert("RGB")
    inputs = _caption_processor(images=image, return_tensors="pt").to(DEVICE)
    out = _caption_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = _caption_processor.batch_decode(out, skip_special_tokens=True)[0]
    return text


def vqa(image_path: Path, question: str, max_new_tokens: int = 20) -> str:
    _lazy_load()
    image = Image.open(image_path).convert("RGB")
    inputs = _vqa_processor(images=image, text=question, return_tensors="pt").to(DEVICE)
    out = _vqa_model.generate(**inputs, max_new_tokens=max_new_tokens)
    ans = _vqa_processor.batch_decode(out, skip_special_tokens=True)[0]
    return ans


def _format_detection_summary(detections: List[Dict[str, Any]]) -> str:
    return ", ".join(
        f"{d.get('class_name', d.get('class_id'))} {(d.get('confidence', 0.0)*100):.0f}%"
        for d in detections[:6]
    ) or "none"


def generate_narrative(caption_text: str, detections: List[Dict[str, Any]], max_new_tokens: int = 80, question: Optional[str] = None) -> Optional[str]:
    """Use a small instruction-tuned model to synthesize a concise scene description.

    The prompt fuses the raw caption and the top detections into a short, natural sentence
    optimized for text-to-speech.
    """
    _lazy_load()

    # Preferred: Gemini if configured
    det_summ = _format_detection_summary(detections)
    if _gemini_model is not None or _gemini_model_alt is not None:
        try:
            q = (question or "").strip() or "Explain the surroundings succinctly."
            system = (
                "You assist blind users. Provide ONE natural sentence that describes the scene. "
                "Do not list object names; avoid enumeration. Avoid speculation. Max 25 words."
            )
            prompt = (
                f"Instruction: {system}\n"
                f"User request: {q}\n"
                f"Caption: {caption_text or 'n/a'}\n"
                f"Detections (for your context, do not enumerate them): {det_summ}\n"
                f"Response:"
            )
            model_to_use = _gemini_model
            try:
                if model_to_use is None and _gemini_model_alt is not None and genai is not None:
                    # configure alt key on the fly
                    genai.configure(api_key=_gemini_model_alt[1])
                    model_to_use = genai.GenerativeModel(_gemini_model_alt[0])
                resp = model_to_use.generate_content(prompt) if model_to_use else None
            except Exception:
                # try alt if primary failed
                if _gemini_model_alt is not None and genai is not None:
                    try:
                        genai.configure(api_key=_gemini_model_alt[1])
                        alt_model = genai.GenerativeModel(_gemini_model_alt[0])
                        resp = alt_model.generate_content(prompt)
                    except Exception:
                        resp = None
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass

    # Fallback: FLAN-T5 small
    if _reasoner_tokenizer is None or _reasoner_model is None:
        return None
    instruction = (
        "You assist blind users. Provide ONE natural sentence that describes the scene. "
        "Do not list object names; avoid enumeration. Avoid speculation. Max 25 words."
    )
    prompt = (
        f"Instruction: {instruction}\n"
        f"Caption: {caption_text or 'n/a'}\n"
        f"Detections (for your context, do not enumerate them): {det_summ}\n"
        f"Response:"
    )
    inputs = _reasoner_tokenizer([prompt], return_tensors="pt").to(DEVICE)
    output_ids = _reasoner_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=2,
        early_stopping=True,
    )
    text = _reasoner_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "Response:" in text:
        text = text.split("Response:", 1)[-1].strip()
    return text.strip() or None


def analyze_image(image_path: Path, question: Optional[str] = None, conf: Optional[float] = None) -> Dict[str, Any]:
    det = detect(image_path, conf=conf if isinstance(conf, (float, int)) else 0.15)
    cap = caption(image_path)
    
    _q = (question or "").strip()
    if not _q:
        _q = "What is in front of me?"
    # VQA: prefer Gemini if configured; fall back to BLIP
    qa = None
    det_summ = _format_detection_summary([])
    try:
        det_summ = _format_detection_summary([])
    except Exception:
        pass
    if (_gemini_model is not None or _gemini_model_alt is not None) and _q:
        # Build intent-aware prompt
        intent_tips = (
            "You are assisting a blind user. Answer clearly and briefly (max 40 words). "
            "If asked to list objects, enumerate concise names from detections/caption. "
            "If asked to describe surroundings, provide a concise sentence. "
            "If asked about danger ahead, infer from large/close objects (vehicles, obstacles) and respond with a brief safety cue."
        )
        det_s = _format_detection_summary([])
        try:
            det_s = _format_detection_summary(det.boxes and [
                {
                    "class_id": int(b.cls[0].item()),
                    "class_name": getattr(getattr(det, "names", {}), "get", lambda *_: None)(int(b.cls[0].item())) if isinstance(getattr(det, "names", {}), dict) else str(int(b.cls[0].item())),
                    "confidence": float(b.conf[0].item()),
                    "xyxy": [float(v) for v in b.xyxy[0].tolist()],
                } for b in det.boxes
            ] or [])
        except Exception:
            pass
        gem_prompt = (
            f"Instruction: {intent_tips}\n"
            f"Question: {_q}\n"
            f"Caption: {cap or 'n/a'}\n"
            f"Detections: {det_s}\n"
            f"Answer:"
        )
        try:
            model_to_use = _gemini_model
            if model_to_use is None and _gemini_model_alt is not None and genai is not None:
                genai.configure(api_key=_gemini_model_alt[1])
                model_to_use = genai.GenerativeModel(_gemini_model_alt[0])
            resp = model_to_use.generate_content(gem_prompt) if model_to_use else None
            qa = (getattr(resp, "text", None) or "").strip() or None
        except Exception:
            qa = None
    if not qa:
        qa = vqa(image_path, _q)

    names = getattr(det, "names", None) or {}
    det_boxes: List[Dict[str, Any]] = []
    try:
        for b in det.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = [float(v) for v in b.xyxy[0].tolist()]
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            det_boxes.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "xyxy": xyxy,
            })
    except Exception:
        pass

    # Use Gemini/Reasoner for a concise non-enumerated narrative. If user asked a question, pass it.
    narrative = generate_narrative(cap, det_boxes, question=question)

    return {
        "device": DEVICE,
        "caption": cap,
        "vqa_answer": qa,
        "detections": det_boxes,
        "narrative": narrative,
        "reasoner_enabled": _reasoner_model is not None,
        "narrative_source": ("reasoner" if narrative else "caption"),
    }


app = FastAPI(title="VISOR Backend (SLM)")
_READY = False

# Dev CORS (open)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend if present
if FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="ui")


@app.on_event("startup")
def _warmup_startup():
    global _READY  # CRITICAL FIX: Declare global to modify module-level variable
    try:
        _lazy_load()
        # Create a proper dummy image: 224x224 RGB (BLIP's expected size)
        from PIL import Image as _I
        import io as _io
        buf = _io.BytesIO()
        _I.new("RGB", (224, 224), (128, 128, 128)).save(buf, format="JPEG")
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(buf.getvalue())
            tmp_path = Path(tmp.name)
        try:
            _ = analyze_image(tmp_path, question="What is in front of me?", conf=0.15)
        finally:
            try: 
                tmp_path.unlink(missing_ok=True)
            except Exception: 
                pass
        _READY = True
        print(f"✓ Models loaded successfully on {DEVICE}")
    except Exception as e:
        _READY = False
        print(f"✗ Warmup failed: {e}")


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    question: Optional[str] = Form(default=None),
    conf: Optional[float] = Form(default=None),
):
    if not _READY:
        return JSONResponse(content={"ready": False, "message": "Model warming up"}, status_code=503)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        result = analyze_image(tmp_path, question=question, conf=conf)
        try:
            print(f"/analyze -> det={len(result.get('detections', []) )} caption_len={len(result.get('caption',''))} vqa_len={len(result.get('vqa_answer',''))}")
        except Exception:
            pass
        result["ready"] = True
        return JSONResponse(content=result)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# Health check
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "ready": _READY}


# ------------------- Vitals API -------------------
def summarize_vitals_short(vitals: Dict[str, Any]) -> Optional[str]:
    """Produce a very short suggestion for a blind user based on vitals.
    Must mention heart rate if available; max ~20 words.
    """
    _lazy_load()
    if not vitals:
        return None
    hr = vitals.get("heart_rate")
    context = f"HR={hr} bpm" if hr is not None else ""
    sys = vitals.get("systolic")
    dia = vitals.get("diastolic")
    spo2 = vitals.get("spo2")
    if sys and dia:
        context += f", BP={sys}/{dia}"
    if spo2:
        context += f", SpO2={spo2}%"

    prompt = (
        "You assist a blind user. Give one short, actionable suggestion based on current vitals. "
        "Mention heart rate. Avoid medical claims. Max 20 words.\n"
        f"Vitals: {context or 'n/a'}\n"
        "Suggestion:"
    )

    # Prefer Gemini
    if _gemini_model is not None or _gemini_model_alt is not None:
        try:
            model_to_use = _gemini_model
            if model_to_use is None and _gemini_model_alt is not None and genai is not None:
                genai.configure(api_key=_gemini_model_alt[1])
                model_to_use = genai.GenerativeModel(_gemini_model_alt[0])
            resp = model_to_use.generate_content(prompt) if model_to_use else None
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass

    # Fallback: FLAN
    if _reasoner_tokenizer is None or _reasoner_model is None:
        return None
    inputs = _reasoner_tokenizer([prompt], return_tensors="pt").to(DEVICE)
    output_ids = _reasoner_model.generate(**inputs, max_new_tokens=40, do_sample=False, num_beams=2, early_stopping=True)
    text = _reasoner_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    if "Suggestion:" in text:
        text = text.split("Suggestion:", 1)[-1].strip()
    return text.strip() or None


@app.post("/vitals")
async def post_vitals(payload: Dict[str, Any]):
    """Receive vitals from a companion app or bridge.
    Example JSON: {"heart_rate":72, "spo2":98, "systolic":120, "diastolic":80, "steps":1234, "ts": 1699999999}
    """
    global LAST_VITALS
    try:
        LAST_VITALS = dict(payload or {})
        return {"ok": True}
    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=400)


@app.get("/vitals/latest")
def get_vitals_latest():
    return {"vitals": LAST_VITALS}


@app.get("/vitals/summary")
def get_vitals_summary():
    sugg = summarize_vitals_short(LAST_VITALS)
    return {"vitals": LAST_VITALS, "suggestion": sugg}