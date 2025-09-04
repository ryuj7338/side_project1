from __future__ import annotations

from pathlib import Path
import os
import uuid
import json
import subprocess
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

# 기존 분석 함수 (두 스키마 모두 호환)
from analysis import analyze_voice  # flat or {"features":{...}} 모두 처리

# ============================
# 디렉터리
# ============================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULT_DIR = STATIC_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# 환경변수 로드(.env 우선, 없으면 sample도)
# ============================
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.sample")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ============================
# FastAPI + CORS
# ============================
app = FastAPI(title="Voice → (KO)Explanation & (EN)Prompt → Image")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 정적/템플릿
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

# ============================
# 유틸: FFmpeg 변환 & Whisper 전사
# ============================
def _ffmpeg_convert_to_wav(src_path: Path, dst_path: Path):
    """webm/mp3/m4a 등 → 16kHz mono WAV"""
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)  # :contentReference[oaicite:4]{index=4}

def transcribe_wav_with_openai(wav_path: Path) -> str:
    """OpenAI Whisper 전사 (키 없으면 빈 문자열)"""
    if not openai_client:
        return ""
    try:
        with open(wav_path, "rb") as f:
            resp = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json",
                temperature=0,
            )
        return getattr(resp, "text", "") or ""
    except Exception:
        return ""  # :contentReference[oaicite:5]{index=5}

# ============================
# KO→EN 프롬프트 생성(역할 지정 + 저작권 안전 스타일 힌트)
# ============================
NEGATIVE_DEFAULT = (
    "low quality, blurry, extra limbs, distorted anatomy, deformed, text, "
    "watermark, logo, frame, oversaturated, underexposed, jpeg artifacts"
)

# 특정 IP를 직접 언급하지 않는 “느낌만” 힌트
SHINCHAN_STYLE_HINT = (
    "chibi, two-head-tall (super-deformed), bold thick outlines, "
    "simple geometric shapes, playful gag-anime vibe, minimal facial features, "
    "original character only; do not reference or imitate any specific copyrighted IP."
)

def _system_prompt_for_bilingual():
    return (
        "You are an expert in interpreting voice analysis results.\n\n"
        "TASK:\n"
        "1) From the given Korean description and tags, write a concise Korean explanation "
        "about the impression/personality this voice might give. 2–3 sentences.\n"
        "2) Then produce a vivid English scene description suitable for image generation. "
        "Describe visuals (character aura/personality, mood, lighting, style) and DO NOT "
        "mention 'voice' or 'audio'. Do NOT mention any brand names, show titles, or copyrighted characters.\n\n"
        "OUTPUT (STRICT JSON only):\n"
        "{\n"
        '  "ko_explanation": string,\n'
        '  "en_prompt": string,\n'
        '  "negative_prompt": string,\n'
        '  "style_tags": string[],\n'
        '  "palette": string[],\n'
        '  "seed_idea": string\n'
        "}\n"
        '- Use "low quality, blurry, extra limbs, distorted anatomy, deformed, text, watermark, logo, frame, oversaturated, underexposed, jpeg artifacts" for negative_prompt.\n'
        "- 3–6 HEX colors for palette; seed_idea is 6–12 words.\n"
    )  # :contentReference[oaicite:6]{index=6}

def _user_prompt_from_ko(ko_description: str, tags: List[str]):
    return (
        "KOREAN DESCRIPTION:\n" + (ko_description or "") + "\n\n"
        "TAGS (mood/style cues):\n" + ", ".join(tags or []) + "\n\n"
        "Return STRICT JSON only."
    )  # :contentReference[oaicite:7]{index=7}

def _fallback_from_ko(ko_description: str, tags: List[str]) -> Dict:
    mood = "calm" if ("차분" in (ko_description or "")) else "energetic"
    style = "semi-realistic"
    subject = "character portrait"
    scene = "soft ambient environment"
    en_prompt = (
        f"A {subject} in a {scene}, conveying an {mood} atmosphere; "
        f"cinematic composition, soft rim light, subtle depth of field; "
        f"clean lines, coherent anatomy; camera: medium shot; style: {style}."
    )
    return {
        "ko_explanation": ko_description or "차분하고 안정적인 인상을 주는 목소리처럼 느껴집니다.",
        "en_prompt": en_prompt,
        "negative_prompt": NEGATIVE_DEFAULT,
        "style_tags": [style, "cinematic", "clean-lines"],
        "palette": ["#A0C4FF", "#BDB2FF", "#FFC6FF", "#FFADAD"],
        "seed_idea": f"{mood} {subject} in {scene}"
    }  # :contentReference[oaicite:8]{index=8}

def _safe_json_parse(txt: str) -> Dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

async def ko_to_bilingual_prompt(ko_description: str, tags: List[str]) -> Dict:
    """한국어 설명+태그 → ko_explanation/en_prompt/… 생성 (LLM 있으면 사용, 없으면 폴백)"""
    if not openai_client:
        j = _fallback_from_ko(ko_description, tags)
        j["en_prompt"] = (j["en_prompt"] + " " + SHINCHAN_STYLE_HINT).strip()
        j.setdefault("style_tags", []).append("shinchan-like")
        return j
    try:
        chat = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _system_prompt_for_bilingual()},
                {"role": "user", "content": _user_prompt_from_ko(ko_description, tags)}
            ],
            temperature=0.7,
        )
        content = chat.choices[0].message.content
        j = _safe_json_parse(content) or {}
        if "negative_prompt" not in j:
            j["negative_prompt"] = NEGATIVE_DEFAULT
        j["ko_explanation"] = j.get("ko_explanation", ko_description or "")
        j["en_prompt"] = j.get("en_prompt", _fallback_from_ko(ko_description, tags)["en_prompt"])
        j["en_prompt"] = (j["en_prompt"] + " " + SHINCHAN_STYLE_HINT).strip()
        j["style_tags"] = (j.get("style_tags") or []) + ["shinchan-like"]
        j["palette"] = j.get("palette", []) or []
        j["seed_idea"] = j.get("seed_idea", "") or ""
        return j
    except Exception:
        j = _fallback_from_ko(ko_description, tags)
        j["en_prompt"] = (j["en_prompt"] + " " + SHINCHAN_STYLE_HINT).strip()
        j.setdefault("style_tags", []).append("shinchan-like")
        return j  # :contentReference[oaicite:9]{index=9}

# ============================
# 라우트
# ============================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not templates:
        return HTMLResponse("<h3>Server is running.</h3>", status_code=200)
    return templates.TemplateResponse("index.html", {"request": request})  # :contentReference[oaicite:10]{index=10}

@app.post("/analyze_api")
async def analyze_api(file: UploadFile = File(...)):
    """
    업로드 → (필요 시) 변환 → 분석 → 미리듣기/프롬프트/전사 반환
    - 분석 결과가 flat 또는 {"features":{...}} 어느 쪽이든 호환
    """
    raw = await file.read()
    if not raw or len(raw) < 3000:
        return {"ok": False, "error": "파일이 비어 있거나 너무 짧습니다."}

    # 원본 저장
    ext = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
    uid = uuid.uuid4().hex
    src_path = UPLOAD_DIR / f"{uid}{ext}"
    with open(src_path, "wb") as f:
        f.write(raw)

    # 변환 → wav
    wav_path = UPLOAD_DIR / f"{uid}.wav"
    try:
        _ffmpeg_convert_to_wav(src_path, wav_path)
    except Exception:
        return {"ok": False, "error": "오디오 변환(FFmpeg)에 실패했습니다."}

    # 분석 (두 타입 모두 지원)
    analysis = analyze_voice(str(wav_path))  # :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12} :contentReference[oaicite:13]{index=13}

    # 전사(선택)
    transcript = ""
    try:
        transcript = transcribe_wav_with_openai(wav_path)
    except Exception:
        transcript = ""

    # 공통 필드 추출(호환 레이어)
    features = analysis.get("features", analysis) or {}
    energy_mean = features.get("energy_mean", analysis.get("energy_mean", 0.0))
    desc_text = (
        analysis.get("description_ko")
        or analysis.get("description")
        or ""
    )

    # 기본 검증
    if (energy_mean or 0.0) < 1e-3 or not desc_text.strip():
        return {"ok": False, "error": "무음 또는 유효하지 않은 녹음"}

    # 한국어 설명 + 영어 프롬프트 동시 생성
    tags = analysis.get("tags", [])
    bilingual = await ko_to_bilingual_prompt(desc_text, tags)

    # 응답
    result = {
        "ok": True,
        "file": {"name": file.filename, "saved_as": src_path.name, "path": str(src_path)},
        "preview_url": f"/uploads/{uid}{ext}",  # 미리듣기 URL
        "transcript": transcript,

        # 분석 결과
        "features": features,
        "tags": tags,

        # 설명/프롬프트(우선권=LLM 결과)
        "description_ko": bilingual.get("ko_explanation", desc_text),
        "en_prompt": bilingual.get("en_prompt", analysis.get("en_prompt", "")),
        "negative": bilingual.get("negative_prompt", analysis.get("negative", NEGATIVE_DEFAULT)),
        "style_tags": bilingual.get("style_tags", analysis.get("style_tags", [])),
        "palette": bilingual.get("palette", analysis.get("palette", [])),
        "seed": bilingual.get("seed_idea", analysis.get("seed", "")),
    }
    return JSONResponse(result)  # :contentReference[oaicite:14]{index=14}

# 업로드 별칭 (과거 프론트 호환)
@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    return await analyze_api(file)  # :contentReference[oaicite:15]{index=15}

# ============================
# 이미지 생성 (키 없으면 placeholder)
# ============================
import base64
from PIL import Image, ImageDraw, ImageFont

@app.post("/image/render")
async def image_render(payload: Dict):
    """
    입력: {"prompt": "...", "negativePrompt": "...", "width": 768, "height": 768}
    출력: {"imageUrl": "/static/results/xxxx.png", "engine": "openai|placeholder"}
    """
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    width = int(payload.get("width", 768))
    height = int(payload.get("height", 768))
    prompt = str(payload.get("prompt", "")).strip()
    negative = str(payload.get("negativePrompt", "")).strip()
    out_fn = f"img_{uuid.uuid4().hex}.png"
    out_path = RESULT_DIR / out_fn

    # 1) OpenAI 이미지 (가능 시)
    if openai_client and prompt:
        try:
            img = openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt + ("" if not negative else f"\nNegative: {negative}"),
                size=f"{width}x{height}",
            )
            b64 = img.data[0].b64_json
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
            return {"imageUrl": f"/static/results/{out_fn}", "engine": "openai"}
        except Exception:
            pass

    # 2) 폴백: placeholder
    try:
        im = Image.new("RGB", (width, height), (242, 244, 248))
        dr = ImageDraw.Draw(im)
        text = "Placeholder Image\n\nConfigure OPENAI_API_KEY\nfor real generation"
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = None
        dr.multiline_text((40, 40), text, fill=(52, 58, 64), font=font, spacing=6)
        im.save(out_path)
        return {"imageUrl": f"/static/results/{out_fn}", "engine": "placeholder"}
    except Exception:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAtMB8g9m9O8AAAAASUVORK5CYII="
            ))
        return {"imageUrl": f"/static/results/{out_fn}", "engine": "fallback"}  # :contentReference[oaicite:16]{index=16}

# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}  # :contentReference[oaicite:17]{index=17}
