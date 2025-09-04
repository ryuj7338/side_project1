from pathlib import Path
import os
import uuid
import subprocess
import json
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

# 너의 기존 분석 함수
from analysis import analyze_voice


# ---------------------------
# 기본 설정
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULT_DIR = STATIC_DIR / "results"
UPLOAD_DIR = BASE_DIR / "uploads"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Voice → (KO)Explanation & (EN)Prompt → Image")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# 정적/템플릿(네 프로젝트에 이미 있는 base.html/index.html/result.html)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------
# 유틸: FFmpeg 변환 / OpenAI Whisper STT
# ---------------------------
def _ffmpeg_convert_to_wav(src_path: Path, dst_path: Path):
    # webm, mp3, m4a 등 → 16kHz mono wav
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst_path)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def transcribe_wav_with_openai(wav_path: Path) -> str:
    if not openai_client:
        return ""
    with open(wav_path, "rb") as f:
        try:
            resp = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json",
                temperature=0,
            )
            return getattr(resp, "text", "") or ""
        except Exception:
            return ""


# ---------------------------
# KO 설명 → EN 프롬프트 (역할 지정 포함)
# ---------------------------
NEGATIVE_DEFAULT = (
    "low quality, blurry, extra limbs, distorted anatomy, deformed, text, "
    "watermark, logo, frame, oversaturated, underexposed, jpeg artifacts"
)

def _system_prompt_for_bilingual():
    # ✅ 역할 지정: 음성 분석 해석 전문가
    return (
        "You are an expert in interpreting voice analysis results.\n\n"
        "TASK:\n"
        "1) From the given Korean description and tags, write a concise Korean explanation "
        "about the impression/personality this voice might give. 2–3 sentences.\n"
        "2) Then produce a vivid English scene description suitable for image generation. "
        "Describe visuals (character aura/personality, mood, lighting, style) and DO NOT "
        "mention 'voice' or 'audio'.\n\n"
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
    )

def _user_prompt_from_ko(ko_description: str, tags: List[str]):
    return (
        "KOREAN DESCRIPTION:\n" + (ko_description or "") + "\n\n"
        "TAGS (mood/style cues):\n" + ", ".join(tags or []) + "\n\n"
        "Return STRICT JSON only."
    )

def _fallback_from_ko(ko_description: str, tags: List[str]) -> Dict:
    # 🔒 OpenAI 키가 없거나 실패할 때 안전 폴백
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
    }

def _safe_json_parse(txt: str) -> Dict:
    try:
        return json.loads(txt)
    except Exception:
        return {}

async def ko_to_bilingual_prompt(ko_description: str, tags: List[str]) -> Dict:
    """한국어 설명+태그 → (ko_explanation, en_prompt, …) JSON 생성.
       OpenAI 있으면 LLM 사용, 실패/없으면 폴백."""
    if not openai_client:
        return _fallback_from_ko(ko_description, tags)
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
        # 최소 보강
        j["ko_explanation"] = j.get("ko_explanation", ko_description or "")
        j["en_prompt"] = j.get("en_prompt", _fallback_from_ko(ko_description, tags)["en_prompt"])
        j["style_tags"] = j.get("style_tags", []) or []
        j["palette"] = j.get("palette", []) or []
        j["seed_idea"] = j.get("seed_idea", "") or ""
        return j
    except Exception:
        return _fallback_from_ko(ko_description, tags)


# ---------------------------
# 라우트
# ---------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze_api")
async def analyze_api(file: UploadFile = File(...)):
    # 1) 파일 검사
    raw = await file.read()
    if not raw or len(raw) < 3000:
        return {"ok": False, "error": "파일이 비어 있거나 너무 짧습니다."}

    # 2) 임시 저장 및 변환(webm/mp3 등 → wav)
    ext = Path(file.filename or "audio.webm").suffix.lower() or ".webm"
    uid = uuid.uuid4().hex
    src_path = UPLOAD_DIR / f"{uid}{ext}"
    with open(src_path, "wb") as f:
        f.write(raw)
    wav_path = UPLOAD_DIR / f"{uid}.wav"
    try:
        _ffmpeg_convert_to_wav(src_path, wav_path)
    except Exception:
        return {"ok": False, "error": "오디오 변환(FFmpeg)에 실패했습니다."}

    # 3) 음성 분석 (네 기존 로직)
    result = analyze_voice(str(wav_path))

    # 4) 전사(선택)
    transcript = ""
    try:
        transcript = transcribe_wav_with_openai(wav_path)
    except Exception:
        transcript = ""

    # 5) 기본 검증 (무음/짧음 등)
    if (result.get("energy_mean", 0.0) < 0.01) or not result.get("description"):
        return {"ok": False, "error": "무음 또는 유효하지 않은 녹음"}

    # 6) 한국어 설명 + 영어 프롬프트 동시 생성
    bilingual = await ko_to_bilingual_prompt(result.get("description",""), result.get("tags", []))
    result.update(bilingual)  # description/tags 유지 + ko_explanation/en_prompt 등 추가

    # 7) 응답(JSON)
    result["ok"] = True
    result["transcript"] = transcript
    return JSONResponse(result)


# ---------------------------
# (선택) 이미지 생성 엔드포인트
# - OPENAI 키 없으면 placeholder PNG 저장
# ---------------------------
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

    # 1) OpenAI Images (가능하면)
    if openai_client and prompt:
        try:
            img = openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt + (("" if not negative else f"\nNegative: {negative}")),
                size=f"{width}x{height}",
            )
            b64 = img.data[0].b64_json
            with open(out_path, "wb") as f:
                f.write(base64.b64decode(b64))
            return {"imageUrl": f"/static/results/{out_fn}", "engine": "openai"}
        except Exception:
            pass

    # 2) 폴백: placeholder PNG 생성
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
        # 최후: 1x1 PNG
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8Xw8AAtMB8g9m9O8AAAAASUVORK5CYII="
            ))
        return {"imageUrl": f"/static/results/{out_fn}", "engine": "fallback"}
