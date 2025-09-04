# app.py
from pathlib import Path
import os
import uuid
import base64
from typing import Dict, Optional

from fastapi import FastAPI, Request, UploadFile, File, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

from analysis import analyze_voice

# ============================
# 기본 경로/디렉터리 설정
# ============================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULT_DIR = STATIC_DIR / "results"   # 이미지 등 산출물 저장
UPLOAD_DIR = BASE_DIR / "uploads"     # 업로드/임시 오디오 저장

if RESULT_DIR.exists() and not RESULT_DIR.is_dir():
    os.rename(RESULT_DIR, str(RESULT_DIR) + ".bak")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

if not UPLOAD_DIR.exists():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# 환경 설정
# ============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # (선택) 사용 중이면 쓰고, 아니면 무시됨

# OpenAI 클라이언트
client = OpenAI()  # OPENAI_API_KEY 필요

# ============================
# FastAPI 앱
# ============================
app = FastAPI(title="Voice → Description → (Prompt)")

# CORS(필요 시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포 시 특정 도메인으로 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적/템플릿
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

# ============================
# 유틸
# ============================
def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def save_b64_png(b64: str) -> str:
    """b64 PNG를 파일로 저장하고 /static 경로 URL 반환"""
    fname = f"{uuid.uuid4().hex}.png"
    out_path = RESULT_DIR / fname
    out_path.write_bytes(base64.b64decode(b64))
    return f"/static/results/{fname}"

# ============================
# 스키마
# ============================
class ImageRequest(BaseModel):
    prompt: str
    negativePrompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024

# ============================
# 라우트
# ============================
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if not templates:
        return HTMLResponse("<h3>Server is running.</h3>", status_code=200)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    """
    프론트에서 업로드한 오디오를 저장 → 분석 → 결과 JSON 반환
    """
    # 1) 업로드 저장
    ext = Path(file.filename).suffix.lower() or ".wav"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

    with save_path.open("wb") as f:
        f.write(await file.read())

    # 2) 분석 (사용자 정의 함수: analysis.py의 analyze_voice)
    analysis = analyze_voice(str(save_path))

    # 3) 응답(JSON)
    payload: Dict = {
        "ok": True,
        "file": {
            "name": file.filename,
            "saved_as": str(save_path.name),
            "path": str(save_path),
        },
        "features": analysis.get("features", {}),
        "description": analysis.get("description_ko", ""),  # 프론트 바인딩 필드
        "en_prompt": analysis.get("en_prompt", ""),
        "negative": analysis.get("negative", ""),
        "style_tags": analysis.get("style_tags", ""),
        "palette": analysis.get("palette", []),
        "seed": analysis.get("seed", ""),
    }
    return JSONResponse(payload)

# /analyze_api 별칭 (프론트 기존 코드 호환)
@app.post("/analyze_api")
async def analyze_api(file: UploadFile = File(...)):
    return await analyze_upload(file)

def _pick_openai_size(w: int, h: int) -> str:
    """
    gpt-image-1 이 허용하는 사이즈로 변환:
    - 정사각형 → 1024x1024
    - 세로가 더 길면 → 1024x1536
    - 가로가 더 길면 → 1536x1024
    """
    if w <= 0 or h <= 0:
        return "1024x1024"
    if w == h:
        return "1024x1024"
    return "1024x1536" if h > w else "1536x1024"

@app.post("/image/render")
async def image_render(req: ImageRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    # 기존 _clamp 제거/유지 상관없음. 허용 사이즈로 변환만 확실히 수행
    size_str = _pick_openai_size(req.width, req.height)

    neg = req.negative_prompt or req.negativePrompt
    final_prompt = req.prompt if not neg else f"{req.prompt}\nNegative: {neg}"

    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=final_prompt,
            size=size_str,     # ← 여기!
            n=1,
            quality="high",
        )
        b64 = resp.data[0].b64_json
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OpenAI image generation failed: {e}")

    url = save_b64_png(b64)
    return {"imageUrl": url}
# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}

# ============================
# 로컬 실행
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)