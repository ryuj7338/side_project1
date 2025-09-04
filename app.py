# app.py
from pathlib import Path
import os
import uuid
import subprocess
from typing import Dict

from fastapi import FastAPI, Request, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

from analysis import analyze_voice

# ============================
# 기본 경로/디렉터리 설정
# ============================
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
RESULT_DIR = STATIC_DIR / "results"   # (필요 시) 결과 산출물 저장
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================
# FastAPI 앱
# ============================
app = FastAPI(title="Voice → Description → (Prompt)")

# CORS(필요 시 도메인 제한)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 배포시 필요한 도메인만 허용 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적/템플릿 (프론트 사용 시)
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None

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

    # 2) 분석
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

# --- 추가: /analyze_api 경로도 /analyze_upload와 동일하게 동작하도록 별칭 라우트 ---
@app.post("/analyze_api")
async def analyze_api(file: UploadFile = File(...)):
    return await analyze_upload(file)
# --- 추가 끝 ---

# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}
