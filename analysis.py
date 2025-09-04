# analysis.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import librosa

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# 환경 변수 로드 & OpenAI 준비 (경로 고정)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
# 프로젝트 루트(= app.py/analysis.py가 있는 폴더)의 .env를 확실하게 읽도록 지정
load_dotenv(BASE_DIR / ".env.sample")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# api_key를 명시적으로 주입 (이 부분이 없어서 키 미인식 오류가 났습니다)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # 키가 전혀 안 잡힐 때도 서비스가 죽지 않도록, 아래 LLM 호출부에서 폴백이 작동합니다.
    # 그래도 콘솔에서 바로 알아볼 수 있게 안내 메시지 남깁니다.
    print("[analysis.py] WARNING: OPENAI_API_KEY is not set. Falling back to default description on LLM failure.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# 유틸
# ---------------------------
def _safe_float(x, ndigits: int = 5) -> Optional[float]:
    try:
        return round(float(x), ndigits)
    except Exception:
        return None

def _nan_robust(values: np.ndarray, fn, default=None):
    try:
        v = fn(values[~np.isnan(values)])
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

@dataclass
class VoiceFeatures:
    duration_sec: Optional[float]
    f0_med: Optional[float]
    f0_range: Optional[float]
    energy_mean: Optional[float]
    zcr_mean: Optional[float]
    sc_mean: Optional[float]
    tempo_bpm_like: Optional[float]
    is_silent: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "f0_med": self.f0_med,
            "f0_range": self.f0_range,
            "energy_mean": self.energy_mean,
            "zcr_mean": self.zcr_mean,
            "sc_mean": self.sc_mean,
            "tempo_bpm_like": self.tempo_bpm_like,
            "is_silent": self.is_silent,
        }

# ---------------------------
# 특징 추출
# ---------------------------
def _extract_features(file_path: str, target_sr: int = 16000) -> VoiceFeatures:
    """
    파일에서 음성 특징을 추출합니다.
    - 규칙 매핑을 하지 않습니다. (숫자만 계산)
    - downstream(LLM)에 넘겨 자연어 설명을 생성합니다.
    """
    # 로드
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    duration = _safe_float(librosa.get_duration(y=y, sr=sr), 6)

    # 무성 구간 판정(대충의 안전장치)
    rms = librosa.feature.rms(y=y).flatten()
    is_silent = bool(np.mean(rms) < 1e-3)

    # F0 추정 (YIN)
    try:
        f0 = librosa.yin(
            y,
            fmin=50,      # 사람 발화 대략 하한
            fmax=1100,    # 상한(가성 포함 넉넉히)
            sr=sr
        )
    except Exception:
        f0 = np.array([np.nan])

    f0_med = _safe_float(_nan_robust(f0, np.nanmedian, default=np.nan), 2)
    f0_p95 = _nan_robust(f0, lambda a: np.nanpercentile(a, 95), default=np.nan)
    f0_p05 = _nan_robust(f0, lambda a: np.nanpercentile(a, 5), default=np.nan)
    f0_range = _safe_float((f0_p95 - f0_p05) if (f0_p95 is not None and f0_p05 is not None) else np.nan, 2)

    # 에너지(RMS 평균)
    energy_mean = _safe_float(float(np.mean(rms)) if len(rms) else np.nan, 6)

    # ZCR 평균(무성/유성 경향)
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    zcr_mean = _safe_float(float(np.mean(zcr)) if len(zcr) else np.nan, 6)

    # 스펙트럴 센트로이드 평균(명료감/밝기 경향)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    sc_mean = _safe_float(float(np.mean(sc)) if len(sc) else np.nan, 2)

    # 템포 근사(bpm) — 발화의 리듬 경향치로만 사용(정확한 언어적 WPM 아님)
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr)
        tempo_bpm_like = _safe_float(float(tempo[0]) if tempo.size else np.nan, 2)
    except Exception:
        tempo_bpm_like = None

    return VoiceFeatures(
        duration_sec=duration,
        f0_med=f0_med,
        f0_range=f0_range,
        energy_mean=energy_mean,
        zcr_mean=zcr_mean,
        sc_mean=sc_mean,
        tempo_bpm_like=tempo_bpm_like,
        is_silent=is_silent
    )

# ---------------------------
# LLM 설명 생성(규칙 매핑 없음)
# ---------------------------
def _llm_describe_voice(features: VoiceFeatures) -> str:
    """
    숫자 특징을 LLM에 넘겨, 사람이 들은 것처럼 자연어 설명을 생성합니다.
    규칙 매핑은 하지 않습니다.
    """
    f = features.to_dict()

    system_msg = (
        "당신은 음성평가 전문가입니다. 숫자 지표에만 매몰되지 말고, "
        "실제 목소리를 들은 것처럼 구체적이되 과장 없이 한국어로 2~4문장 작성하세요. "
        "존중하는 어조를 유지하며, 편견/모욕/민감 정보는 피하세요."
    )

    user_msg = f"""
다음은 한 화자의 음성 분석 특징값입니다.

- 평균 피치(f0_med): {f.get('f0_med')}
- 피치 범위(f0_range): {f.get('f0_range')}
- 평균 에너지(RMS 평균): {f.get('energy_mean')}
- 무성-유성 경향(ZCR 평균): {f.get('zcr_mean')}
- 스펙트럴 센트로이드 평균(sc_mean): {f.get('sc_mean')}
- 리듬 유사 BPM(tempo_bpm_like): {f.get('tempo_bpm_like')}
- 길이(duration_sec): {f.get('duration_sec')}
- 무성 판정(is_silent): {f.get('is_silent')}

요청:
1) 이 목소리를 들었을 때 떠오르는 인상/성격/분위기를 사람처럼 묘사해 주세요.
2) 2~4문장, 중립/존중 어조.
3) '숫자상으로는' 같은 표현, 표/리스트 없이 자연스러운 문장만 출력.
4) 한국어로만 출력.
"""

    try:
        # 키가 없을 수도 있으므로 try/except로 감싸고 폴백 제공
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.8,
            max_tokens=220,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",    "content": user_msg},
            ],
        )
        text = (res.choices[0].message.content or "").strip()
        if not text or len(text) < 10:
            raise ValueError("empty llm response")
        return text
    except Exception as e:
        # 모델 호출 실패 시 안전 폴백(서비스 다운 방지)
        print(f"[analysis.py] LLM call failed: {type(e).__name__}: {e}")
        return "차분하고 안정적인 톤이 느껴집니다. 과장되지 않고 또렷하게 전달되어 신뢰감을 주는 인상입니다."

# ---------------------------
# (옵션) 우측 패널용 필드 생성기
# ---------------------------
def _build_visual_fields(features: VoiceFeatures) -> Dict[str, Any]:
    en_prompt = (
        "A character illustration in the style of Crayon Shin-chan, "
        "playful, exaggerated cartoon features, thick bold outlines, "
        "flat colors, childlike and humorous vibe"
    )
    negative = (
        "realistic, photorealistic, 3d render, high detail, "
        "distorted anatomy, extra limbs, text, watermark, logo"
    )
    style_tags = ["crayon shinchan", "cartoon", "flat-color", "bold-outline"]
    palette = ["#FFD966", "#FF9ACB", "#FF8A00", "#19B5FE"]  # 짱구풍 밝은 색상 팔레트
    seed = "playful Shinchan style cartoon portrait"

    return {
        "en_prompt": en_prompt,
        "negative": negative,
        "style_tags": style_tags,
        "palette": palette,
        "seed": seed,
    }

# === NEW: 느낌 기반 아바타 스타일 제안(정체성 단정 없음) ======================
def _avatar_suggestions(features: VoiceFeatures) -> Dict[str, Any]:
    """
    음향 특징을 바탕으로 '아바타 스타일 힌트'를 제안합니다.
    - 성별/나이 등 민감정보를 단정하지 않고, 디자인 가이드만 제공합니다.
    """
    f = features.to_dict()
    f0_med = f.get("f0_med") or 0.0
    f0_rng = f.get("f0_range") or 0.0
    energy  = f.get("energy_mean") or 0.0
    zcr     = f.get("zcr_mean") or 0.0
    tempo   = f.get("tempo_bpm_like") or 0.0

    # 기본값
    hair_length = "medium"      # short / medium / long
    face_shape  = "round"       # round / oval / heart / square
    vibe        = "soft"        # soft / lively / bold / calm
    extras      = []            # bangs, ribbon, headphones 등
    colors      = ["#FFD966", "#FF9ACB", "#FF8A00", "#19B5FE"]  # 짱구풍 밝은 팔레트

    # 피치 중심 제안(정체성 X, 단지 디자인 힌트)
    if f0_med >= 220:
        hair_length = "long"
        face_shape  = "round"
        vibe        = "soft"
        extras.append("wispy bangs")
    elif 150 <= f0_med < 220:
        hair_length = "medium"
        face_shape  = "oval"
        vibe        = "lively"
    else:
        hair_length = "short"
        face_shape  = "square"
        vibe        = "calm"

    # 피치 범위가 넓으면 표정/헤어 디테일을 조금 더
    if f0_rng >= 80:
        extras.append("expressive eyebrows")
        vibe = "lively" if vibe == "soft" else vibe

    # 에너지(볼륨감) 기반 색/소품
    if energy >= 0.03:
        extras.append("bold outline emphasis")
    else:
        colors = ["#FFF0B3", "#FFC8E7", "#FFD1A6", "#A7E2FF"]  # 더 파스텔

    # ZCR 높으면 공기 섞인 호흡감 → 가벼운 앞머리/리본 같은 소품
    if zcr >= 0.08:
        extras.append("light ribbon")

    # 템포가 빠르면 포즈를 경쾌하게
    pose = "playful pose" if tempo and tempo >= 110 else "relaxed pose"

    return {
        "hair_length": hair_length,
        "face_shape": face_shape,
        "vibe": vibe,
        "extras": extras,
        "suggested_palette": colors,
        "pose": pose,
        # 프롬프트 합성용 짧은 문구
        "avatar_hint_text": (
            f"{hair_length} hair, {face_shape} face, {vibe} vibe, "
            f"{pose}" + (", " + ", ".join(extras) if extras else "")
        ),
    }

# === NEW: 성별 + 스타일 힌트 추측 ============================
def _avatar_suggestions(features: VoiceFeatures) -> Dict[str, Any]:
    f = features.to_dict()
    f0_med = f.get("f0_med") or 0.0
    f0_rng = f.get("f0_range") or 0.0
    energy = f.get("energy_mean") or 0.0
    tempo  = f.get("tempo_bpm_like") or 0.0

    # ---- 기본값 ----
    gender_guess = "neutral"   # male / female / neutral
    hair_length = "medium"
    face_shape  = "round"
    vibe        = "soft"
    pose        = "relaxed pose"
    extras      = []

    # ---- 성별 추측(아주 단순 규칙) ----
    if f0_med >= 180:   # 평균 피치가 높은 경우
        gender_guess = "female"
        hair_length = "long"
        face_shape  = "oval"
        vibe        = "lively"
    elif f0_med <= 140: # 평균 피치가 낮은 경우
        gender_guess = "male"
        hair_length = "short"
        face_shape  = "square"
        vibe        = "calm"
    else:
        gender_guess = "neutral"
        hair_length = "medium"
        face_shape  = "round"
        vibe        = "soft"

    # ---- 범위/에너지/리듬 기반 추가 조정 ----
    if f0_rng >= 80:
        extras.append("expressive eyebrows")
        vibe = "energetic"
    if energy >= 0.035:
        extras.append("bold outline emphasis")
    if tempo and tempo >= 110:
        pose = "playful pose"
        extras.append("motion lines")
    elif tempo and tempo <= 90:
        pose = "calm standing pose"

    # ---- 최종 반환 ----
    return {
        "gender_guess": gender_guess,
        "hair_length": hair_length,
        "face_shape": face_shape,
        "vibe": vibe,
        "pose": pose,
        "extras": extras,
        "avatar_hint_text": (
            f"{gender_guess} character, {hair_length} hair, {face_shape} face, "
            f"{vibe} vibe, {pose}" + (", " + ", ".join(extras) if extras else "")
        ),
    }



# ---------------------------
# 외부에서 호출하는 메인 함수
# ---------------------------
# ✅ 항상 합성해서 반환: 프론트에서 체크박스 없이도 자동 반영
def analyze_voice(file_path: str) -> Dict[str, Any]:
    features = _extract_features(file_path)
    description_ko = _llm_describe_voice(features)

    # 짱구 스타일 프리셋 고정
    visual = _build_visual_fields(features)
    avatar = _avatar_suggestions(features)

    # 프롬프트에 항상 합성
    hint = avatar.get("avatar_hint_text")
    if hint:
        visual["en_prompt"] = f"{visual['en_prompt']}, {hint}"

    return {
        "features": features.to_dict(),
        "description_ko": description_ko,
        **visual,
        "avatar": avatar,
    }


