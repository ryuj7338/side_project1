from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path

import numpy as np
import librosa

from dotenv import load_dotenv
from openai import OpenAI

# ============================
# 환경 & OpenAI
# ============================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.sample")  # sample만 읽던 버전도 흡수  :contentReference[oaicite:18]{index=18}
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[analysis.py] WARNING: OPENAI_API_KEY not set — LLM 설명은 폴백 사용")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # :contentReference[oaicite:19]{index=19}

# ============================
# 유틸
# ============================
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
        }  # :contentReference[oaicite:20]{index=20}

# ============================
# 특징 추출
# ============================
def _extract_features(file_path: str, target_sr: int = 16000) -> VoiceFeatures:
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    duration = _safe_float(librosa.get_duration(y=y, sr=sr), 6)

    rms = librosa.feature.rms(y=y).flatten()
    is_silent = bool(np.mean(rms) < 1e-3)

    try:
        f0 = librosa.yin(y, fmin=50, fmax=1100, sr=sr)
    except Exception:
        f0 = np.array([np.nan])

    f0_med = _safe_float(_nan_robust(f0, np.nanmedian, default=np.nan), 2)
    f0_p95 = _nan_robust(f0, lambda a: np.nanpercentile(a, 95), default=np.nan)
    f0_p05 = _nan_robust(f0, lambda a: np.nanpercentile(a, 5), default=np.nan)
    f0_range = _safe_float((f0_p95 - f0_p05) if (f0_p95 is not None and f0_p05 is not None) else np.nan, 2)

    energy_mean = _safe_float(float(np.mean(rms)) if len(rms) else np.nan, 6)
    zcr = librosa.feature.zero_crossing_rate(y=y).flatten()
    zcr_mean = _safe_float(float(np.mean(zcr)) if len(zcr) else np.nan, 6)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()
    sc_mean = _safe_float(float(np.mean(sc)) if len(sc) else np.nan, 2)

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
    )  # :contentReference[oaicite:21]{index=21}

# ============================
# 태그 생성(프롬프트 힌트)
# ============================
def _simple_tags_from_features(f: VoiceFeatures) -> List[str]:
    tags: List[str] = []
    if f.f0_med is None:
        pass
    elif f.f0_med < 150:
        tags.append("낮고 안정적인")
    elif f.f0_med < 220:
        tags.append("중간 높이")
    else:
        tags.append("높고 또렷한")

    if (f.energy_mean or 0) < 0.02:
        tags.append("차분한")
    elif (f.energy_mean or 0) < 0.05:
        tags.append("안정적")
    else:
        tags.append("활기찬")

    if (f.zcr_mean or 0) < 0.04:
        tags.append("부드러운")
    elif (f.zcr_mean or 0) < 0.08:
        tags.append("또렷한")
    else:
        tags.append("경쾌한")

    if (f.sc_mean or 0) > 2500:
        tags.append("빛감이 있는")
    return tags  # :contentReference[oaicite:22]{index=22}

# ============================
# LLM 설명(한국어), 실패 시 폴백
# ============================
def _llm_describe_voice(features: VoiceFeatures) -> str:
    f = features.to_dict()
    system_msg = (
        "당신은 음성평가 전문가입니다. 숫자를 참고하되, 실제 목소리를 들은 듯 한국어로 2~4문장 설명하세요. "
        "과장·편견·민감정보는 피하고 존중하는 어조를 사용하세요."
    )
    user_msg = f"""
- 평균 피치(f0_med): {f.get('f0_med')}
- 피치 범위(f0_range): {f.get('f0_range')}
- 평균 에너지(RMS 평균): {f.get('energy_mean')}
- ZCR 평균: {f.get('zcr_mean')}
- 스펙트럴 센트로이드 평균: {f.get('sc_mean')}
- 리듬 유사 BPM: {f.get('tempo_bpm_like')}
- 길이: {f.get('duration_sec')}
- 무성 판정: {f.get('is_silent')}

요청: 자연스러운 한국어 문장만 2~4문장으로 출력.
"""
    if not client:
        return "차분하고 안정적인 톤이 느껴집니다. 과장되지 않고 또렷하게 전달되어 신뢰감을 주는 인상입니다."
    try:
        res = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.8, max_tokens=220,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
        )
        text = (res.choices[0].message.content or "").strip()
        if not text or len(text) < 10:
            raise ValueError("empty llm response")
        return text
    except Exception:
        return "차분하고 안정적인 톤이 느껴집니다. 과장되지 않고 또렷하게 전달되어 신뢰감을 주는 인상입니다."  # :contentReference[oaicite:23]{index=23}

# ============================
# 기본 우측 패널 필드
# ============================
def _build_visual_fields(features: VoiceFeatures) -> Dict[str, Any]:
    en_prompt = (
        "A character portrait in soft ambient light, conveying an energetic yet calm presence; "
        "cinematic composition; subtle depth of field; clean lines, coherent anatomy."
    )
    negative = "low quality, blurry, extra limbs, distorted anatomy, deformed, text, watermark, logo, frame, oversaturated, underexposed, jpeg artifacts"
    style_tags = "semi-realistic, cinematic, clean-lines"
    palette = ["#A0C4FF", "#BDB2FF", "#FFC6FF", "#FFADAD"]
    seed = "character portrait in soft ambient light"
    return {
        "en_prompt": en_prompt,
        "negative": negative,
        "style_tags": style_tags,
        "palette": palette,
        "seed": seed,
    }  # :contentReference[oaicite:24]{index=24}

# ============================
# 외부 API: 호환 스키마로 반환
# ============================
def analyze_voice(file_path: str) -> Dict[str, Any]:
    """
    반환 스키마(호환):
      {
        "features": {...},            # 수치
        "description_ko": "...",      # 한국어 설명
        "tags": [...],                # 간단 태그(없어도 프론트에서 기본값 처리)
        "en_prompt"/"negative"/...    # 우측 패널 기본값
      }
    """
    features = _extract_features(file_path)
    description_ko = _llm_describe_voice(features)
    tags = _simple_tags_from_features(features)  # (없던 태그 생성 추가)  :contentReference[oaicite:25]{index=25}

    result: Dict[str, Any] = {
        "features": features.to_dict(),
        "description_ko": description_ko,
        "tags": tags,
    }
    result.update(_build_visual_fields(features))
    return result  # :contentReference[oaicite:26]{index=26}
