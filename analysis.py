from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import librosa

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.sample")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[analysis.py] WARNING: OPENAI_API_KEY is not set. Falling back to default description on LLM failure.")

client = OpenAI(api_key=OPENAI_API_KEY)

def _safe_float(x, ndigits: int = 5) -> Optional[float]:
    try: return round(float(x), ndigits)
    except Exception: return None

def _nan_robust(values: np.ndarray, fn, default=None):
    try:
        v = fn(values[~np.isnan(values)])
        if np.isnan(v): return default
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
        duration_sec=duration, f0_med=f0_med, f0_range=f0_range,
        energy_mean=energy_mean, zcr_mean=zcr_mean, sc_mean=sc_mean,
        tempo_bpm_like=tempo_bpm_like, is_silent=is_silent
    )

def _llm_describe_voice(features: VoiceFeatures) -> str:
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
1) 이 목소리를 들었을 때 떠오르는 인상/성격/분위를 2~4문장으로 설명.
2) 중립/존중 어조, 편견/모욕/민감정보 회피.
3) 표/리스트 없이 자연스러운 문장만.
4) 한국어로만 작성.
"""
    try:
        res = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.8, max_tokens=220,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}],
        )
        text = (res.choices[0].message.content or "").strip()
        if not text or len(text) < 10:
            raise ValueError("empty llm response")
        return text
    except Exception as e:
        print(f"[analysis.py] LLM call failed: {type(e).__name__}: {e}")
        return "차분하고 안정적인 톤이 느껴집니다. 과장되지 않고 또렷하게 전달되어 신뢰감을 주는 인상입니다."

def _build_visual_fields(features: VoiceFeatures) -> Dict[str, Any]:
    """
    백엔드가 EN 프롬프트를 만든다(짱구풍 무드).
    - 특정 IP 보호 대상물을 직접 모사하지 않도록 '분위기/특징'을 서술형으로 유지.
    """
    en_prompt = (
        "A playful, mischievous child character portrait in a 2D cartoon style, "
        "thick black outlines, flat bold colors, simple geometric shapes, "
        "exaggerated facial expressions, comedic proportions, minimal shading, "
        "clean background; cheerful, cheeky vibe; Shin-chan-like (Jjanggu-style) mood."
    )
    negative = (
        "photorealistic, 3D render, detailed realism, blurry, low quality, extra limbs, "
        "distorted anatomy, watermark, logo, text, frame, oversaturated, underexposed, jpeg artifacts"
    )
    style_tags = ["짱구풍", "2D-cartoon", "thick-outline", "flat-colors", "cheeky", "simple-shapes", "exaggerated-expression"]
    palette = ["#FFD166", "#EF476F", "#06D6A0", "#118AB2", "#073B4C"]
    seed = "cheeky 2D cartoon child with thick outlines and flat colors"

    return {
        "en_prompt": en_prompt,
        "negative_prompt": negative,
        "style_tags": style_tags,
        "palette": palette,
        "seed_idea": seed,
    }

def analyze_voice(file_path: str) -> Dict[str, Any]:
    features = _extract_features(file_path)
    description_ko = _llm_describe_voice(features)

    result: Dict[str, Any] = {
        "features": features.to_dict(),
        "description_ko": description_ko,
    }
    result.update(_build_visual_fields(features))
    return result
