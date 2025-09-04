# analysis.py
from pathlib import Path
from typing import Optional, Dict

def analyze_voice(wav_path: Path, transcript: Optional[str] = None) -> Dict:
    """
    음성 파일을 분석해 피치/에너지/톤 관련 라벨과 지표를 반환합니다.
    - librosa/numpy는 지연 임포트
    - NaN/inf/빈 배열 방어 처리
    - 무음 오탐을 줄이기 위해 RMS 75퍼센타일 + 피크레벨 기반 보정
    - 반환값은 dict로 일정한 스키마 유지
    - transcript는 향후 말하기 속도/내용 분석에 활용 가능 (현재는 미사용)
    """
    try:
        # --- 지연 임포트 ---
        import numpy as np
        import librosa

        # --- 음성 로드 (16kHz mono) ---
        y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        if y is None or len(y) == 0:
            raise ValueError("오디오가 비어 있습니다.")

        # 비정상 값 방어
        if np.isnan(y).any() or np.isinf(y).any():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 특징 추출 ---
        # 1) F0 (yin 기반)
        try:
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[np.isfinite(f0)] if f0 is not None else np.array([])
            f0_med = float(np.nanmedian(f0)) if f0.size > 0 else 0.0
        except Exception:
            f0_med = 0.0

        # 2) RMS, ZCR, Spectral Centroid
        try:
            rms = librosa.feature.rms(y=y)[0]
        except Exception:
            rms = np.array([0.0], dtype=float)
        try:
            zcr = librosa.feature.zero_crossing_rate(y)[0]
        except Exception:
            zcr = np.array([0.0], dtype=float)
        try:
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        except Exception:
            sc = np.array([0.0], dtype=float)

        # 안전 평균 함수
        def _safe_mean(a, default=0.0):
            if a is None or a.size == 0:
                return default
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return float(np.mean(a))

        energy_mean = _safe_mean(rms)
        zcr_mean = _safe_mean(zcr)
        sc_mean = _safe_mean(sc)

        # --- 무음 보정 ---
        rms_p75 = float(np.percentile(np.nan_to_num(rms, nan=0.0), 75)) if rms.size > 0 else 0.0
        peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
        looks_silent = (rms_p75 < 1e-3) and (peak < 2e-2)

        # --- 라벨링 ---
        # 피치
        if f0_med == 0:
            pitch_label = "안정적인"
        elif f0_med < 150:
            pitch_label = "낮고 안정적인"
        elif f0_med < 220:
            pitch_label = "중간 높이의"
        else:
            pitch_label = "높고 또렷한"

        # 에너지
        if energy_mean < 0.02:
            energy_label = "차분한"
        elif energy_mean < 0.05:
            energy_label = "안정적이며 편안한"
        else:
            energy_label = "활기 있고 에너지가 느껴지는"

        # 톤
        if zcr_mean < 0.04:
            tone_label = "따뜻하고 부드러운"
        elif zcr_mean < 0.08:
            tone_label = "부드럽지만 또렷한"
        else:
            tone_label = "선명하고 경쾌한"

        # 밝기 보정
        if sc_mean > 2500:
            tone_label = "밝고 선명한"

        # --- 설명 구성 ---
        description_lines = [
            f"{pitch_label} 목소리를 가진 사람.",
            f"{energy_label} 인상이며,",
            f"{tone_label} 느낌입니다."
        ]
        description = "\n".join(description_lines)
        tags = [pitch_label, energy_label, tone_label]

        # --- 무음 처리 ---
        if looks_silent:
            return {
                "f0_med": round(f0_med, 4),
                "energy_mean": round(energy_mean, 6),
                "zcr_mean": round(zcr_mean, 6),
                "sc_mean": round(sc_mean, 2),
                "description": "",
                "tags": [],
                "is_silent": True,
            }

        # --- 정상 반환 ---
        return {
            "f0_med": round(f0_med, 4),
            "energy_mean": round(energy_mean, 6),
            "zcr_mean": round(zcr_mean, 6),
            "sc_mean": round(sc_mean, 2),
            "description": description,
            "tags": tags,
            "is_silent": False,
        }

    except Exception as e:
        return {
            "f0_med": 0.0,
            "energy_mean": 0.0,
            "zcr_mean": 0.0,
            "sc_mean": 0.0,
            "description": "",
            "tags": [],
            "is_silent": True,
            "error": str(e),
        }
