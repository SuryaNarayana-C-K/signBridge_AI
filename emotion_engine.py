"""
==============================================================================
SIGNBRIDGE AI — Multimodal Emotion Recognition Engine
==============================================================================
Tri-Modal Architecture:
    1. Speech Emotion Model (Audio Waveform): 50% weight (superb/wav2vec2-base-superb-er)
    2. NLP Emotion Model (Whisper Transcript): 30% weight (j-hartmann/emotion-english-distilroberta-base)
    3. Acoustic Prosody DSP (Physics & Dynamics): 20% weight (Comprehensive Prosody Features)

    Fusion Formula:
        P_final(e) = 0.50 * P_speech(e) + 0.30 * P_nlp(e) + 0.20 * P_prosody(e)
==============================================================================
"""

import os
import numpy as np

# Configurable Tri-Modal Fusion Weights
SPEECH_WEIGHT = 0.50
NLP_WEIGHT = 0.30
PROSODY_WEIGHT = 0.20

SPEECH_MODEL_ID = "superb/wav2vec2-base-superb-er"
NLP_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

ALL_SUPPORTED_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "FEAR", "SURPRISE", "DISGUST", "NEUTRAL"]

# Label Mappings to SignBridge Standard 7 Emotions
SPEECH_LABEL_MAP = {
    "neu": "NEUTRAL",
    "hap": "HAPPY",
    "ang": "ANGRY",
    "sad": "SAD",
}

NLP_LABEL_MAP = {
    "joy": "HAPPY",
    "sadness": "SAD",
    "anger": "ANGRY",
    "fear": "FEAR",
    "surprise": "SURPRISE",
    "disgust": "DISGUST",
    "neutral": "NEUTRAL",
}


# --------------------------------------------------------------------------
# 1. Model Loading & Caching
# --------------------------------------------------------------------------
def load_speech_emotion_model():
    """Loads the pretrained Wav2Vec2 speech emotion recognition model."""
    try:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

        import torch
        torch.set_num_threads(1)

        from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SPEECH_MODEL_ID)
        model = AutoModelForAudioClassification.from_pretrained(SPEECH_MODEL_ID)
        model.eval()
        return feature_extractor, model
    except Exception as exc:
        print(f"[SIGNBRIDGE AI] Speech emotion model load warning: {exc}")
        return None, None


def load_nlp_emotion_model():
    """Loads the pretrained DistilRoBERTa NLP emotion classification model."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(NLP_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(NLP_MODEL_ID)
        model.eval()
        return tokenizer, model
    except Exception as exc:
        print(f"[SIGNBRIDGE AI] NLP emotion model load warning: {exc}")
        return None, None


def load_pretrained_emotion_model():
    """Combined loader for Streamlit cache_resource.
    Returns (speech_feature_extractor, speech_model, nlp_tokenizer, nlp_model)."""
    speech_fe, speech_m = load_speech_emotion_model()
    nlp_tok, nlp_m = load_nlp_emotion_model()
    return (speech_fe, speech_m, nlp_tok, nlp_m)


def load_all_emotion_models():
    """Load and return all emotion models and processors (Speech Wav2Vec2 + NLP DistilRoBERTa).
    Returns (speech_feature_extractor, speech_model, nlp_tokenizer, nlp_model)."""
    return load_pretrained_emotion_model()


__all__ = [
    "load_speech_emotion_model",
    "load_nlp_emotion_model",
    "load_pretrained_emotion_model",
    "load_all_emotion_models",
    "extract_comprehensive_prosody",
    "predict_prosody_emotion",
    "predict_speech_emotion",
    "predict_ml_emotion",
    "predict_text_emotion",
    "fuse_emotions",
    "calculate_acoustic_intensity",
    "analyze_audio_emotion_multimodal",
    "analyze_audio_emotion_hybrid",
    "analyze_audio_emotion",
    "ALL_SUPPORTED_EMOTIONS",
    "SPEECH_WEIGHT",
    "NLP_WEIGHT",
    "PROSODY_WEIGHT",
]


# --------------------------------------------------------------------------
# 2. Comprehensive Acoustic-Prosodic Feature Engine (DSP)
# --------------------------------------------------------------------------
def extract_comprehensive_prosody(audio: np.ndarray, sr: int = 16000) -> dict:
    """Extract acoustic prosody and spectral features from raw 16kHz audio."""
    if len(audio) == 0:
        return {}

    duration_s = float(len(audio) / sr)
    frame_size = int(sr * 0.03)  # 30ms frames
    hop_size = int(sr * 0.015)   # 15ms hop

    # 1. Frame-level RMS energy & Dynamics
    rms_frames = []
    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i : i + frame_size]
        rms_frames.append(float(np.sqrt(np.mean(frame**2))))

    rms_arr = np.array(rms_frames) if rms_frames else np.array([float(np.sqrt(np.mean(audio**2)))])
    rms_mean = float(np.mean(rms_arr))
    rms_std = float(np.std(rms_arr))
    rms_max = float(np.max(rms_arr)) if len(rms_arr) else 0.0
    dynamic_range_db = float(20 * np.log10((rms_max + 1e-6) / (np.percentile(rms_arr, 10) + 1e-6)))

    # 2. Pitch / F0 via Autocorrelation & Peak Validation
    pitches = []
    min_lag = int(sr / 450)  # Max 450 Hz
    max_lag = int(sr / 75)   # Min 75 Hz
    voiced_thresh = max(0.015, 0.20 * rms_mean)

    for i in range(0, len(audio) - frame_size + 1, hop_size):
        frame = audio[i : i + frame_size]
        frame_rms = np.sqrt(np.mean(frame**2))
        if frame_rms > voiced_thresh:
            frame_centered = frame - np.mean(frame)
            autocorr = np.correlate(frame_centered, frame_centered, mode="full")
            autocorr = autocorr[len(frame_centered) - 1 :]
            if len(autocorr) > max_lag:
                search_region = autocorr[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    peak_val = autocorr[peak_idx]
                    if autocorr[0] > 0 and (peak_val / autocorr[0]) > 0.35:
                        freq = sr / peak_idx
                        if 75 <= freq <= 450:
                            pitches.append(freq)

    pitches = np.array(pitches) if pitches else np.array([])
    pitch_mean = float(np.mean(pitches)) if len(pitches) > 0 else 0.0
    pitch_std = float(np.std(pitches)) if len(pitches) > 0 else 0.0
    pitch_min = float(np.min(pitches)) if len(pitches) > 0 else 0.0
    pitch_max = float(np.max(pitches)) if len(pitches) > 0 else 0.0
    pitch_range = float(pitch_max - pitch_min) if len(pitches) > 0 else 0.0
    voiced_ratio = float(len(pitches) / (len(rms_arr) + 1e-6))

    jitter = float(np.mean(np.abs(np.diff(pitches))) / (pitch_mean + 1e-6)) if len(pitches) > 2 else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)

    # 3. Spectral Analysis via FFT
    n_fft = 512
    window = np.hanning(n_fft)
    stft_frames = []
    for i in range(0, len(audio) - n_fft + 1, hop_size):
        frame = audio[i : i + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        stft_frames.append(spectrum)

    stft_frames = np.array(stft_frames) if stft_frames else np.zeros((1, n_fft // 2 + 1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    centroids = []
    bandwidths = []
    fluxes = []
    high_freq_ratios = []
    hf_indices = np.where(freqs >= 1800)[0]

    for idx, spec in enumerate(stft_frames):
        spec_sum = np.sum(spec)
        if spec_sum > 1e-6:
            c = np.sum(freqs * spec) / spec_sum
            bw = np.sqrt(np.sum(((freqs - c) ** 2) * spec) / spec_sum)
            centroids.append(c)
            bandwidths.append(bw)
            if len(hf_indices) > 0:
                high_freq_ratios.append(np.sum(spec[hf_indices]) / spec_sum)
        else:
            centroids.append(0.0)
            bandwidths.append(0.0)
            high_freq_ratios.append(0.0)

        if idx > 0:
            flux = np.sqrt(np.mean((spec - stft_frames[idx - 1]) ** 2))
            fluxes.append(flux)

    spectral_centroid = float(np.mean(centroids)) if centroids else 0.0
    spectral_bandwidth = float(np.mean(bandwidths)) if bandwidths else 0.0
    spectral_flux = float(np.mean(fluxes)) if fluxes else 0.0
    high_freq_energy = float(np.mean(high_freq_ratios)) if high_freq_ratios else 0.0

    # 4. Temporal Rhythm & Speaking Dynamics
    silence_thresh = max(0.015, 0.15 * rms_mean)
    silent_frames = np.sum(rms_arr < silence_thresh)
    pause_ratio = float(silent_frames / (len(rms_arr) + 1e-6))

    peaks = 0
    for k in range(1, len(rms_arr) - 1):
        if rms_arr[k] > 0.03 and rms_arr[k] > rms_arr[k - 1] and rms_arr[k] > rms_arr[k + 1]:
            peaks += 1
    speaking_rate = float(peaks / duration_s) if duration_s > 0.4 else 0.0

    return {
        "duration_s": round(duration_s, 3),
        "rms_mean": round(rms_mean, 4),
        "rms_std": round(rms_std, 4),
        "rms_max": round(rms_max, 4),
        "dynamic_range_db": round(dynamic_range_db, 1),
        "pitch_mean": round(pitch_mean, 1),
        "pitch_std": round(pitch_std, 1),
        "pitch_min": round(pitch_min, 1),
        "pitch_max": round(pitch_max, 1),
        "pitch_range": round(pitch_range, 1),
        "voiced_ratio": round(voiced_ratio, 3),
        "jitter": round(jitter, 4),
        "zcr": round(zcr, 4),
        "spectral_centroid": round(spectral_centroid, 1),
        "spectral_bandwidth": round(spectral_bandwidth, 1),
        "spectral_flux": round(spectral_flux, 4),
        "high_freq_energy": round(high_freq_energy, 4),
        "pause_ratio": round(pause_ratio, 3),
        "speaking_rate": round(speaking_rate, 2),
    }


# --------------------------------------------------------------------------
# 3. Balanced Prosody DSP Classifier (Zero Neutral Bias)
# --------------------------------------------------------------------------
def predict_prosody_emotion(audio: np.ndarray, sr: int = 16000, overall_feat: dict = None) -> dict:
    """Calculates unbiased acoustic probability distribution from physical prosody features.
    Neutral bias removed: all emotions start at baseline 0.0 logit and compete on acoustic evidence."""
    if overall_feat is None:
        overall_feat = extract_comprehensive_prosody(audio, sr)

    duration_s = overall_feat.get("duration_s", len(audio) / sr)

    if duration_s < 0.35 or overall_feat.get("rms_mean", 0.0) < 0.006:
        return {
            "probabilities": {e: (1.0 / len(ALL_SUPPORTED_EMOTIONS)) for e in ALL_SUPPORTED_EMOTIONS},
            "features": overall_feat,
        }

    # Temporal windowing: 1.8s windows with 0.6s step
    win_len = int(sr * 1.8)
    step_len = int(sr * 0.6)

    if len(audio) <= win_len:
        windows = [audio]
    else:
        windows = [audio[i : i + win_len] for i in range(0, len(audio) - win_len + 1, step_len)]
        if not windows:
            windows = [audio]

    window_scores = []

    for win_audio in windows:
        f = extract_comprehensive_prosody(win_audio, sr)
        if not f:
            continue

        rms = f["rms_mean"]
        rms_max = f["rms_max"]
        p_mean = f["pitch_mean"]
        p_std = f["pitch_std"]
        p_max = f["pitch_max"]
        p_range = f["pitch_range"]
        centroid = f["spectral_centroid"]
        flux = f["spectral_flux"]
        hf_energy = f["high_freq_energy"]
        spk_rate = f["speaking_rate"]
        pause_r = f["pause_ratio"]
        jitter = f["jitter"]
        zcr = f["zcr"]

        # Logit accumulator initialized to 0.0 for all 7 emotions (No hardcoded bias!)
        scores = {e: 0.0 for e in ALL_SUPPORTED_EMOTIONS}

        # 1. ANGRY (High energy, sharp high-frequency burst, high flux, elevated pitch)
        if rms > 0.18:
            scores["ANGRY"] += (rms - 0.18) * 16.0
        if rms_max > 0.32:
            scores["ANGRY"] += (rms_max - 0.32) * 10.0
        if hf_energy > 0.10 and rms > 0.14:
            scores["ANGRY"] += (hf_energy - 0.10) * 18.0
        if flux > 0.022 and rms > 0.15:
            scores["ANGRY"] += min(4.0, flux * 50.0)
        if p_mean > 185 and rms > 0.15:
            scores["ANGRY"] += min(3.0, (p_mean - 185) / 28.0)

        # 2. HAPPY (High expressive pitch variation, bright centroid, elevated pitch, dynamic tempo)
        if p_std > 20 and p_mean > 160:
            scores["HAPPY"] += min(5.0, (p_std - 20) / 7.0 + (p_mean - 160) / 22.0)
        if p_range > 70 and p_mean > 165:
            scores["HAPPY"] += min(4.0, (p_range - 70) / 30.0)
        if centroid > 700 and hf_energy < 0.25 and rms > 0.07:
            scores["HAPPY"] += min(3.0, (centroid - 700) / 280.0)
        if spk_rate > 6.5 and 0.07 < rms < 0.38:
            scores["HAPPY"] += min(2.5, (spk_rate - 6.5) / 2.2)

        # 3. SAD (Subdued energy, low pitch < 135 Hz, high pauses, slow speaking rate)
        if rms < 0.075 and pause_r > 0.16:
            scores["SAD"] += (0.075 - rms) * 50.0 + (pause_r - 0.16) * 10.0
        if 0 < p_mean < 135 and rms < 0.09:
            scores["SAD"] += min(4.0, (135 - p_mean) / 16.0)
        if pause_r > 0.22 and rms < 0.09:
            scores["SAD"] += min(3.5, (pause_r - 0.22) * 12.0)
        if 0 < centroid < 450 and rms < 0.08:
            scores["SAD"] += min(2.5, (450 - centroid) / 100.0)

        # 4. FEAR (Pitch jitter/instability, elevated pitch, high ZCR, erratic breathiness)
        if jitter > 0.045 and p_mean > 185:
            scores["FEAR"] += min(4.0, (jitter - 0.045) * 40.0 + (p_mean - 185) / 35.0)
        if zcr > 0.13 and rms < 0.26:
            scores["FEAR"] += min(2.5, (zcr - 0.13) * 16.0)

        # 5. SURPRISE (Steep pitch jump / wide range, abrupt onset, high peak)
        if p_range > 130 and p_max > 240:
            scores["SURPRISE"] += min(4.5, (p_range - 130) / 30.0 + (p_max - 240) / 30.0)

        # 6. DISGUST (Low creaky pitch, slow articulation, low centroid with modest energy)
        if 0 < p_mean < 118 and rms > 0.06 and spk_rate < 5.2:
            scores["DISGUST"] += min(3.5, (118 - p_mean) / 18.0 + (5.2 - spk_rate))

        # 7. NEUTRAL (Moderate balanced pitch 110-175 Hz, moderate energy 0.06-0.17, low pitch variance)
        if 110 <= p_mean <= 175 and 0.05 <= rms <= 0.17 and p_std < 20:
            scores["NEUTRAL"] += min(3.5, (20 - p_std) / 5.0 + 1.0)
        elif 0.04 <= rms <= 0.14 and p_std < 16:
            scores["NEUTRAL"] += 1.5

        window_scores.append(scores)

    if not window_scores:
        return {
            "probabilities": {e: (1.0 / len(ALL_SUPPORTED_EMOTIONS)) for e in ALL_SUPPORTED_EMOTIONS},
            "features": overall_feat,
        }

    # Aggregate scores across temporal windows
    agg_scores = {e: 0.0 for e in ALL_SUPPORTED_EMOTIONS}
    for ws in window_scores:
        for e in agg_scores:
            agg_scores[e] += ws[e]

    score_vals = np.array([agg_scores[e] for e in ALL_SUPPORTED_EMOTIONS])
    # Softmax normalization
    exp_scores = np.exp(score_vals - np.max(score_vals))
    probs = exp_scores / np.sum(exp_scores)
    prob_dict = {e: float(probs[i]) for i, e in enumerate(ALL_SUPPORTED_EMOTIONS)}

    return {
        "probabilities": prob_dict,
        "features": overall_feat,
    }


# --------------------------------------------------------------------------
# 4. Pretrained Speech Emotion Inference (Audio Waveform)
# --------------------------------------------------------------------------
def predict_speech_emotion(audio: np.ndarray, feature_extractor, model, sr: int = 16000) -> dict:
    """Runs inference with superb/wav2vec2-base-superb-er across temporal windows.
    Returns normalized 7-emotion probability distribution."""
    if feature_extractor is None or model is None or len(audio) < 1600:  # < 0.1s
        return None

    try:
        import torch

        # Normalize audio amplitude to [-1, 1]
        max_val = np.max(np.abs(audio))
        norm_audio = (audio / max_val).astype(np.float32) if max_val > 1e-5 else audio.astype(np.float32)

        # Temporal windowing for longer audio: 2.5s windows with 1.0s step
        win_size = int(sr * 2.5)
        step_size = int(sr * 1.0)

        if len(norm_audio) <= win_size:
            audio_segments = [norm_audio]
        else:
            audio_segments = [
                norm_audio[i : i + win_size] 
                for i in range(0, len(norm_audio) - win_size + 1, step_size)
            ]
            if not audio_segments:
                audio_segments = [norm_audio]

        id2label = getattr(model.config, "id2label", {0: "neu", 1: "hap", 2: "ang", 3: "sad"})
        segment_probs_list = []

        for seg in audio_segments:
            inputs = feature_extractor(seg, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()

            if isinstance(probs, float):
                probs = [probs]

            seg_prob_dict = {}
            for idx, p in enumerate(probs):
                raw_label = id2label.get(idx, str(idx)).lower()
                clean_label = SPEECH_LABEL_MAP.get(raw_label, raw_label.upper())
                seg_prob_dict[clean_label] = float(p)
            segment_probs_list.append(seg_prob_dict)

        # Average across segments
        raw_speech_probs = {}
        for sp in segment_probs_list:
            for em, pr in sp.items():
                raw_speech_probs[em] = raw_speech_probs.get(em, 0.0) + (pr / len(segment_probs_list))

        # Project to 7 standard emotions
        full_speech_probs = {}
        for em in ALL_SUPPORTED_EMOTIONS:
            full_speech_probs[em] = raw_speech_probs.get(em, 0.0)

        # Normalize sum
        total_p = sum(full_speech_probs.values())
        if total_p > 0:
            full_speech_probs = {k: v / total_p for k, v in full_speech_probs.items()}

        return full_speech_probs
    except Exception as exc:
        print(f"[SIGNBRIDGE AI] Speech Emotion inference error: {exc}")
        return None


# Backwards-compatibility alias for speech emotion model inference
predict_ml_emotion = predict_speech_emotion


# --------------------------------------------------------------------------
# 5. Pretrained NLP Emotion Classification (Whisper Transcript)
# --------------------------------------------------------------------------
def predict_text_emotion(transcript: str, tokenizer, model) -> dict:
    """Classifies the Whisper transcript into emotion probabilities using DistilRoBERTa.
    No keyword rules: uses the actual deep learning representation."""
    if not transcript or not transcript.strip() or tokenizer is None or model is None:
        return None

    try:
        import torch

        inputs = tokenizer(transcript.strip(), return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        id2label = getattr(model.config, "id2label", {})
        nlp_probs = {e: 0.0 for e in ALL_SUPPORTED_EMOTIONS}

        for idx, p in enumerate(probs):
            raw_label = id2label.get(idx, str(idx)).lower()
            clean_label = NLP_LABEL_MAP.get(raw_label, "NEUTRAL")
            nlp_probs[clean_label] = nlp_probs.get(clean_label, 0.0) + float(p)

        # Normalize to sum = 1.0
        total_p = sum(nlp_probs.values())
        if total_p > 0:
            nlp_probs = {k: v / total_p for k, v in nlp_probs.items()}

        return nlp_probs
    except Exception as exc:
        print(f"[SIGNBRIDGE AI] NLP Emotion inference error: {exc}")
        return None


# --------------------------------------------------------------------------
# 6. Tri-Modal Fusion & Conflict Handling
# --------------------------------------------------------------------------
def fuse_emotions(
    speech_probs: dict,
    nlp_probs: dict,
    prosody_probs: dict,
    speech_weight: float = SPEECH_WEIGHT,
    nlp_weight: float = NLP_WEIGHT,
    prosody_weight: float = PROSODY_WEIGHT,
) -> tuple[dict, str, float]:
    """Combines Speech Emotion (50%), NLP Transcript (30%), and Prosody DSP (20%).
    
    Handles missing models cleanly with dynamic weight renormalization.
    Returns (fused_probabilities_dict, top_emotion, authentic_confidence_pct)."""
    
    active_weights = {}
    active_sources = {}

    if speech_probs is not None:
        active_weights["speech"] = speech_weight
        active_sources["speech"] = speech_probs
    if nlp_probs is not None:
        active_weights["nlp"] = nlp_weight
        active_sources["nlp"] = nlp_probs
    if prosody_probs is not None:
        active_weights["prosody"] = prosody_weight
        active_sources["prosody"] = prosody_probs

    if not active_sources:
        # Equal uniform distribution fallback
        uniform = {e: 1.0 / len(ALL_SUPPORTED_EMOTIONS) for e in ALL_SUPPORTED_EMOTIONS}
        return uniform, "UNCERTAIN", 30.0

    # Renormalize active weights to sum = 1.0
    total_w = sum(active_weights.values())
    norm_weights = {k: v / total_w for k, v in active_weights.items()}

    fused = {e: 0.0 for e in ALL_SUPPORTED_EMOTIONS}
    for source_name, src_probs in active_sources.items():
        w = norm_weights[source_name]
        for e in ALL_SUPPORTED_EMOTIONS:
            fused[e] += w * src_probs.get(e, 0.0)

    # Renormalize fused sum
    sum_f = sum(fused.values())
    if sum_f > 0:
        fused = {k: v / sum_f for k, v in fused.items()}

    # Ranked predictions
    sorted_emotions = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_prob = sorted_emotions[0]
    second_emotion, second_prob = sorted_emotions[1] if len(sorted_emotions) > 1 else ("NEUTRAL", 0.0)

    # Authentic Confidence Calculation (Directly from probability & separation margin)
    margin = top_prob - second_prob
    authentic_confidence = float(top_prob * 100.0)

    # Uncertainty / Conflict Resolution:
    # If the top probability is too low (< 28%) or the top 2 emotions are in a near-tie (< 5% margin with low prob)
    if top_prob < 0.28 or (top_prob < 0.35 and margin < 0.05):
        final_label = "UNCERTAIN"
    else:
        final_label = top_emotion

    return fused, final_label, authentic_confidence


# --------------------------------------------------------------------------
# 7. Independent Acoustic Intensity Calculation
# --------------------------------------------------------------------------
def calculate_acoustic_intensity(features: dict) -> tuple[float, str]:
    """Calculates physical vocal intensity independently from emotion label.
    Evaluates acoustic energy (RMS), dynamic range (dB), and pitch deviation."""
    rms_mean = features.get("rms_mean", 0.10)
    rms_max = features.get("rms_max", 0.15)
    dyn_db = features.get("dynamic_range_db", 18.0)
    pitch_std = features.get("pitch_std", 20.0)

    rms_factor = min(1.0, rms_mean / 0.28)
    peak_factor = min(1.0, rms_max / 0.45)
    dyn_factor = min(1.0, max(0.0, dyn_db - 8.0) / 28.0)
    pitch_factor = min(1.0, pitch_std / 55.0)

    raw_intensity = (0.40 * rms_factor + 0.25 * peak_factor + 0.20 * dyn_factor + 0.15 * pitch_factor) * 100.0
    intensity_pct = float(np.clip(raw_intensity, 12.0, 98.0))

    if intensity_pct >= 68.0:
        intensity_tier = "HIGH"
    elif intensity_pct >= 38.0:
        intensity_tier = "MODERATE"
    else:
        intensity_tier = "LOW"

    return intensity_pct, intensity_tier


# --------------------------------------------------------------------------
# 8. Main Public Multimodal Emotion Analyzer
# --------------------------------------------------------------------------
def analyze_audio_emotion_multimodal(
    audio: np.ndarray,
    sr: int = 16000,
    transcript: str = "",
    models_tuple: tuple = None,
) -> dict:
    """Comprehensive Tri-Modal Emotion Recognition Entrypoint.
    
    Inputs:
        audio: 16 kHz mono float32 numpy array
        sr: sampling rate (16000)
        transcript: Whisper ASR text transcript
        models_tuple: cached (speech_fe, speech_model, nlp_tok, nlp_model)
    """
    duration_s = float(len(audio) / sr) if sr > 0 else 0.0

    # 1. Quality & Silence Check
    if duration_s < 0.35:
        return {
            "emotion": "UNCERTAIN",
            "confidence": 25.0,
            "intensity": 15.0,
            "intensity_tier": "LOW",
            "engine": "Multimodal AI (Audio too short)",
            "uncertainty_note": "Audio clip too short (< 0.35s) for reliable emotion classification.",
            "features": {"duration_s": round(duration_s, 2), "rms_mean": 0.0, "pitch_mean": 0.0, "speaking_rate": 0.0},
            "probabilities": {e: (100.0 if e == "NEUTRAL" else 0.0) for e in ALL_SUPPORTED_EMOTIONS},
            "breakdown": {"speech": {}, "nlp": {}, "prosody": {}},
        }

    features = extract_comprehensive_prosody(audio, sr)
    if features.get("rms_mean", 0.0) < 0.005:
        return {
            "emotion": "UNCERTAIN",
            "confidence": 20.0,
            "intensity": 10.0,
            "intensity_tier": "LOW",
            "engine": "Multimodal AI (Silence Detected)",
            "uncertainty_note": "Audio volume too low / silence detected.",
            "features": features,
            "probabilities": {e: (100.0 if e == "NEUTRAL" else 0.0) for e in ALL_SUPPORTED_EMOTIONS},
            "breakdown": {"speech": {}, "nlp": {}, "prosody": {}},
        }

    # Unpack cached models
    if models_tuple is not None and len(models_tuple) == 4:
        speech_fe, speech_m, nlp_tok, nlp_m = models_tuple
    elif models_tuple is not None and len(models_tuple) == 2:
        speech_fe, speech_m = models_tuple
        nlp_tok, nlp_m = None, None
    else:
        speech_fe, speech_m, nlp_tok, nlp_m = None, None, None, None

    # 2. Speech Emotion Inference (Audio 50%)
    speech_probs = predict_speech_emotion(audio, speech_fe, speech_m, sr=sr)

    # 3. NLP Emotion Inference (Text 30%)
    nlp_probs = predict_text_emotion(transcript, nlp_tok, nlp_m)

    # 4. Prosody DSP Inference (Acoustics 20%)
    dsp_result = predict_prosody_emotion(audio, sr=sr, overall_feat=features)
    prosody_probs = dsp_result.get("probabilities", {})

    # 5. Tri-Modal Fusion
    fused_probs, top_emotion, confidence = fuse_emotions(
        speech_probs=speech_probs,
        nlp_probs=nlp_probs,
        prosody_probs=prosody_probs,
        speech_weight=SPEECH_WEIGHT,
        nlp_weight=NLP_WEIGHT,
        prosody_weight=PROSODY_WEIGHT,
    )

    # 6. Independent Intensity
    intensity_pct, intensity_tier = calculate_acoustic_intensity(features)

    # Prepare detailed percentage breakdown for hackathon transparency
    prob_percentages = {k: round(v * 100.0, 1) for k, v in fused_probs.items()}
    speech_percentages = {k: round(v * 100.0, 1) for k, v in speech_probs.items()} if speech_probs else {}
    nlp_percentages = {k: round(v * 100.0, 1) for k, v in nlp_probs.items()} if nlp_probs else {}
    prosody_percentages = {k: round(v * 100.0, 1) for k, v in prosody_probs.items()} if prosody_probs else {}

    # Build active engine description
    sources = []
    if speech_probs:
        sources.append("Speech 50%")
    if nlp_probs:
        sources.append("NLP 30%")
    if prosody_probs:
        sources.append("Prosody 20%")
    engine_desc = f"Multimodal AI ({' · '.join(sources)})" if sources else "Prosody DSP Fallback"

    return {
        "emotion": top_emotion,
        "confidence": round(confidence, 1),
        "intensity": round(intensity_pct, 1),
        "intensity_tier": intensity_tier,
        "engine": engine_desc,
        "transcript": transcript,
        "features": features,
        "probabilities": prob_percentages,
        "breakdown": {
            "speech": speech_percentages,
            "nlp": nlp_percentages,
            "prosody": prosody_percentages,
        },
    }


# Backwards compatibility aliases
analyze_audio_emotion_hybrid = analyze_audio_emotion_multimodal
analyze_audio_emotion = analyze_audio_emotion_multimodal
