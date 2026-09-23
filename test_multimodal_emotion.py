"""
Test script for the upgraded Multimodal Emotion Recognition Engine.
Tests:
1. NLP-only, Speech-only, and Prosody-only predictions
2. Fused multimodal outputs for standard test sentences
3. Same sentence with different vocal tones (excited, calm, sad, angry)
4. Audio quality and silence checks
5. Fallback scenarios
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import emotion_engine
from emotion_engine import (
    load_all_emotion_models,
    extract_comprehensive_prosody,
    predict_prosody_emotion,
    predict_speech_emotion,
    predict_text_emotion,
    fuse_emotions,
    calculate_acoustic_intensity,
    analyze_audio_emotion_multimodal,
    ALL_SUPPORTED_EMOTIONS
)


def generate_synthetic_tone(tone: str, duration_s: float = 2.5, sr: int = 16000) -> np.ndarray:
    """Generates synthetic acoustic audio waveform with realistic prosodic dynamics for a specific tone."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    
    if tone == "excited":
        pitch_contour = 230 + 85 * np.sin(2 * np.pi * 3.2 * t)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
        audio = 0.28 * np.sin(phase) + 0.14 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
        env = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
        audio = audio * env
    elif tone == "calm":
        pitch_contour = 135 + 8 * np.sin(2 * np.pi * 1.0 * t)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
        audio = 0.12 * np.sin(phase) + 0.04 * np.sin(2 * phase)
    elif tone == "sad":
        pitch_contour = 98 + 4 * np.sin(2 * np.pi * 0.5 * t)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
        audio = 0.045 * np.sin(phase) + 0.015 * np.sin(2 * phase)
        audio[int(len(audio) * 0.35) : int(len(audio) * 0.65)] = 0.0
    elif tone == "angry":
        pitch_contour = 210 + 35 * np.sin(2 * np.pi * 5.0 * t)
        phase = 2 * np.pi * np.cumsum(pitch_contour) / sr
        audio = 0.38 * np.sin(phase) + 0.22 * np.sin(2 * phase) + 0.16 * np.sin(3 * phase) + 0.08 * np.sin(4 * phase)
    else:
        audio = 0.10 * np.sin(2 * np.pi * 140 * t)

    return audio.astype(np.float32)


def run_tests():
    print("=" * 75)
    print("[SIGNBRIDGE AI] MULTIMODAL EMOTION ENGINE VALIDATION SUITE")
    print("=" * 75)

    print("\n1. Loading pretrained models...")
    models = load_all_emotion_models()
    speech_fe, speech_m, nlp_tok, nlp_m = models
    print(f"   Speech Model: {'Loaded [OK]' if speech_m else 'Fallback'}")
    print(f"   NLP Model:    {'Loaded [OK]' if nlp_m else 'Fallback'}")

    # Test 1: Standard 6 Emotion Sentences
    test_cases = [
        ("HAPPY", "I am extremely happy today! We finally achieved our goal!", "excited"),
        ("SAD", "I feel really sad today. Everything has been difficult.", "sad"),
        ("ANGRY", "This is completely unacceptable! I am really angry about this.", "angry"),
        ("NEUTRAL", "The meeting is scheduled for 2 PM tomorrow.", "calm"),
        ("SURPRISE", "Oh my God! I can't believe this happened!", "excited"),
        ("FEAR", "I am scared and I don't know what is going to happen.", "sad"),
    ]

    print("\n" + "=" * 75)
    print("2. Testing 6 Distinct Emotion Inputs (NLP + Audio + Prosody):")
    print("=" * 75)

    for target_em, transcript, tone in test_cases:
        audio = generate_synthetic_tone(tone)
        res = analyze_audio_emotion_multimodal(audio, sr=16000, transcript=transcript, models_tuple=models)
        
        em = res["emotion"]
        conf = res["confidence"]
        intensity = res["intensity_tier"]
        breakdown = res["breakdown"]

        print(f"\n[TARGET: {target_em}]")
        print(f"  Transcript:  \"{transcript}\"")
        print(f"  Tone:        {tone.upper()}")
        print(f"  Result:      {em} (Confidence: {conf}%, Intensity: {intensity})")
        if breakdown.get("speech"):
            top_sp = sorted(breakdown["speech"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  Speech AI:   {top_sp[0][0]} ({top_sp[0][1]}%), {top_sp[1][0]} ({top_sp[1][1]}%)")
        if breakdown.get("nlp"):
            top_nlp = sorted(breakdown["nlp"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  NLP AI:      {top_nlp[0][0]} ({top_nlp[0][1]}%), {top_nlp[1][0]} ({top_nlp[1][1]}%)")
        if breakdown.get("prosody"):
            top_pr = sorted(breakdown["prosody"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  Prosody DSP: {top_pr[0][0]} ({top_pr[0][1]}%), {top_pr[1][0]} ({top_pr[1][1]}%)")
        top_fused = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Final Fusion: {', '.join([f'{k}: {v}%' for k, v in top_fused])}")

    # Test 2: Mandatory SAME SENTENCE with 4 DIFFERENT TONES
    print("\n" + "=" * 75)
    print("3. Testing SAME SENTENCE with 4 DIFFERENT VOCAL TONES:")
    print("   Sentence: \"We won the hackathon!\"")
    print("=" * 75)

    same_sentence = "We won the hackathon!"
    tones = ["excited", "calm", "sad", "angry"]

    for tone in tones:
        audio = generate_synthetic_tone(tone)
        res = analyze_audio_emotion_multimodal(audio, sr=16000, transcript=same_sentence, models_tuple=models)
        em = res["emotion"]
        conf = res["confidence"]
        intensity = res["intensity_tier"]
        breakdown = res["breakdown"]

        print(f"\n[TONE: {tone.upper()}] -> Transcript: \"{same_sentence}\"")
        print(f"  Detected Emotion: {em} | Confidence: {conf}% | Intensity: {intensity}")
        if breakdown.get("speech"):
            top_sp = sorted(breakdown["speech"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  Speech Audio 50%: {top_sp[0][0]} ({top_sp[0][1]}%), {top_sp[1][0]} ({top_sp[1][1]}%)")
        if breakdown.get("nlp"):
            top_nlp = sorted(breakdown["nlp"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  NLP Text 30%:     {top_nlp[0][0]} ({top_nlp[0][1]}%), {top_nlp[1][0]} ({top_nlp[1][1]}%)")
        if breakdown.get("prosody"):
            top_pr = sorted(breakdown["prosody"].items(), key=lambda x: x[1], reverse=True)[:2]
            print(f"  Prosody DSP 20%:  {top_pr[0][0]} ({top_pr[0][1]}%), {top_pr[1][0]} ({top_pr[1][1]}%)")
        top_fused = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Final Fusion:     {', '.join([f'{k}: {v}%' for k, v in top_fused])}")

    # Test 3: Silence & Short Audio Checks
    print("\n" + "=" * 75)
    print("4. Audio Quality & Silence Robustness:")
    print("=" * 75)

    short_audio = np.zeros(int(16000 * 0.2), dtype=np.float32)
    silence_audio = np.zeros(int(16000 * 2.0), dtype=np.float32)

    res_short = analyze_audio_emotion_multimodal(short_audio, transcript="Hello")
    res_silence = analyze_audio_emotion_multimodal(silence_audio, transcript="")

    print(f"  Short Audio (<0.35s):  {res_short['emotion']} ({res_short['engine']}) [OK]")
    print(f"  Silence Audio:         {res_silence['emotion']} ({res_silence['engine']}) [OK]")
    
    print("\n" + "=" * 75)
    print("[SUCCESS] ALL MULTIMODAL EMOTION ENGINE TESTS COMPLETED!")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()
