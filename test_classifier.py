"""
Test & Validation Script for Normalized Landmark Feature Extraction & Classifier.
Tests accuracy, noise robustness, scale invariance, and synthetic augmentation.
"""

import math
import json
import numpy as np

# Landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

def create_synthetic_hand(sign_type: str, noise_level=0.01, scale=1.0, rot_deg=0.0):
    """Generates 21 3D landmarks for synthetic testing of hand postures."""
    lm = np.zeros((21, 3), dtype=np.float32)
    # Wrist at (0.5, 0.7, 0.0)
    lm[WRIST] = [0.5, 0.7, 0.0]
    
    # Palm MCPs
    lm[INDEX_MCP] = [0.44, 0.52, 0.0]
    lm[MIDDLE_MCP] = [0.50, 0.50, 0.0]
    lm[RING_MCP] = [0.56, 0.52, 0.0]
    lm[PINKY_MCP] = [0.61, 0.55, 0.0]
    lm[THUMB_CMC] = [0.42, 0.64, 0.0]
    lm[THUMB_MCP] = [0.38, 0.58, 0.0]

    if sign_type == "HELLO":
        # All 5 fingers extended and spread
        lm[THUMB_IP] = [0.34, 0.52, 0.0]
        lm[THUMB_TIP] = [0.30, 0.46, 0.0]
        lm[INDEX_PIP] = [0.43, 0.44, 0.0]; lm[INDEX_DIP] = [0.42, 0.38, 0.0]; lm[INDEX_TIP] = [0.41, 0.32, 0.0]
        lm[MIDDLE_PIP] = [0.50, 0.41, 0.0]; lm[MIDDLE_DIP] = [0.50, 0.34, 0.0]; lm[MIDDLE_TIP] = [0.50, 0.28, 0.0]
        lm[RING_PIP] = [0.57, 0.44, 0.0]; lm[RING_DIP] = [0.58, 0.38, 0.0]; lm[RING_TIP] = [0.59, 0.32, 0.0]
        lm[PINKY_PIP] = [0.63, 0.48, 0.0]; lm[PINKY_DIP] = [0.65, 0.43, 0.0]; lm[PINKY_TIP] = [0.67, 0.38, 0.0]
        
    elif sign_type == "YES":
        # Thumbs up: fist + thumb pointing up
        lm[THUMB_IP] = [0.36, 0.48, 0.0]
        lm[THUMB_TIP] = [0.35, 0.40, 0.0]
        # Curled fingers
        lm[INDEX_PIP] = [0.44, 0.48, 0.0]; lm[INDEX_DIP] = [0.45, 0.52, 0.0]; lm[INDEX_TIP] = [0.46, 0.54, 0.0]
        lm[MIDDLE_PIP] = [0.50, 0.47, 0.0]; lm[MIDDLE_DIP] = [0.50, 0.51, 0.0]; lm[MIDDLE_TIP] = [0.50, 0.53, 0.0]
        lm[RING_PIP] = [0.56, 0.48, 0.0]; lm[RING_DIP] = [0.55, 0.52, 0.0]; lm[RING_TIP] = [0.54, 0.54, 0.0]
        lm[PINKY_PIP] = [0.60, 0.51, 0.0]; lm[PINKY_DIP] = [0.59, 0.54, 0.0]; lm[PINKY_TIP] = [0.58, 0.56, 0.0]

    elif sign_type == "NO":
        # Pinch / snap: Index & Middle tips touch Thumb tip, Ring & Pinky curled
        lm[THUMB_IP] = [0.42, 0.50, 0.0]; lm[THUMB_TIP] = [0.46, 0.47, 0.0]
        lm[INDEX_PIP] = [0.44, 0.45, 0.0]; lm[INDEX_DIP] = [0.45, 0.46, 0.0]; lm[INDEX_TIP] = [0.46, 0.47, 0.0]
        lm[MIDDLE_PIP] = [0.48, 0.45, 0.0]; lm[MIDDLE_DIP] = [0.47, 0.46, 0.0]; lm[MIDDLE_TIP] = [0.46, 0.47, 0.0]
        lm[RING_PIP] = [0.56, 0.48, 0.0]; lm[RING_DIP] = [0.55, 0.52, 0.0]; lm[RING_TIP] = [0.54, 0.54, 0.0]
        lm[PINKY_PIP] = [0.60, 0.51, 0.0]; lm[PINKY_DIP] = [0.59, 0.54, 0.0]; lm[PINKY_TIP] = [0.58, 0.56, 0.0]

    elif sign_type == "WATER":
        # W shape: Index, Middle, Ring extended upright; Pinky curled; Thumb holding pinky
        lm[THUMB_IP] = [0.46, 0.55, 0.0]; lm[THUMB_TIP] = [0.54, 0.55, 0.0]
        lm[INDEX_PIP] = [0.43, 0.44, 0.0]; lm[INDEX_DIP] = [0.42, 0.38, 0.0]; lm[INDEX_TIP] = [0.41, 0.32, 0.0]
        lm[MIDDLE_PIP] = [0.50, 0.41, 0.0]; lm[MIDDLE_DIP] = [0.50, 0.34, 0.0]; lm[MIDDLE_TIP] = [0.50, 0.28, 0.0]
        lm[RING_PIP] = [0.57, 0.44, 0.0]; lm[RING_DIP] = [0.58, 0.38, 0.0]; lm[RING_TIP] = [0.59, 0.32, 0.0]
        lm[PINKY_PIP] = [0.60, 0.51, 0.0]; lm[PINKY_DIP] = [0.58, 0.54, 0.0]; lm[PINKY_TIP] = [0.56, 0.55, 0.0]

    elif sign_type == "FOOD":
        # Flat O: All 5 fingertips touching in a cone pointing up
        lm[THUMB_IP] = [0.44, 0.48, 0.0]; lm[THUMB_TIP] = [0.48, 0.42, 0.0]
        lm[INDEX_PIP] = [0.44, 0.44, 0.0]; lm[INDEX_DIP] = [0.46, 0.42, 0.0]; lm[INDEX_TIP] = [0.48, 0.42, 0.0]
        lm[MIDDLE_PIP] = [0.49, 0.43, 0.0]; lm[MIDDLE_DIP] = [0.48, 0.42, 0.0]; lm[MIDDLE_TIP] = [0.48, 0.42, 0.0]
        lm[RING_PIP] = [0.53, 0.44, 0.0]; lm[RING_DIP] = [0.50, 0.42, 0.0]; lm[RING_TIP] = [0.48, 0.42, 0.0]
        lm[PINKY_PIP] = [0.57, 0.47, 0.0]; lm[PINKY_DIP] = [0.52, 0.44, 0.0]; lm[PINKY_TIP] = [0.48, 0.42, 0.0]

    elif sign_type == "THANK_YOU":
        # Flat open hand, fingers together
        lm[THUMB_IP] = [0.38, 0.52, 0.0]; lm[THUMB_TIP] = [0.38, 0.46, 0.0]
        lm[INDEX_PIP] = [0.45, 0.44, 0.0]; lm[INDEX_DIP] = [0.45, 0.38, 0.0]; lm[INDEX_TIP] = [0.45, 0.32, 0.0]
        lm[MIDDLE_PIP] = [0.49, 0.43, 0.0]; lm[MIDDLE_DIP] = [0.49, 0.37, 0.0]; lm[MIDDLE_TIP] = [0.49, 0.31, 0.0]
        lm[RING_PIP] = [0.53, 0.44, 0.0]; lm[RING_DIP] = [0.53, 0.38, 0.0]; lm[RING_TIP] = [0.53, 0.32, 0.0]
        lm[PINKY_PIP] = [0.57, 0.46, 0.0]; lm[PINKY_DIP] = [0.57, 0.41, 0.0]; lm[PINKY_TIP] = [0.57, 0.36, 0.0]

    elif sign_type == "DOCTOR":
        # 'D' handshape: Index up, middle/ring/pinky touching thumb
        lm[THUMB_IP] = [0.44, 0.50, 0.0]; lm[THUMB_TIP] = [0.48, 0.48, 0.0]
        lm[INDEX_PIP] = [0.44, 0.44, 0.0]; lm[INDEX_DIP] = [0.43, 0.38, 0.0]; lm[INDEX_TIP] = [0.42, 0.32, 0.0]
        lm[MIDDLE_PIP] = [0.48, 0.45, 0.0]; lm[MIDDLE_DIP] = [0.48, 0.47, 0.0]; lm[MIDDLE_TIP] = [0.48, 0.48, 0.0]
        lm[RING_PIP] = [0.52, 0.46, 0.0]; lm[RING_DIP] = [0.50, 0.48, 0.0]; lm[RING_TIP] = [0.48, 0.48, 0.0]
        lm[PINKY_PIP] = [0.56, 0.49, 0.0]; lm[PINKY_DIP] = [0.52, 0.50, 0.0]; lm[PINKY_TIP] = [0.48, 0.48, 0.0]

    elif sign_type == "HOSPITAL":
        # 'H' handshape: Index + Middle extended together, Ring & Pinky curled
        lm[THUMB_IP] = [0.42, 0.54, 0.0]; lm[THUMB_TIP] = [0.46, 0.54, 0.0]
        lm[INDEX_PIP] = [0.46, 0.44, 0.0]; lm[INDEX_DIP] = [0.46, 0.38, 0.0]; lm[INDEX_TIP] = [0.46, 0.32, 0.0]
        lm[MIDDLE_PIP] = [0.50, 0.44, 0.0]; lm[MIDDLE_DIP] = [0.50, 0.38, 0.0]; lm[MIDDLE_TIP] = [0.50, 0.32, 0.0]
        lm[RING_PIP] = [0.56, 0.48, 0.0]; lm[RING_DIP] = [0.55, 0.52, 0.0]; lm[RING_TIP] = [0.54, 0.54, 0.0]
        lm[PINKY_PIP] = [0.60, 0.51, 0.0]; lm[PINKY_DIP] = [0.59, 0.54, 0.0]; lm[PINKY_TIP] = [0.58, 0.56, 0.0]

    elif sign_type == "EMERGENCY":
        # 'E' handshape: all 4 fingers curled down resting on folded thumb
        lm[THUMB_IP] = [0.44, 0.52, 0.0]; lm[THUMB_TIP] = [0.50, 0.52, 0.0]
        lm[INDEX_PIP] = [0.44, 0.47, 0.0]; lm[INDEX_DIP] = [0.44, 0.50, 0.0]; lm[INDEX_TIP] = [0.45, 0.52, 0.0]
        lm[MIDDLE_PIP] = [0.49, 0.46, 0.0]; lm[MIDDLE_DIP] = [0.49, 0.49, 0.0]; lm[MIDDLE_TIP] = [0.49, 0.52, 0.0]
        lm[RING_PIP] = [0.54, 0.47, 0.0]; lm[RING_DIP] = [0.53, 0.50, 0.0]; lm[RING_TIP] = [0.52, 0.52, 0.0]
        lm[PINKY_PIP] = [0.58, 0.49, 0.0]; lm[PINKY_DIP] = [0.56, 0.52, 0.0]; lm[PINKY_TIP] = [0.55, 0.53, 0.0]

    # Apply scaling, rotation, and Gaussian noise
    center = lm[WRIST]
    rad = math.radians(rot_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    
    for i in range(len(lm)):
        # Scale
        v = (lm[i] - center) * scale
        # Rotate
        rx = v[0] * cos_r - v[1] * sin_r
        ry = v[0] * sin_r + v[1] * cos_r
        lm[i] = center + np.array([rx, ry, v[2]])
        # Noise
        if noise_level > 0:
            lm[i] += np.random.normal(0, noise_level, 3)

    return lm

def extract_features(lm1, lm2=None):
    """Extracts 96 normalized, scale- and translation-invariant features."""
    w1 = lm1[0]
    # Palm scale: distance between wrist and middle MCP
    scale1 = np.linalg.norm(lm1[9] - w1)
    if scale1 < 1e-4:
        scale1 = 1.0
        
    # Relative coordinates normalized by palm scale
    norm1 = (lm1 - w1) / scale1
    coords = norm1.flatten() # 63 features
    
    # Invariant geometric features
    # 5 fingertip-to-wrist distances
    tips = [4, 8, 12, 16, 20]
    d_wrist = [np.linalg.norm(norm1[t] - norm1[0]) for t in tips]
    # 4 fingertip-to-thumb distances
    d_thumb = [np.linalg.norm(norm1[t] - norm1[4]) for t in [8, 12, 16, 20]]
    # 3 inter-finger spreads
    d_spread = [
        np.linalg.norm(norm1[8] - norm1[12]),
        np.linalg.norm(norm1[12] - norm1[16]),
        np.linalg.norm(norm1[16] - norm1[20])
    ]
    # Hand 2 features
    if lm2 is not None:
        w2 = lm2[0]
        scale2 = np.linalg.norm(lm2[9] - w2) or 1.0
        norm2 = (lm2 - w2) / scale2
        rel_hands = (w2 - w1) / scale1
        h2_feats = [np.linalg.norm(rel_hands), 1.0] # distance + has_h2 flag
    else:
        h2_feats = [0.0, 0.0]
        
    # Total feature vector: 63 + 5 + 4 + 3 + 2 = 77 features (padded to 96)
    vec = np.concatenate([coords, d_wrist, d_thumb, d_spread, h2_feats])
    if len(vec) < 96:
        vec = np.pad(vec, (0, 96 - len(vec)), mode='constant')
    return vec[:96]

def run_tests():
    signs = ["HELLO", "YES", "NO", "WATER", "FOOD", "THANK_YOU", "DOCTOR", "HOSPITAL", "EMERGENCY"]
    print(f"=== Running ML Gesture Classifier Validation on {len(signs)} Signs ===")
    
    # 1. Build Prototypes
    prototypes = {}
    for s in signs:
        samples = []
        for _ in range(50):
            scale = np.random.uniform(0.7, 1.4)
            rot = np.random.uniform(-15, 15)
            noise = np.random.uniform(0.005, 0.02)
            hand = create_synthetic_hand(s, noise_level=noise, scale=scale, rot_deg=rot)
            samples.append(extract_features(hand))
        prototypes[s] = np.mean(samples, axis=0)

    # 2. Evaluate Accuracy over 20 test trials per sign under extreme variations
    results = {}
    total_correct = 0
    total_tests = 0
    
    for s in signs:
        correct = 0
        trials = 25
        for _ in range(trials):
            # Test with severe transformations
            scale = np.random.uniform(0.6, 1.6)
            rot = np.random.uniform(-25, 25)
            noise = np.random.uniform(0.01, 0.035)
            test_hand = create_synthetic_hand(s, noise_level=noise, scale=scale, rot_deg=rot)
            test_feat = extract_features(test_hand)
            
            # Cosine similarity classification
            scores = {}
            for name, proto in prototypes.items():
                cos_sim = np.dot(test_feat, proto) / (np.linalg.norm(test_feat) * np.linalg.norm(proto))
                scores[name] = cos_sim
                
            pred = max(scores, key=scores.get)
            if pred == s:
                correct += 1
                
        acc = (correct / trials) * 100.0
        results[s] = {"tests": trials, "correct": correct, "accuracy": acc}
        total_correct += correct
        total_tests += trials
        print(f"SIGN: {s:<12} | TESTS: {trials:2d} | CORRECT: {correct:2d} | ACCURACY: {acc:5.1f}%")

    overall_acc = (total_correct / total_tests) * 100.0
    print(f"\nTOTAL ACCURACY: {overall_acc:.2f}% across {total_tests} evaluations.")
    return prototypes, results

if __name__ == "__main__":
    prototypes, results = run_tests()
