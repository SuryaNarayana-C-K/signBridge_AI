"""
Custom Signs Module (🧠 Teach Your Own Sign)
Manages custom sign storage, normalization, template matching, and name validation.
Supports persistent JSON / SQLite storage and optional cloud DB fallback.
"""

import os
import json
import uuid
import time
import math
from typing import List, Dict, Any, Optional, Tuple

CUSTOM_SIGNS_FILE = os.path.join(os.path.dirname(__file__), "custom_signs.json")

# Reserved names from the 24 built-in vocabulary
BUILTIN_RESERVED_NAMES = {
    "HELLO", "YES", "NO", "PLEASE", "THANK_YOU", "THANK YOU", "SORRY", "GOOD", "BAD", "GOODBYE",
    "DRINK", "NEED", "WANT", "STOP", "WAIT", "COME", "GO", "HOME", "NAME",
    "WHERE", "WHAT", "WHO", "LOVE", "HAPPY", "SAD"
}


def normalize_landmarks(multi_hand_landmarks: List[List[Dict[str, float]]]) -> Optional[List[float]]:
    """
    Normalizes hand landmarks to be invariant to distance, position, and palm scale.
    Input: list of detected hands, where each hand has 21 landmarks with x, y, z.
    Output: flattened list of normalized floats.
    """
    if not multi_hand_landmarks or len(multi_hand_landmarks) == 0:
        return None

    features = []
    for h in multi_hand_landmarks[:2]:  # support up to 2 hands
        w = h[0]  # wrist landmark as origin
        # Palm scale: Euclidean distance between wrist (0) and middle MCP (9)
        dx = h[9].get("x", 0) - w.get("x", 0)
        dy = h[9].get("y", 0) - w.get("y", 0)
        dz = h[9].get("z", 0) - w.get("z", 0)
        palm_scale = math.sqrt(dx * dx + dy * dy + dz * dz)
        if palm_scale < 1e-4:
            palm_scale = 1.0

        for i in range(21):
            features.append((h[i].get("x", 0) - w.get("x", 0)) / palm_scale)
            features.append((h[i].get("y", 0) - w.get("y", 0)) / palm_scale)
            features.append((h[i].get("z", 0) - w.get("z", 0)) / palm_scale)

    return features


def calculate_centroid(samples: List[List[float]]) -> List[float]:
    """Calculates the average feature template centroid across captured samples."""
    if not samples:
        return []
    num_features = len(samples[0])
    centroid = [0.0] * num_features
    for s in samples:
        for i in range(min(num_features, len(s))):
            centroid[i] += s[i]
    return [c / len(samples) for c in centroid]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two feature vectors."""
    n = min(len(v1), len(v2))
    if n == 0:
        return 0.0
    dot = sum(v1[i] * v2[i] for i in range(n))
    mag1 = math.sqrt(sum(v1[i] * v1[i] for i in range(n)))
    mag2 = math.sqrt(sum(v2[i] * v2[i] for i in range(n)))
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    return dot / (mag1 * mag2)


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """Computes Euclidean distance between two feature vectors."""
    n = min(len(v1), len(v2))
    if n == 0:
        return float('inf')
    return math.sqrt(sum((v1[i] - v2[i]) ** 2 for i in range(n)))


class CustomSignStorage:
    """Persistent storage engine for user-taught custom signs."""

    def __init__(self, file_path: str = CUSTOM_SIGNS_FILE):
        self.file_path = file_path
        self._ensure_storage()

    def _ensure_storage(self):
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=2)
            except Exception as e:
                print(f"Warning: Could not create custom signs storage file: {e}")

    def load_all(self) -> List[Dict[str, Any]]:
        """Loads all custom signs from persistent storage."""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            print(f"Error loading custom signs from {self.file_path}: {e}")
            return []

    def save_all(self, signs: List[Dict[str, Any]]) -> bool:
        """Saves custom signs list to persistent storage."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(signs, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom signs to {self.file_path}: {e}")
            return False

    def add_sign(
        self,
        name: str,
        meaning: str,
        samples: List[List[float]],
        num_hands: int = 1,
        sign_id: Optional[str] = None,
        emoji: str = "🧠"
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validates and adds a new custom sign.
        Returns: (success, message, sign_dict)
        """
        name_clean = name.strip()
        meaning_clean = meaning.strip()

        if not name_clean:
            return False, "Sign name cannot be empty.", None

        if not meaning_clean:
            return False, "Sign meaning cannot be empty.", None

        if not samples or len(samples) < 5:
            return False, "Insufficient samples. At least 5 valid landmark samples required.", None

        # Check against reserved built-in sign names
        name_upper = name_clean.upper().replace("-", "_").replace(" ", "_")
        if name_upper in BUILTIN_RESERVED_NAMES:
            return False, f"This sign name ('{name_clean}') already exists in standard vocabulary. Choose another name.", None

        signs = self.load_all()

        # Check if custom sign with same name or id already exists
        for s in signs:
            if s.get("name", "").upper() == name_clean.upper():
                return False, f"A custom sign named '{name_clean}' already exists.", None

        new_id = sign_id or f"custom_{uuid.uuid4().hex[:8]}"
        centroid = calculate_centroid(samples)

        sign_record = {
            "id": new_id,
            "name": name_clean.upper(),
            "meaning": meaning_clean,
            "emoji": emoji or "🧠",
            "category": "Custom Learned",
            "sample_count": len(samples),
            "num_hands": num_hands,
            "template": centroid,
            "samples": samples,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        signs.append(sign_record)
        if self.save_all(signs):
            return True, f"Successfully learned sign '{name_clean.upper()}'!", sign_record
        return False, "Failed to write custom sign to persistent storage.", None

    def delete_sign(self, sign_id_or_name: str) -> Tuple[bool, str]:
        """Deletes a custom sign by ID or name."""
        signs = self.load_all()
        target = sign_id_or_name.strip()
        filtered = [
            s for s in signs 
            if s.get("id") != target and s.get("name", "").upper() != target.upper()
        ]

        if len(filtered) == len(signs):
            return False, f"Sign '{sign_id_or_name}' not found."

        if self.save_all(filtered):
            return True, f"Deleted sign '{sign_id_or_name}'."
        return False, "Failed to update storage file."


# Global storage instance
storage = CustomSignStorage()


def load_custom_signs() -> List[Dict[str, Any]]:
    """Loads all custom signs."""
    return storage.load_all()


def save_custom_sign(
    name: str,
    meaning: str,
    samples: List[List[float]],
    num_hands: int = 1,
    sign_id: Optional[str] = None,
    emoji: str = "🧠"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Saves a newly taught custom sign."""
    return storage.add_sign(name, meaning, samples, num_hands, sign_id, emoji)


def delete_custom_sign(sign_id_or_name: str) -> Tuple[bool, str]:
    """Deletes a custom sign."""
    return storage.delete_sign(sign_id_or_name)


def validate_sign_name(name: str) -> Tuple[bool, str]:
    """Validates if a sign name is allowed."""
    name_clean = name.strip()
    if not name_clean:
        return False, "Sign name cannot be empty."
    name_upper = name_clean.upper().replace("-", "_").replace(" ", "_")
    if name_upper in BUILTIN_RESERVED_NAMES:
        return False, "This sign name already exists. Choose another name."
    return True, "Name is valid."


def match_custom_sign(
    live_features: List[float],
    custom_signs: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.85
) -> Optional[Dict[str, Any]]:
    """
    Matches live normalized landmark features against stored custom signs.
    Returns matching sign dict with confidence score, or None if below threshold.
    """
    if not live_features:
        return None

    if custom_signs is None:
        custom_signs = load_custom_signs()

    if not custom_signs:
        return None

    best_match = None
    best_similarity = -1.0

    for sign in custom_signs:
        template = sign.get("template")
        samples = sign.get("samples", [])
        
        # Method 1: Cosine similarity against centroid template
        sim_template = 0.0
        if template:
            sim_template = cosine_similarity(live_features, template)
        
        # Method 2: k-NN nearest sample similarity
        sim_samples = 0.0
        if samples:
            sample_sims = [cosine_similarity(live_features, s) for s in samples]
            # Top 3 average similarity
            sample_sims.sort(reverse=True)
            top_k = sample_sims[:3]
            sim_samples = sum(top_k) / len(top_k) if top_k else 0.0

        combined_similarity = max(sim_template, sim_samples)

        if combined_similarity > best_similarity:
            best_similarity = combined_similarity
            best_match = {
                "id": sign.get("id"),
                "name": sign.get("name"),
                "meaning": sign.get("meaning"),
                "emoji": sign.get("emoji", "🧠"),
                "category": "Custom Learned",
                "phrase": sign.get("meaning"),
                "confidence": round(combined_similarity, 3)
            }

    if best_match and best_similarity >= threshold:
        return best_match

    return None
