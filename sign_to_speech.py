"""
Sign-to-Speech (🤟 Sign → Speech) Module
24 Everyday Communication Signs with MediaPipe Hands tracking,
Hybrid Geometric + Calibrated ML Engine, Category Filters, and Browser Text-to-Speech.
"""

import json
import streamlit as st
import streamlit.components.v1 as components

from custom_signs import (
    load_custom_signs,
    save_custom_sign,
    delete_custom_sign,
    validate_sign_name,
    BUILTIN_RESERVED_NAMES
)

# 24 Everyday Communication Signs Vocabulary
VOCABULARY_24_SIGNS = [
    # --- EVERYDAY (9 signs) ---
    {
        "id": "HELLO",
        "name": "HELLO",
        "emoji": "👋",
        "phrase": "Hello!",
        "desc": "Open palm facing camera with all 5 fingers upright and spread.",
        "category": "Everyday",
        "color": "#38bdf8",
    },
    {
        "id": "YES",
        "name": "YES",
        "emoji": "👍",
        "phrase": "Yes.",
        "desc": "Thumbs-up gesture with 4 fingers curled into a fist.",
        "category": "Everyday",
        "color": "#10b981",
    },
    {
        "id": "NO",
        "name": "NO",
        "emoji": "👎",
        "phrase": "No.",
        "desc": "Thumbs-down OR Index + Middle pinching against Thumb.",
        "category": "Everyday",
        "color": "#ef4444",
    },
    {
        "id": "PLEASE",
        "name": "PLEASE",
        "emoji": "🙏",
        "phrase": "Please.",
        "desc": "Flat open hand pressed over chest or prayer hands posture.",
        "category": "Everyday",
        "color": "#a855f7",
    },
    {
        "id": "THANK_YOU",
        "name": "THANK YOU",
        "emoji": "🤝",
        "phrase": "Thank you.",
        "desc": "Flat open hand moving outward from chin towards camera.",
        "category": "Everyday",
        "color": "#8b5cf6",
    },
    {
        "id": "SORRY",
        "name": "SORRY",
        "emoji": "🙇",
        "phrase": "Sorry.",
        "desc": "Closed fist held over heart / chest area in circular motion.",
        "category": "Everyday",
        "color": "#ec4899",
    },
    {
        "id": "GOOD",
        "name": "GOOD",
        "emoji": "👌",
        "phrase": "Good.",
        "desc": "'OK' sign: thumb and index forming circle with other 3 fingers up.",
        "category": "Everyday",
        "color": "#10b981",
    },
    {
        "id": "BAD",
        "name": "BAD",
        "emoji": "👎",
        "phrase": "Bad.",
        "desc": "Flat hand turned downward and pushed downward.",
        "category": "Everyday",
        "color": "#f43f5e",
    },
    {
        "id": "GOODBYE",
        "name": "GOODBYE",
        "emoji": "👋",
        "phrase": "Goodbye!",
        "desc": "Open hand waving side-to-side.",
        "category": "Everyday",
        "color": "#0ea5e9",
    },

    # --- NEEDS (3 signs) ---
    {
        "id": "DRINK",
        "name": "DRINK",
        "emoji": "🥤",
        "phrase": "I want a drink.",
        "desc": "'C' handshape bringing thumb to mouth like holding a cup.",
        "category": "Needs",
        "color": "#0284c7",
    },
    {
        "id": "NEED",
        "name": "NEED",
        "emoji": "🤲",
        "phrase": "I need this.",
        "desc": "Bent hooked index finger pointing and tapping downward.",
        "category": "Needs",
        "color": "#d97706",
    },
    {
        "id": "WANT",
        "name": "WANT",
        "emoji": "🫳",
        "phrase": "I want this.",
        "desc": "Clawed curved fingers pulling inward toward chest.",
        "category": "Needs",
        "color": "#fb923c",
    },

    # --- ACTIONS (4 signs) ---
    {
        "id": "STOP",
        "name": "STOP",
        "emoji": "🛑",
        "phrase": "Stop.",
        "desc": "Vertical flat palm facing camera firmly (traffic stop).",
        "category": "Actions",
        "color": "#dc2626",
    },
    {
        "id": "WAIT",
        "name": "WAIT",
        "emoji": "✋",
        "phrase": "Please wait.",
        "desc": "Open hand held steady with relaxed spread fingers.",
        "category": "Actions",
        "color": "#eab308",
    },
    {
        "id": "COME",
        "name": "COME",
        "emoji": "🫴",
        "phrase": "Come here.",
        "desc": "Index finger or open hand gesturing inward towards body.",
        "category": "Actions",
        "color": "#6366f1",
    },
    {
        "id": "GO",
        "name": "GO",
        "emoji": "👉",
        "phrase": "Go.",
        "desc": "Index finger pointing and gesturing outward/forward.",
        "category": "Actions",
        "color": "#8b5cf6",
    },

    # --- PEOPLE & PLACES (2 signs) ---
    {
        "id": "HOME",
        "name": "HOME",
        "emoji": "🏠",
        "phrase": "I want to go home.",
        "desc": "Pinched fingertips touching cheek or hands meeting in roof shape.",
        "category": "People & Places",
        "color": "#14b8a6",
    },
    {
        "id": "NAME",
        "name": "NAME",
        "emoji": "🏷️",
        "phrase": "My name.",
        "desc": "'H' shapes (Index + Middle) crossed and tapping over each other.",
        "category": "People & Places",
        "color": "#a855f7",
    },

    # --- QUESTIONS (3 signs) ---
    {
        "id": "WHERE",
        "name": "WHERE",
        "emoji": "🤷",
        "phrase": "Where?",
        "desc": "Open palms facing upward with gentle shrugging motion.",
        "category": "Questions",
        "color": "#f59e0b",
    },
    {
        "id": "WHAT",
        "name": "WHAT",
        "emoji": "❓",
        "phrase": "What?",
        "desc": "Open hand tilted sideways with index pointing up slightly.",
        "category": "Questions",
        "color": "#38bdf8",
    },
    {
        "id": "WHO",
        "name": "WHO",
        "emoji": "🔍",
        "phrase": "Who?",
        "desc": "Index finger circling around mouth / chin.",
        "category": "Questions",
        "color": "#10b981",
    },

    # --- EMOTIONS (3 signs) ---
    {
        "id": "LOVE",
        "name": "LOVE",
        "emoji": "🤟",
        "phrase": "Love.",
        "desc": "'I Love You' sign: Thumb, Index, Pinky extended; Middle, Ring curled.",
        "category": "Emotions",
        "color": "#f43f5e",
    },
    {
        "id": "HAPPY",
        "name": "HAPPY",
        "emoji": "😊",
        "phrase": "I am happy.",
        "desc": "Open flat palm patting/stroking upward across chest.",
        "category": "Emotions",
        "color": "#10b981",
    },
    {
        "id": "SAD",
        "name": "SAD",
        "emoji": "😢",
        "phrase": "I am sad.",
        "desc": "Open hand moving downward along face with downward orientation.",
        "category": "Emotions",
        "color": "#3b82f6",
    },
]


def render_sign_to_speech_component():
    """Renders the self-contained HTML5 + WebRTC + MediaPipe + Web Speech API component with Custom Sign Learning."""
    
    custom_signs_list = load_custom_signs()
    custom_signs_json = json.dumps(custom_signs_list)
    reserved_names_json = json.dumps(list(BUILTIN_RESERVED_NAMES))
    vocab_json = json.dumps(VOCABULARY_24_SIGNS)
    
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sign to Speech</title>
        <!-- Google Fonts -->
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
        
        <!-- MediaPipe CDN -->
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
        
        <style>
            :root {{
                --bg-main: #080D16;
                --bg-card: #111A28;
                --bg-panel: #162235;
                --bg-info: #172F4A;
                --border-info: #214B73;
                --border-color: #23324A;
                --border-secondary: #23324A;
                --accent-primary: #2563EB;
                --accent-primary-hover: #1D4ED8;
                --accent-blue: #3B82F6;
                --accent-cyan: #22D3EE;
                --accent-emerald: #22C55E;
                --accent-amber: #F59E0B;
                --accent-rose: #EF4444;
                --accent-purple: #8B5CF6;
                --text-primary: #F8FAFC;
                --text-secondary: #94A3B8;
                --text-muted: #64748B;
                --font-sans: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                --font-mono: 'JetBrains Mono', monospace;
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                user-select: none;
            }}

            body {{
                font-family: var(--font-sans);
                background-color: transparent;
                color: var(--text-primary);
                padding: 4px;
                overflow-x: hidden;
            }}

            .main-container {{
                display: grid;
                grid-template-columns: 1.15fr 1fr;
                gap: 20px;
                width: 100%;
                max-width: 1300px;
                margin: 0 auto;
            }}

            @media (max-width: 900px) {{
                .main-container {{
                    grid-template-columns: 1fr;
                }}
            }}

            /* Video / Camera Panel */
            .camera-card {{
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                gap: 14px;
                position: relative;
            }}

            .camera-card.active-stream {{
                border-color: var(--accent-primary);
            }}

            .card-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}

            .card-title {{
                font-size: 1.10rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--text-primary);
                letter-spacing: -0.01em;
            }}

            .status-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 5px 12px;
                border-radius: 999px;
                font-size: 0.80rem;
                font-weight: 600;
                background: var(--bg-panel);
                border: 1px solid var(--border-color);
                color: var(--text-secondary);
                transition: all 0.2s ease;
            }}

            .status-badge.ready {{
                background: rgba(34, 197, 94, 0.12);
                border-color: rgba(34, 197, 94, 0.35);
                color: var(--accent-emerald);
            }}

            .status-badge.detecting {{
                background: var(--bg-info);
                border-color: var(--border-info);
                color: var(--accent-blue);
            }}

            .status-badge.detected {{
                background: rgba(37, 99, 235, 0.20);
                border-color: var(--accent-primary);
                color: #FFFFFF;
            }}

            .status-badge.custom-detected {{
                background: rgba(139, 92, 246, 0.25);
                border-color: var(--accent-purple);
                color: #DDD6FE;
                box-shadow: 0 0 12px rgba(139, 92, 246, 0.3);
            }}

            .status-badge.warning {{
                background: rgba(245, 158, 11, 0.12);
                border-color: rgba(245, 158, 11, 0.35);
                color: var(--accent-amber);
            }}

            .status-badge.error {{
                background: rgba(239, 68, 68, 0.15);
                border-color: rgba(239, 68, 68, 0.35);
                color: #F87171;
            }}

            .status-dot {{
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: currentColor;
                display: inline-block;
            }}

            .video-viewport {{
                position: relative;
                width: 100%;
                aspect-ratio: 4 / 3;
                background: #080C14;
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid var(--border-color);
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            video#webcam {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scaleX(-1);
                display: block;
            }}

            canvas#output_canvas {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                transform: scaleX(-1);
                pointer-events: none;
            }}

            .hud-overlay {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                right: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                pointer-events: none;
            }}

            .hud-tag {{
                background: rgba(11, 15, 23, 0.85);
                backdrop-filter: blur(6px);
                border: 1px solid var(--border-color);
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-family: var(--font-mono);
                color: var(--text-secondary);
            }}

            .camera-controls {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}

            .btn {{
                background: var(--bg-card);
                border: 1px solid var(--border-secondary);
                color: var(--text-primary);
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 0.88rem;
                font-weight: 600;
                font-family: var(--font-sans);
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: all 0.2s ease;
            }}

            .btn:hover {{
                background: var(--bg-panel);
                border-color: var(--accent-primary);
            }}

            .btn:active {{
                transform: translateY(0);
            }}

            .btn-primary {{
                background: var(--accent-primary);
                border: 1px solid var(--accent-primary);
                color: #FFFFFF;
                box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
            }}

            .btn-primary:hover {{
                background: var(--accent-primary-hover);
                border-color: var(--accent-primary-hover);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
            }}

            .btn-purple {{
                background: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
                border: 1px solid #8B5CF6;
                color: #FFFFFF;
                box-shadow: 0 2px 8px rgba(124, 58, 237, 0.30);
            }}

            .btn-purple:hover {{
                background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
                box-shadow: 0 4px 14px rgba(139, 92, 246, 0.45);
            }}

            .btn-danger {{
                background: #3A1820;
                border: 1px solid #7F1D1D;
                color: #F87171;
            }}

            .btn-danger:hover {{
                background: #4C1D24;
                border-color: #991B1B;
                color: #FFFFFF;
            }}

            /* Output / Speech Recognition Panel */
            .output-card {{
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 18px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }}

            .detected-display {{
                background: var(--bg-info);
                border: 1px solid var(--border-info);
                border-radius: 12px;
                padding: 18px 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                transition: all 0.20s ease;
                position: relative;
                overflow: hidden;
            }}

            .detected-display.active-sign {{
                background: var(--bg-card);
                border-color: var(--accent-primary);
                box-shadow: 0 0 16px rgba(37, 99, 235, 0.15);
            }}

            .detected-display.active-custom-sign {{
                background: #15182D;
                border-color: var(--accent-purple);
                box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
            }}

            .sign-header-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}

            .sign-category {{
                font-size: 0.72rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--accent-blue);
                background: rgba(37, 99, 235, 0.18);
                padding: 3px 8px;
                border-radius: 6px;
            }}

            .sign-category.custom-tag {{
                color: #C4B5FD;
                background: rgba(139, 92, 246, 0.25);
                border: 1px solid rgba(139, 92, 246, 0.4);
            }}

            .confidence-meter {{
                display: flex;
                align-items: center;
                gap: 8px;
                font-family: var(--font-mono);
                font-size: 0.82rem;
                color: var(--text-secondary);
            }}

            .confidence-bar-bg {{
                width: 80px;
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 999px;
                overflow: hidden;
            }}

            .confidence-bar-fill {{
                height: 100%;
                width: 0%;
                background: var(--accent-primary);
                border-radius: 999px;
                transition: width 0.15s ease;
            }}

            .sign-big-visual {{
                display: flex;
                align-items: center;
                gap: 16px;
            }}

            .sign-big-emoji-wrap {{
                width: 64px;
                height: 64px;
                background: var(--bg-panel);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2.5rem;
                flex-shrink: 0;
            }}

            .sign-big-details {{
                display: flex;
                flex-direction: column;
                gap: 2px;
            }}

            .sign-name-text {{
                font-size: 1.65rem;
                font-weight: 800;
                letter-spacing: -0.02em;
                color: var(--text-primary);
            }}

            .sign-phrase-text {{
                font-size: 1.0rem;
                font-weight: 500;
                color: var(--accent-blue);
            }}

            .action-bar {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
                padding-top: 4px;
            }}

            .sentence-box {{
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 14px;
                display: flex;
                flex-direction: column;
                gap: 8px;
            }}

            .sentence-title {{
                font-size: 0.78rem;
                font-weight: 600;
                color: var(--text-secondary);
                text-transform: uppercase;
                letter-spacing: 0.05em;
                display: flex;
                justify-content: space-between;
            }}

            .sentence-content {{
                min-height: 48px;
                max-height: 90px;
                overflow-y: auto;
                font-size: 1.05rem;
                font-weight: 500;
                color: var(--text-primary);
                line-height: 1.4;
                padding-right: 4px;
            }}

            .tuning-card {{
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 12px 14px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}

            .tuning-row {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                font-size: 0.82rem;
                color: var(--text-secondary);
            }}

            .tuning-input {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            input[type="range"] {{
                accent-color: var(--accent-primary);
                cursor: pointer;
            }}

            /* Debug Diagnostic Card */
            .debug-card {{
                grid-column: 1 / -1;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 14px 18px;
                font-family: var(--font-mono);
                font-size: 0.80rem;
                color: var(--text-secondary);
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 10px;
            }}

            .debug-item span {{
                color: var(--accent-blue);
                font-weight: 700;
            }}

            /* ===================================================
               🧠 TEACH YOUR OWN SIGN SECTION
               =================================================== */
            .teach-card {{
                grid-column: 1 / -1;
                background: linear-gradient(135deg, #10192A 0%, #151D33 100%);
                border: 1px solid #2B3D5B;
                border-radius: 16px;
                padding: 20px 24px;
                display: flex;
                flex-direction: column;
                gap: 18px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            }}

            .teach-card.highlight-recording {{
                border-color: #8B5CF6;
                box-shadow: 0 0 24px rgba(139, 92, 246, 0.25);
            }}

            .teach-header-btn {{
                background: transparent;
                border: none;
                color: var(--text-primary);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: space-between;
                width: 100%;
                font-size: 1.15rem;
                font-weight: 700;
            }}

            .teach-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 18px;
            }}

            @media (max-width: 850px) {{
                .teach-grid {{
                    grid-template-columns: 1fr;
                }}
            }}

            .teach-input-group {{
                display: flex;
                flex-direction: column;
                gap: 6px;
            }}

            .teach-label {{
                font-size: 0.84rem;
                font-weight: 600;
                color: #CBD5E1;
            }}

            .teach-input {{
                background: #0B121E;
                border: 1px solid #28374E;
                color: #F8FAFC;
                padding: 10px 14px;
                border-radius: 8px;
                font-size: 0.92rem;
                font-family: var(--font-sans);
                outline: none;
                transition: border-color 0.2s ease;
            }}

            .teach-input:focus {{
                border-color: var(--accent-blue);
                box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
            }}

            .quality-box {{
                background: #0B121E;
                border: 1px solid #23324A;
                border-radius: 10px;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            }}

            .quality-badge {{
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 0.80rem;
                font-weight: 700;
            }}

            .quality-good {{
                background: rgba(34, 197, 94, 0.15);
                border: 1px solid rgba(34, 197, 94, 0.4);
                color: #4ADE80;
            }}

            .quality-warning {{
                background: rgba(245, 158, 11, 0.15);
                border: 1px solid rgba(245, 158, 11, 0.4);
                color: #FBBF24;
            }}

            .quality-bad {{
                background: rgba(239, 68, 68, 0.15);
                border: 1px solid rgba(239, 68, 68, 0.4);
                color: #F87171;
            }}

            .learning-summary-box {{
                background: rgba(139, 92, 246, 0.08);
                border: 1px solid rgba(139, 92, 246, 0.35);
                border-radius: 12px;
                padding: 16px 20px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }}

            .summary-item-row {{
                display: flex;
                justify-content: space-between;
                font-size: 0.88rem;
                color: #94A3B8;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                padding-bottom: 6px;
            }}

            .summary-item-row b {{
                color: #F8FAFC;
            }}

            /* ===================================================
               📚 MY LEARNED SIGNS SECTION
               =================================================== */
            .my-signs-card {{
                grid-column: 1 / -1;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 14px;
            }}

            .custom-signs-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
                gap: 12px;
            }}

            .custom-sign-item {{
                background: #131E30;
                border: 1px solid #23354F;
                border-radius: 12px;
                padding: 14px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                transition: all 0.2s ease;
            }}

            .custom-sign-item:hover {{
                border-color: #8B5CF6;
                transform: translateY(-2px);
                box-shadow: 0 4px 14px rgba(139, 92, 246, 0.15);
            }}

            .custom-sign-top {{
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}

            .custom-sign-name {{
                font-weight: 800;
                font-size: 0.98rem;
                color: #F8FAFC;
            }}

            .custom-sign-meaning {{
                font-size: 0.85rem;
                color: #A78BFA;
                font-weight: 600;
            }}

            .custom-sign-meta {{
                font-size: 0.74rem;
                color: #64748B;
                display: flex;
                justify-content: space-between;
            }}

            .custom-sign-actions {{
                display: flex;
                gap: 6px;
                margin-top: 4px;
            }}

            /* Calibration Studio Card */
            .calibration-card {{
                grid-column: 1 / -1;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 18px 22px;
                display: flex;
                flex-direction: column;
                gap: 14px;
            }}

            .calibration-grid {{
                display: grid;
                grid-template-columns: 1fr 1.2fr 1fr;
                gap: 14px;
                align-items: center;
            }}

            @media (max-width: 900px) {{
                .calibration-grid {{
                    grid-template-columns: 1fr;
                }}
            }}

            .sample-progress-bar {{
                width: 100%;
                height: 8px;
                background: var(--bg-panel);
                border-radius: 999px;
                overflow: hidden;
                margin-top: 6px;
            }}

            .sample-progress-fill {{
                height: 100%;
                width: 0%;
                background: var(--accent-primary);
                border-radius: 999px;
                transition: width 0.1s ease;
            }}

            .sample-progress-fill.purple-fill {{
                background: linear-gradient(90deg, #8B5CF6, #C084FC);
            }}

            /* Vocabulary Section & Category Filter Tabs */
            .vocab-section {{
                grid-column: 1 / -1;
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 14px;
                padding: 20px;
            }}

            .category-tabs {{
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                margin-top: 12px;
            }}

            .cat-tab {{
                background: var(--bg-panel);
                border: 1px solid var(--border-color);
                color: var(--text-secondary);
                padding: 6px 14px;
                border-radius: 8px;
                font-size: 0.82rem;
                font-weight: 600;
                font-family: var(--font-sans);
                cursor: pointer;
                transition: all 0.2s ease;
            }}

            .cat-tab:hover, .cat-tab.active {{
                background: var(--accent-primary);
                color: #FFFFFF;
                border-color: var(--accent-primary);
            }}

            .vocab-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
                gap: 12px;
                margin-top: 14px;
            }}

            .vocab-card {{
                background: var(--bg-card);
                border: 1px solid var(--border-color);
                border-radius: 12px;
                padding: 14px;
                display: flex;
                flex-direction: column;
                gap: 6px;
                cursor: pointer;
                transition: all 0.2s ease;
            }}

            .vocab-card:hover {{
                background: #1A2940;
                border-color: var(--accent-primary);
                transform: translateY(-1px);
            }}

            .vocab-card.highlight {{
                border-color: var(--accent-primary);
                background: #1A2940;
            }}

            .vocab-card.highlight-custom {{
                border-color: var(--accent-purple) !important;
                background: #1F1938 !important;
            }}

            .vocab-top {{
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}

            .vocab-icon-box {{
                width: 36px;
                height: 36px;
                background: var(--bg-panel);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.3rem;
            }}

            .vocab-name {{
                font-weight: 700;
                font-size: 0.92rem;
                color: var(--text-primary);
            }}

            .vocab-desc {{
                font-size: 0.75rem;
                color: var(--text-secondary);
                line-height: 1.35;
            }}

            .switch {{
                position: relative;
                display: inline-block;
                width: 36px;
                height: 20px;
            }}

            .switch input {{
                opacity: 0;
                width: 0;
                height: 0;
            }}

            .slider {{
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #334155;
                transition: .2s;
                border-radius: 20px;
            }}

            .slider:before {{
                position: absolute;
                content: "";
                height: 14px;
                width: 14px;
                left: 3px;
                bottom: 3px;
                background-color: white;
                transition: .2s;
                border-radius: 50%;
            }}

            input:checked + .slider {{
                background-color: var(--accent-primary);
            }}

            input:checked + .slider:before {{
                transform: translateX(16px);
            }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <!-- LEFT COLUMN: WEBCAM VIEWPORT -->
            <div class="camera-card">
                <div class="card-header">
                    <div class="card-title">
                        <span>📹 Live Sign Camera</span>
                    </div>
                    <div id="statusBadge" class="status-badge">
                        <span class="status-dot"></span>
                        <span id="statusText">Initializing Camera...</span>
                    </div>
                </div>

                <div class="video-viewport">
                    <video id="webcam" playsinline autoplay muted></video>
                    <canvas id="output_canvas"></canvas>
                    
                    <div class="hud-overlay">
                        <div class="hud-tag" id="fpsTag">FPS: --</div>
                        <div class="hud-tag" id="handsTag">Hands: 0</div>
                    </div>
                </div>

                <div class="camera-controls">
                    <button id="toggleCameraBtn" class="btn btn-primary" onclick="toggleCamera()">
                        <span>⏹️ Pause Camera</span>
                    </button>
                    <button id="restartCameraBtn" class="btn" onclick="restartCamera()">
                        <span>🔄 Reconnect</span>
                    </button>
                    <div style="margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #94a3b8;">
                        <span>Show Skeleton</span>
                        <label class="switch">
                            <input type="checkbox" id="skeletonToggle" checked>
                            <span class="slider"></span>
                        </label>
                    </div>
                </div>
            </div>

            <!-- RIGHT COLUMN: DETECTED SIGN & SPEECH -->
            <div class="output-card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🤟 Recognition & Voice</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #94a3b8;">
                        <span>Auto-Speak</span>
                        <label class="switch">
                            <input type="checkbox" id="autoSpeakToggle" checked>
                            <span class="slider"></span>
                        </label>
                    </div>
                </div>

                <!-- Detected Sign Card -->
                <div id="detectedDisplay" class="detected-display">
                    <div class="sign-header-row">
                        <div id="signCategory" class="sign-category">Ready</div>
                        <div class="confidence-meter">
                            <span>Confidence:</span>
                            <div class="confidence-bar-bg">
                                <div id="confidenceBar" class="confidence-bar-fill"></div>
                            </div>
                            <span id="confidenceText" style="font-weight: 700; width: 38px;">0%</span>
                        </div>
                    </div>

                    <div class="sign-big-visual">
                        <div id="signEmoji" class="sign-big-emoji-wrap">✋</div>
                        <div class="sign-big-details">
                            <div id="signName" class="sign-name-text">Waiting for sign...</div>
                            <div id="signPhrase" class="sign-phrase-text">Show your hand clearly to the camera</div>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="action-bar">
                    <button id="speakBtn" class="btn btn-primary" style="flex: 1; justify-content: center; font-size: 1rem; padding: 12px;" onclick="speakCurrentSign()">
                        <span>🔊 Speak Now</span>
                    </button>
                    <button id="clearBtn" class="btn btn-danger" onclick="clearSentence()">
                        <span>🗑️ Clear</span>
                    </button>
                </div>

                <!-- Sentence History Stream -->
                <div class="sentence-box">
                    <div class="sentence-title">
                        <span>Spoken Sentence Stream</span>
                        <span id="wordCount">0 phrases</span>
                    </div>
                    <div id="sentenceContent" class="sentence-content">
                        <span style="color: #64748b; font-style: italic;">Detected spoken phrases will accumulate here...</span>
                    </div>
                </div>

                <!-- Settings / Threshold Accordion -->
                <div class="tuning-card">
                    <div class="tuning-row">
                        <span>Confidence Threshold: <b id="threshVal" style="color: var(--accent-cyan)">75%</b></span>
                        <div class="tuning-input">
                            <input type="range" id="confSlider" min="55" max="95" value="75" oninput="updateSettings()">
                        </div>
                    </div>
                    <div class="tuning-row">
                        <span>Speech Voice:</span>
                        <select id="voiceSelect" style="background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 3px 8px; font-size: 0.78rem; max-width: 170px;">
                            <option value="">Default Voice</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- LIVE DIAGNOSTICS HUD -->
            <div class="debug-card" id="debugHud">
                <div class="debug-item">MediaPipe Status: <span id="dbgMpStatus">INITIALIZING</span></div>
                <div class="debug-item">Camera: <span id="dbgCamera">CONNECTING</span></div>
                <div class="debug-item">Processing Loop: <span id="dbgLoop">RUNNING</span></div>
                <div class="debug-item">Frames Processed: <span id="dbgFrames">0</span></div>
                <div class="debug-item">Hands Detected: <span id="dbgHands">0</span></div>
                <div class="debug-item">Landmarks: <span id="dbgLandmarks">NO</span></div>
                <div class="debug-item">Detected Sign: <span id="dbgRawPred">--</span></div>
            </div>

            <!-- ===================================================
                 🧠 1. TEACH YOUR OWN SIGN CARD (ADDITIVE FEATURE)
                 =================================================== -->
            <div class="teach-card" id="teachCard">
                <div class="card-header">
                    <div class="card-title">
                        <span>🧠 Teach Your Own Sign</span>
                        <span style="background: rgba(139, 92, 246, 0.22); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.4); font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; font-weight: 700;">NEW FEATURE</span>
                    </div>
                    <span style="font-size: 0.82rem; color: #94A3B8;">Teach the AI any new gesture and its spoken meaning</span>
                </div>

                <div class="teach-grid">
                    <!-- Left: Form Inputs -->
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <div class="teach-input-group">
                            <label class="teach-label" for="customMeaningInput">1. What does this sign mean? *</label>
                            <input type="text" id="customMeaningInput" class="teach-input" placeholder="e.g. Good Morning, I Need Help, Thank You" oninput="onCustomMeaningChanged()">
                        </div>

                        <div class="teach-input-group">
                            <label class="teach-label" for="customNameInput">2. Sign Name (Short ID) *</label>
                            <input type="text" id="customNameInput" class="teach-input" placeholder="e.g. good_morning">
                        </div>

                        <!-- Live Hand Quality Indicator -->
                        <div class="quality-box">
                            <div>
                                <div style="font-size: 0.75rem; color: #64748B; margin-bottom: 3px;">Hand Position Quality</div>
                                <div id="qualityBadge" class="quality-badge quality-bad">
                                    <span>🔴 No hand detected</span>
                                </div>
                            </div>
                            <span id="qualityAdvice" style="font-size: 0.76rem; color: #94A3B8; max-width: 200px; text-align: right;">Show your hand clearly to the webcam</span>
                        </div>
                    </div>

                    <!-- Right: Capture Controls & Summary -->
                    <div style="display: flex; flex-direction: column; gap: 12px;">
                        <div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; color: #CBD5E1;">
                                <span id="teachStatusText">3. Capture Samples (Hold Gesture):</span>
                                <b id="customSampleCountText" style="color: #A78BFA;">0 / 25 samples</b>
                            </div>
                            <div class="sample-progress-bar">
                                <div id="customProgressFill" class="sample-progress-fill purple-fill"></div>
                            </div>
                        </div>

                        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                            <button id="startTeachBtn" class="btn btn-purple" style="flex: 1; justify-content: center; font-size: 0.90rem; padding: 10px;" onclick="startCustomSignLearning()">
                                <span>🚀 Start Learning (25 Samples)</span>
                            </button>
                            <button id="cancelTeachBtn" class="btn" style="display: none; font-size: 0.90rem;" onclick="cancelCustomSignLearning()">
                                <span>⏹️ Cancel</span>
                            </button>
                        </div>

                        <div style="font-size: 0.76rem; color: #64748B; line-height: 1.35;">
                            💡 <b>Pro-Tip:</b> Hold your hand steady while recording. Move slightly in 3D space to capture natural variations for maximum accuracy.
                        </div>

                        <!-- Learning Summary Box (Revealed after 25 samples) -->
                        <div id="learningSummaryBox" class="learning-summary-box" style="display: none;">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <span style="font-size: 0.85rem; font-weight: 700; color: #DDD6FE; letter-spacing: 0.04em;">📋 SIGN LEARNED SUMMARY</span>
                                <span style="background: #059669; color: #FFFFFF; font-size: 0.70rem; padding: 2px 8px; border-radius: 999px; font-weight: 700;">Ready to Save</span>
                            </div>
                            <div class="summary-item-row">
                                <span>Name:</span>
                                <b id="summarySignName">--</b>
                            </div>
                            <div class="summary-item-row">
                                <span>Meaning:</span>
                                <b id="summarySignMeaning">--</b>
                            </div>
                            <div class="summary-item-row">
                                <span>Samples:</span>
                                <b id="summarySampleCount">25 samples</b>
                            </div>
                            <div class="summary-item-row">
                                <span>Recognition:</span>
                                <b style="color: #34D399;">Active & Calibrated</b>
                            </div>

                            <div id="conflictErrorMsg" style="color: #F87171; font-size: 0.80rem; display: none; font-weight: 600;"></div>

                            <div style="display: flex; gap: 8px; margin-top: 4px;">
                                <button id="saveSignBtn" class="btn btn-primary" style="flex: 1; justify-content: center; background: linear-gradient(135deg, #059669, #10B981); border-color: #10B981;" onclick="saveCustomSign()">
                                    <span>💾 Save Sign</span>
                                </button>
                                <button class="btn" onclick="startCustomSignLearning()">
                                    <span>🔄 Re-record</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- ===================================================
                 📚 2. MY LEARNED SIGNS REPOSITORY
                 =================================================== -->
            <div class="my-signs-card">
                <div class="card-header">
                    <div class="card-title">
                        <span>📚 My Learned Signs</span>
                        <span id="learnedCountBadge" style="background: #1E293B; color: #94A3B8; font-size: 0.75rem; padding: 2px 8px; border-radius: 999px;">0 Signs</span>
                    </div>
                    <span style="font-size: 0.80rem; color: #94A3B8;">User-taught custom signs are persisted and active</span>
                </div>

                <div id="customSignsGrid" class="custom-signs-grid">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>

            <!-- CALIBRATION & TRAINING STUDIO (24 BUILT-IN SIGNS) -->
            <div class="calibration-card">
                <div class="card-header">
                    <div class="card-title">
                        <span>🛠️ Live Calibration Studio (24 Signs)</span>
                    </div>
                    <span style="font-size: 0.8rem; color: #38bdf8;">Custom calibrate standard signs for your camera & posture</span>
                </div>

                <div class="calibration-grid">
                    <div>
                        <label style="font-size: 0.8rem; color: #94a3b8; display: block; margin-bottom: 4px;">Select Sign to Calibrate:</label>
                        <select id="calibSignSelect" style="width: 100%; background: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; font-size: 0.88rem;">
                            <!-- Populated dynamically -->
                        </select>
                    </div>

                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8;">
                            <span id="calibLabelText">Recording Samples:</span>
                            <b id="sampleCountText" style="color: #38bdf8;">0 / 50 samples</b>
                        </div>
                        <div class="sample-progress-bar">
                            <div id="sampleProgressFill" class="sample-progress-fill"></div>
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 8px;">
                            <button id="recordSamplesBtn" class="btn btn-primary" style="flex: 1; justify-content: center; font-size: 0.82rem;" onclick="startSampleCollection()">
                                <span>🔴 Record Samples (50 Frames)</span>
                            </button>
                        </div>
                    </div>

                    <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                        <button class="btn btn-primary" style="flex: 1; justify-content: center; font-size: 0.82rem; background: linear-gradient(135deg, #059669, #10b981);" onclick="trainUpdatedModel()">
                            <span>⚡ Train / Save Sign</span>
                        </button>
                        <button class="btn" style="font-size: 0.82rem;" onclick="resetModelToDefaults()">
                            <span>🔄 Reset Defaults</span>
                        </button>
                    </div>
                </div>
            </div>

            <!-- VOCABULARY REFERENCE MATRIX (24 MVP SIGNS + CUSTOM LEARNED) -->
            <div class="vocab-section">
                <div class="card-header">
                    <div class="card-title">
                        <span>📖 Supported Sign Vocabulary</span>
                    </div>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Click any card to preview voice</span>
                </div>

                <!-- Category Tabs Filter -->
                <div class="category-tabs" id="categoryTabsWrap">
                    <button class="cat-tab active" onclick="filterCategory('ALL', this)">All</button>
                    <button class="cat-tab" onclick="filterCategory('Custom Learned', this)" id="customCatTab">Custom (0)</button>
                    <button class="cat-tab" onclick="filterCategory('Everyday', this)">Everyday (9)</button>
                    <button class="cat-tab" onclick="filterCategory('Needs', this)">Needs (3)</button>
                    <button class="cat-tab" onclick="filterCategory('Actions', this)">Actions (4)</button>
                    <button class="cat-tab" onclick="filterCategory('People & Places', this)">People & Places (2)</button>
                    <button class="cat-tab" onclick="filterCategory('Questions', this)">Questions (3)</button>
                    <button class="cat-tab" onclick="filterCategory('Emotions', this)">Emotions (3)</button>
                </div>

                <div id="vocabGrid" class="vocab-grid">
                    <!-- Populated dynamically via JS -->
                </div>
            </div>
        </div>

        <script>
            // --- VOCABULARY DEFINITION & INITIAL STATE ---
            const BUILTIN_VOCABULARY = {vocab_json};
            const SERVER_CUSTOM_SIGNS = {custom_signs_json};
            const RESERVED_NAMES = new Set({reserved_names_json});

            let VOCABULARY = [...BUILTIN_VOCABULARY];
            let CUSTOM_SIGNS = [];

            // Load Custom Signs from LocalStorage / Server
            try {{
                const localData = localStorage.getItem('sign_custom_user_signs_v1');
                if (localData) {{
                    CUSTOM_SIGNS = JSON.parse(localData);
                }} else if (SERVER_CUSTOM_SIGNS && SERVER_CUSTOM_SIGNS.length > 0) {{
                    CUSTOM_SIGNS = SERVER_CUSTOM_SIGNS;
                    localStorage.setItem('sign_custom_user_signs_v1', JSON.stringify(CUSTOM_SIGNS));
                }}
            }} catch(e) {{
                CUSTOM_SIGNS = SERVER_CUSTOM_SIGNS || [];
            }}

            // Merge Custom Signs into active VOCABULARY
            function syncCustomSignsIntoVocabulary() {{
                VOCABULARY = [...BUILTIN_VOCABULARY];
                CUSTOM_SIGNS.forEach(cs => {{
                    VOCABULARY.push({{
                        id: cs.id,
                        name: cs.name,
                        emoji: cs.emoji || '🧠',
                        phrase: cs.meaning,
                        desc: `Custom taught sign: "${{cs.meaning}}" (${{cs.sample_count || 25}} samples)`,
                        category: 'Custom Learned',
                        color: '#8B5CF6',
                        isCustom: true
                    }});
                }});
                const customTab = document.getElementById('customCatTab');
                if (customTab) {{
                    customTab.textContent = `Custom (${{CUSTOM_SIGNS.length}})`;
                }}
                const countBadge = document.getElementById('learnedCountBadge');
                if (countBadge) {{
                    countBadge.textContent = `${{CUSTOM_SIGNS.length}} Sign${{CUSTOM_SIGNS.length === 1 ? '' : 's'}}`;
                }}
            }}

            // --- DOM ELEMENTS ---
            const videoElement = document.getElementById('webcam');
            const canvasElement = document.getElementById('output_canvas');
            const canvasCtx = canvasElement.getContext('2d');
            const statusBadge = document.getElementById('statusBadge');
            const statusText = document.getElementById('statusText');
            const detectedDisplay = document.getElementById('detectedDisplay');
            const signEmoji = document.getElementById('signEmoji');
            const signName = document.getElementById('signName');
            const signPhrase = document.getElementById('signPhrase');
            const signCategory = document.getElementById('signCategory');
            const confidenceBar = document.getElementById('confidenceBar');
            const confidenceText = document.getElementById('confidenceText');
            const sentenceContent = document.getElementById('sentenceContent');
            const wordCount = document.getElementById('wordCount');
            const autoSpeakToggle = document.getElementById('autoSpeakToggle');
            const skeletonToggle = document.getElementById('skeletonToggle');
            const confSlider = document.getElementById('confSlider');
            const threshVal = document.getElementById('threshVal');
            const voiceSelect = document.getElementById('voiceSelect');
            const fpsTag = document.getElementById('fpsTag');
            const handsTag = document.getElementById('handsTag');
            const toggleCameraBtn = document.getElementById('toggleCameraBtn');

            // Diagnostics HUD elements
            const dbgMpStatus = document.getElementById('dbgMpStatus');
            const dbgCamera = document.getElementById('dbgCamera');
            const dbgLoop = document.getElementById('dbgLoop');
            const dbgFrames = document.getElementById('dbgFrames');
            const dbgHands = document.getElementById('dbgHands');
            const dbgLandmarks = document.getElementById('dbgLandmarks');
            const dbgRawPred = document.getElementById('dbgRawPred');

            // Calibration Studio elements
            const calibSignSelect = document.getElementById('calibSignSelect');
            const recordSamplesBtn = document.getElementById('recordSamplesBtn');
            const calibLabelText = document.getElementById('calibLabelText');
            const sampleCountText = document.getElementById('sampleCountText');
            const sampleProgressFill = document.getElementById('sampleProgressFill');

            // Teach Your Own Sign elements
            const customMeaningInput = document.getElementById('customMeaningInput');
            const customNameInput = document.getElementById('customNameInput');
            const qualityBadge = document.getElementById('qualityBadge');
            const qualityAdvice = document.getElementById('qualityAdvice');
            const customSampleCountText = document.getElementById('customSampleCountText');
            const customProgressFill = document.getElementById('customProgressFill');
            const startTeachBtn = document.getElementById('startTeachBtn');
            const cancelTeachBtn = document.getElementById('cancelTeachBtn');
            const learningSummaryBox = document.getElementById('learningSummaryBox');
            const summarySignName = document.getElementById('summarySignName');
            const summarySignMeaning = document.getElementById('summarySignMeaning');
            const conflictErrorMsg = document.getElementById('conflictErrorMsg');
            const customSignsGrid = document.getElementById('customSignsGrid');

            // --- STATE VARIABLES ---
            let hands = null;
            let isCameraRunning = true;
            let isProcessingFrame = false;
            let confidenceThreshold = 0.75;
            let totalFramesCount = 0;
            let fpsFramesCount = 0;
            let lastFpsCheckTime = performance.now();
            let currentFps = 0;

            // Custom Sign Learning State
            let isCustomLearning = false;
            let customRecordingBuffer = [];
            const CUSTOM_TARGET_SAMPLES = 25;
            const CUSTOM_SIGN_THRESHOLD = 0.85;

            // Temporal Smoothing Buffer
            const SMOOTHING_WINDOW_SIZE = 9;
            const STABILITY_REQUIRED = 5;
            let predictionHistory = [];

            // Debounce for Speech
            let lastSpokenSign = null;
            let lastSpokenTime = 0;
            const SPEAK_DEBOUNCE_MS = 2500;
            let sentenceList = [];
            let currentActiveSign = null;

            // Built-in Signs Calibration State
            let isRecordingSamples = false;
            let targetRecordingSign = null;
            let currentRecordingBuffer = [];
            const TARGET_SAMPLE_COUNT = 50;
            let CUSTOM_MODEL_PROTOTYPES = {{}};

            try {{
                const saved = localStorage.getItem('sign_custom_prototypes_24');
                if (saved) CUSTOM_MODEL_PROTOTYPES = JSON.parse(saved);
            }} catch(e) {{}}

            // --- FEATURE EXTRACTION & NORMALIZATION ---
            function extractNormalizedFeatures(multiHandLandmarks) {{
                if (!multiHandLandmarks || multiHandLandmarks.length === 0) return null;
                const features = [];
                for (let hIdx = 0; hIdx < Math.min(2, multiHandLandmarks.length); hIdx++) {{
                    const h = multiHandLandmarks[hIdx];
                    const w = h[0]; // Wrist as origin
                    const dx = h[9].x - w.x;
                    const dy = h[9].y - w.y;
                    const dz = (h[9].z || 0) - (w.z || 0);
                    const palmScale = Math.hypot(dx, dy, dz) || 1.0;
                    for (let i = 0; i < 21; i++) {{
                        features.push((h[i].x - w.x) / palmScale);
                        features.push((h[i].y - w.y) / palmScale);
                        features.push(((h[i].z || 0) - (w.z || 0)) / palmScale);
                    }}
                }}
                return features;
            }}

            function calculateCentroid(samples) {{
                if (!samples || samples.length === 0) return [];
                const n = samples[0].length;
                const centroid = new Array(n).fill(0);
                samples.forEach(s => {{
                    for (let i = 0; i < n; i++) centroid[i] += s[i];
                }});
                for (let i = 0; i < n; i++) centroid[i] /= samples.length;
                return centroid;
            }}

            function cosineSimilarity(v1, v2) {{
                const n = Math.min(v1.length, v2.length);
                if (n === 0) return 0;
                let dot = 0, mag1 = 0, mag2 = 0;
                for (let i = 0; i < n; i++) {{
                    dot += v1[i] * v2[i];
                    mag1 += v1[i] * v1[i];
                    mag2 += v2[i] * v2[i];
                }}
                mag1 = Math.sqrt(mag1);
                mag2 = Math.sqrt(mag2);
                if (mag1 < 1e-6 || mag2 < 1e-6) return 0;
                return dot / (mag1 * mag2);
            }}

            // --- PROVEN GEOMETRIC RECOGNITION ENGINE (PRIORITY 1) ---
            function dist(p1, p2) {{
                return Math.hypot(p1.x - p2.x, p1.y - p2.y, (p1.z || 0) - (p2.z || 0));
            }}

            function isFingerExtended(lm, tipIdx, pipIdx, mcpIdx) {{
                const wrist = lm[0];
                const dTip = dist(lm[tipIdx], wrist);
                const dPip = dist(lm[pipIdx], wrist);
                return dTip > dPip * 1.22;
            }}

            function isThumbExtended(lm) {{
                const dTipPinky = dist(lm[4], lm[17]);
                const dMcpPinky = dist(lm[2], lm[17]);
                const dTipWrist = dist(lm[4], lm[0]);
                const dMcpWrist = dist(lm[2], lm[0]);
                return (dTipPinky > dMcpPinky * 1.15) && (dTipWrist > dMcpWrist * 1.12);
            }}

            function classifyHandGeometric(multiHandLandmarks) {{
                if (!multiHandLandmarks || multiHandLandmarks.length === 0) return null;

                const h1 = multiHandLandmarks[0];
                const h2 = multiHandLandmarks.length > 1 ? multiHandLandmarks[1] : null;

                const thumbExt = isThumbExtended(h1);
                const indexExt = isFingerExtended(h1, 8, 6, 5);
                const middleExt = isFingerExtended(h1, 12, 10, 9);
                const ringExt = isFingerExtended(h1, 16, 14, 13);
                const pinkyExt = isFingerExtended(h1, 20, 18, 17);
                const extCount = (thumbExt ? 1 : 0) + (indexExt ? 1 : 0) + (middleExt ? 1 : 0) + (ringExt ? 1 : 0) + (pinkyExt ? 1 : 0);

                const dTI = dist(h1[4], h1[8]);
                const dTM = dist(h1[4], h1[12]);

                // Two-Handed Gestures
                if (h2) {{
                    const h1Wrist = h1[0], h2Wrist = h2[0];
                    const h1Ext = [8,12,16,20].filter(i => isFingerExtended(h1, i, i-2, i-3)).length;
                    const h2Ext = [8,12,16,20].filter(i => isFingerExtended(h2, i, i-2, i-3)).length;
                    const handsClose = dist(h1Wrist, h2Wrist) < 0.38;

                    if (handsClose) {{
                        if (h1Ext >= 3 && h2Ext >= 3) {{
                            const palmDist = dist(h1[9], h2[9]);
                            if (palmDist < 0.18) {{
                                return {{ id: 'THANK_YOU', conf: 0.95 }};
                            }}
                        }}
                        if (indexExt && middleExt && !ringExt && !pinkyExt && dist(h1[8], h2[8]) < 0.12) {{
                            return {{ id: 'NAME', conf: 0.93 }};
                        }}
                    }}
                }}

                // Single Hand Gestures
                // 1. HELLO: All 5 fingers extended & spread
                if (extCount >= 4 && indexExt && middleExt && ringExt && pinkyExt) {{
                    const fingerSpread = dist(h1[8], h1[20]);
                    if (fingerSpread > 0.18) {{
                        return {{ id: 'HELLO', conf: 0.96 }};
                    }}
                }}

                // 2. LOVE: 'I Love You' sign
                if (thumbExt && indexExt && !middleExt && !ringExt && pinkyExt) {{
                    return {{ id: 'LOVE', conf: 0.95 }};
                }}

                // 3. YES: Thumbs Up
                if (thumbExt && !indexExt && !middleExt && !ringExt && !pinkyExt) {{
                    if (h1[4].y < h1[3].y && h1[4].y < h1[0].y) {{
                        return {{ id: 'YES', conf: 0.95 }};
                    }}
                }}

                // 4. NO: Thumbs Down or pinch
                if (thumbExt && !indexExt && !middleExt && !ringExt && !pinkyExt) {{
                    if (h1[4].y > h1[0].y) {{
                        return {{ id: 'NO', conf: 0.94 }};
                    }}
                }}

                // 5. GOOD: OK sign
                if (dTI < 0.08 && middleExt && ringExt && pinkyExt) {{
                    return {{ id: 'GOOD', conf: 0.95 }};
                }}

                // 6. STOP: Vertical flat open hand facing camera
                if (extCount >= 4 && indexExt && middleExt && ringExt && pinkyExt && h1[8].y < h1[0].y - 0.20) {{
                    return {{ id: 'STOP', conf: 0.92 }};
                }}

                // 7. GO: Index pointing forward/side
                if (indexExt && !middleExt && !ringExt && !pinkyExt) {{
                    return {{ id: 'GO', conf: 0.91 }};
                }}

                // 8. COME: Index gesturing inward
                if (indexExt && !middleExt && !ringExt && !pinkyExt && h1[8].y > h1[6].y) {{
                    return {{ id: 'COME', conf: 0.89 }};
                }}

                // 9. DRINK: C-shape cup hand
                if (!indexExt && !middleExt && !ringExt && !pinkyExt && dTI > 0.09 && dTI < 0.20) {{
                    return {{ id: 'DRINK', conf: 0.90 }};
                }}

                // 10. NEED: Hooked bent index
                if (indexExt && !middleExt && !ringExt && !pinkyExt && h1[8].y > h1[6].y) {{
                    return {{ id: 'NEED', conf: 0.90 }};
                }}

                // 11. WANT: Curved claw hand
                if (extCount >= 3 && dTI > 0.12 && dTI < 0.22 && h1[8].y > h1[6].y) {{
                    return {{ id: 'WANT', conf: 0.90 }};
                }}

                return null;
            }}

            function classifyHandHybrid(multiHandLandmarks) {{
                // 1. Check custom calibrated model prototypes first
                if (CUSTOM_MODEL_PROTOTYPES && Object.keys(CUSTOM_MODEL_PROTOTYPES).length > 0) {{
                    try {{
                        const feat = extractNormalizedFeatures(multiHandLandmarks);
                        if (feat) {{
                            let bestSign = null, minSqDist = Infinity;
                            for (const [signId, proto] of Object.entries(CUSTOM_MODEL_PROTOTYPES)) {{
                                let sqDist = 0;
                                for (let i = 0; i < Math.min(feat.length, proto.length); i++) {{
                                    sqDist += Math.pow(feat[i] - proto[i], 2);
                                }}
                                if (sqDist < minSqDist) {{
                                    minSqDist = sqDist;
                                    bestSign = signId;
                                }}
                            }}
                            if (bestSign && minSqDist < 0.75) {{
                                const conf = Math.max(0.75, 1.0 - (minSqDist / 2.0));
                                return {{ id: bestSign, conf: conf }};
                            }}
                        }}
                    }} catch (err) {{}}
                }}

                // 2. Fallback to geometric landmark classifier
                return classifyHandGeometric(multiHandLandmarks);
            }}

            // --- CUSTOM SIGN MATCHER (PRIORITY 2) ---
            function matchCustomSigns(feat) {{
                if (!feat || CUSTOM_SIGNS.length === 0) return null;

                let bestSign = null;
                let bestSimilarity = -1;

                for (const sign of CUSTOM_SIGNS) {{
                    let maxSampleSim = -1;
                    if (sign.samples && sign.samples.length > 0) {{
                        const sampleSims = sign.samples.map(s => cosineSimilarity(feat, s));
                        sampleSims.sort((a, b) => b - a);
                        const topK = sampleSims.slice(0, 3);
                        maxSampleSim = topK.reduce((a, b) => a + b, 0) / topK.length;
                    }}

                    let templateSim = -1;
                    if (sign.template && sign.template.length > 0) {{
                        templateSim = cosineSimilarity(feat, sign.template);
                    }}

                    const combined = Math.max(maxSampleSim, templateSim);
                    if (combined > bestSimilarity) {{
                        bestSimilarity = combined;
                        bestSign = sign;
                    }}
                }}

                if (bestSign && bestSimilarity >= CUSTOM_SIGN_THRESHOLD) {{
                    return {{
                        id: bestSign.id,
                        name: bestSign.name,
                        phrase: bestSign.meaning,
                        meaning: bestSign.meaning,
                        emoji: bestSign.emoji || '🧠',
                        category: 'Custom Learned',
                        conf: bestSimilarity,
                        isCustom: true
                    }};
                }}
                return null;
            }}

            // --- QUALITY CHECK DURING LEARNING ---
            function evaluateHandQuality(multiHandLandmarks) {{
                if (!multiHandLandmarks || multiHandLandmarks.length === 0) {{
                    qualityBadge.className = "quality-badge quality-bad";
                    qualityBadge.innerHTML = "<span>🔴 No hand detected</span>";
                    qualityAdvice.textContent = "Show your hand clearly to the webcam";
                    return false;
                }}

                const h1 = multiHandLandmarks[0];
                const wrist = h1[0];
                const dx = h1[9].x - wrist.x;
                const dy = h1[9].y - wrist.y;
                const palmScale = Math.hypot(dx, dy);

                if (palmScale < 0.08) {{
                    qualityBadge.className = "quality-badge quality-warning";
                    qualityBadge.innerHTML = "<span>🟡 Move hand closer</span>";
                    qualityAdvice.textContent = "Hand is too far from camera";
                    return false;
                }} else if (wrist.x < 0.08 || wrist.x > 0.92 || wrist.y < 0.08 || wrist.y > 0.92) {{
                    qualityBadge.className = "quality-badge quality-warning";
                    qualityBadge.innerHTML = "<span>🟡 Center your hand</span>";
                    qualityAdvice.textContent = "Move hand towards camera center";
                    return true;
                }} else {{
                    qualityBadge.className = "quality-badge quality-good";
                    qualityBadge.innerHTML = "<span>🟢 Good hand position</span>";
                    qualityAdvice.textContent = "Move hand slightly to capture natural variations";
                    return true;
                }}
            }}

            // --- PROCESS RECOGNITION FRAME ---
            function processRecognitionResults(results) {{
                const handsDetected = results.multiHandLandmarks ? results.multiHandLandmarks.length : 0;
                handsTag.textContent = `Hands: ${{handsDetected}}`;
                dbgHands.textContent = `${{handsDetected}}`;
                dbgLandmarks.textContent = handsDetected > 0 ? "YES" : "NO";

                // Update Quality Feedback
                const isQualityGood = evaluateHandQuality(results.multiHandLandmarks);

                // 1. Handle Live Custom Sign Learning Sample Collection
                if (isCustomLearning && handsDetected > 0 && isQualityGood) {{
                    const feat = extractNormalizedFeatures(results.multiHandLandmarks);
                    if (feat) {{
                        customRecordingBuffer.push(feat);
                        const progress = Math.min(100, Math.round((customRecordingBuffer.length / CUSTOM_TARGET_SAMPLES) * 100));
                        customProgressFill.style.width = `${{progress}}%`;
                        customSampleCountText.textContent = `${{customRecordingBuffer.length}} / ${{CUSTOM_TARGET_SAMPLES}} samples`;
                        if (customRecordingBuffer.length >= CUSTOM_TARGET_SAMPLES) {{
                            finishCustomSignLearning();
                        }}
                    }}
                }}

                // 2. Handle Live 24 Signs Calibration Sampling
                if (isRecordingSamples && handsDetected > 0) {{
                    const feat = extractNormalizedFeatures(results.multiHandLandmarks);
                    if (feat) {{
                        currentRecordingBuffer.push(feat);
                        const progress = Math.min(100, Math.round((currentRecordingBuffer.length / TARGET_SAMPLE_COUNT) * 100));
                        sampleProgressFill.style.width = `${{progress}}%`;
                        sampleCountText.textContent = `${{currentRecordingBuffer.length}} / ${{TARGET_SAMPLE_COUNT}} samples`;
                        if (currentRecordingBuffer.length >= TARGET_SAMPLE_COUNT) {{
                            finishSampleCollection();
                        }}
                    }}
                }}

                if (handsDetected === 0) {{
                    statusBadge.className = "status-badge ready";
                    statusText.textContent = "Ready - Show your hand";
                    predictionHistory = [];
                    confidenceBar.style.width = "0%";
                    confidenceText.textContent = "0%";
                    highlightVocabCard(null);
                    dbgRawPred.textContent = "--";
                    return;
                }}

                statusBadge.className = "status-badge detecting";
                statusText.textContent = "Detecting Gestures...";

                // --- RECOGNITION HIERARCHY ---
                // Priority 1: Built-in 24 Signs Classifier
                let rawPred = classifyHandHybrid(results.multiHandLandmarks);
                let isCustomMatch = false;

                // Priority 2: Custom Learned Signs Matcher (if standard sign not matched)
                if (!rawPred) {{
                    const feat = extractNormalizedFeatures(results.multiHandLandmarks);
                    if (feat) {{
                        const customPred = matchCustomSigns(feat);
                        if (customPred) {{
                            rawPred = customPred;
                            isCustomMatch = true;
                        }}
                    }}
                }}

                if (rawPred) {{
                    const icon = isCustomMatch ? '🧠 ' : '';
                    dbgRawPred.textContent = `${{icon}}${{rawPred.name || rawPred.id}} (${{Math.round(rawPred.conf * 100)}}%)`;
                }} else {{
                    dbgRawPred.textContent = "🤔 Unknown Sign";
                }}

                if (rawPred && rawPred.conf >= confidenceThreshold) {{
                    predictionHistory.push(rawPred.id);
                }} else {{
                    predictionHistory.push(null);
                }}

                if (predictionHistory.length > SMOOTHING_WINDOW_SIZE) {{
                    predictionHistory.shift();
                }}

                // Majority voting
                const counts = {{}};
                predictionHistory.forEach(id => {{
                    if (id) counts[id] = (counts[id] || 0) + 1;
                }});

                let winnerId = null;
                let maxCount = 0;
                for (const [id, cnt] of Object.entries(counts)) {{
                    if (cnt > maxCount) {{
                        maxCount = cnt;
                        winnerId = id;
                    }}
                }}

                if (winnerId && maxCount >= STABILITY_REQUIRED) {{
                    const vocabItem = VOCABULARY.find(v => v.id === winnerId);
                    if (vocabItem) {{
                        const confPercent = Math.round((rawPred ? rawPred.conf : 0.90) * 100);
                        const isCustom = vocabItem.isCustom || vocabItem.category === 'Custom Learned';
                        
                        detectedDisplay.className = isCustom ? 'detected-display active-custom-sign' : 'detected-display active-sign';
                        signEmoji.textContent = vocabItem.emoji;
                        signName.textContent = vocabItem.name;
                        signPhrase.textContent = `"${{vocabItem.phrase}}"`;
                        
                        if (isCustom) {{
                            signCategory.className = 'sign-category custom-tag';
                            signCategory.textContent = '🧠 CUSTOM SIGN DETECTED';
                            statusBadge.className = 'status-badge custom-detected';
                            statusText.textContent = `🧠 Custom Sign: ${{vocabItem.name}} (${{confPercent}}%)`;
                        }} else {{
                            signCategory.className = 'sign-category';
                            signCategory.textContent = vocabItem.category;
                            statusBadge.className = 'status-badge detected';
                            statusText.textContent = `Sign: ${{vocabItem.name}} (${{confPercent}}%)`;
                        }}

                        confidenceBar.style.width = `${{confPercent}}%`;
                        confidenceText.textContent = `${{confPercent}}%`;

                        highlightVocabCard(vocabItem.id, isCustom);
                        currentActiveSign = vocabItem;

                        // Trigger Debounced Speech
                        const currentTime = Date.now();
                        if (autoSpeakToggle.checked) {{
                            if (lastSpokenSign !== vocabItem.id || (currentTime - lastSpokenTime > SPEAK_DEBOUNCE_MS)) {{
                                speakPhrase(vocabItem.phrase);
                                lastSpokenSign = vocabItem.id;
                                lastSpokenTime = currentTime;

                                if (sentenceList.length === 0 || sentenceList[sentenceList.length - 1] !== vocabItem.phrase) {{
                                    sentenceList.push(vocabItem.phrase);
                                    updateSentenceDisplay();
                                }}
                            }}
                        }}
                    }}
                }} else {{
                    if (rawPred) {{
                        const confPercent = Math.round(rawPred.conf * 100);
                        confidenceBar.style.width = `${{confPercent}}%`;
                        confidenceText.textContent = `${{confPercent}}%`;
                        statusBadge.className = "status-badge warning";
                        statusText.textContent = `Analyzing: ${{rawPred.name || rawPred.id}} (${{confPercent}}%)`;
                    }} else {{
                        confidenceBar.style.width = "10%";
                        confidenceText.textContent = "--";
                        statusBadge.className = "status-badge warning";
                        statusText.textContent = "🤔 Unknown Sign";
                    }}
                }}
            }}

            // --- CANVAS DRAWING ---
            function drawHandSkeleton(results) {{
                canvasCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

                if (!skeletonToggle.checked || !results.multiHandLandmarks) {{
                    canvasCtx.restore();
                    return;
                }}

                const CONNECTIONS = [
                    [0,1],[1,2],[2,3],[3,4],        // Thumb
                    [0,5],[5,6],[6,7],[7,8],        // Index
                    [0,9],[9,10],[10,11],[11,12],   // Middle
                    [0,13],[13,14],[14,15],[15,16], // Ring
                    [0,17],[17,18],[18,19],[19,20], // Pinky
                    [5,9],[9,13],[13,17]           // Palm base
                ];

                for (const landmarks of results.multiHandLandmarks) {{
                    canvasCtx.strokeStyle = isCustomLearning ? 'rgba(168, 85, 247, 0.9)' : 'rgba(6, 182, 212, 0.85)';
                    canvasCtx.lineWidth = 3;
                    for (const [i, j] of CONNECTIONS) {{
                        const p1 = landmarks[i];
                        const p2 = landmarks[j];
                        canvasCtx.beginPath();
                        canvasCtx.moveTo(p1.x * canvasElement.width, p1.y * canvasElement.height);
                        canvasCtx.lineTo(p2.x * canvasElement.width, p2.y * canvasElement.height);
                        canvasCtx.stroke();
                    }}

                    for (let i = 0; i < landmarks.length; i++) {{
                        const p = landmarks[i];
                        const x = p.x * canvasElement.width;
                        const y = p.y * canvasElement.height;
                        
                        canvasCtx.beginPath();
                        if (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) {{
                            canvasCtx.arc(x, y, 6, 0, 2 * Math.PI);
                            canvasCtx.fillStyle = isCustomLearning ? '#C084FC' : '#10b981';
                            canvasCtx.shadowColor = isCustomLearning ? '#C084FC' : '#10b981';
                            canvasCtx.shadowBlur = 10;
                        }} else {{
                            canvasCtx.arc(x, y, 4, 0, 2 * Math.PI);
                            canvasCtx.fillStyle = isCustomLearning ? '#E9D5FF' : '#38bdf8';
                            canvasCtx.shadowBlur = 0;
                        }}
                        canvasCtx.fill();
                    }}
                }}
                canvasCtx.restore();
            }}

            // --- CONTINUOUS FRAME LOOP ---
            async function continuousRenderLoop() {{
                if (!isCameraRunning) {{
                    dbgLoop.textContent = "PAUSED";
                    requestAnimationFrame(continuousRenderLoop);
                    return;
                }}

                dbgLoop.textContent = "RUNNING";
                totalFramesCount++;
                fpsFramesCount++;
                dbgFrames.textContent = `${{totalFramesCount}}`;

                const now = performance.now();
                if (now - lastFpsCheckTime >= 1000) {{
                    currentFps = Math.round((fpsFramesCount * 1000) / (now - lastFpsCheckTime));
                    fpsTag.textContent = `FPS: ${{currentFps}}`;
                    fpsFramesCount = 0;
                    lastFpsCheckTime = now;
                }}

                if (hands && !isProcessingFrame && videoElement.videoWidth > 0 && !videoElement.paused) {{
                    isProcessingFrame = true;
                    try {{
                        await hands.send({{ image: videoElement }});
                    }} catch(err) {{
                        console.error("hands.send error:", err);
                        dbgMpStatus.textContent = "ERROR: " + err.message;
                    }} finally {{
                        isProcessingFrame = false;
                    }}
                }}

                requestAnimationFrame(continuousRenderLoop);
            }}

            // --- MEDIAPIPE INITIALIZATION ---
            async function initMediaPipeHands() {{
                try {{
                    dbgMpStatus.textContent = "LOADING...";
                    statusBadge.className = "status-badge ready";
                    statusText.textContent = "Loading AI Hand Tracker...";

                    hands = new Hands({{
                        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${{file}}`
                    }});

                    hands.setOptions({{
                        maxNumHands: 2,
                        modelComplexity: 1,
                        minDetectionConfidence: 0.60,
                        minTrackingConfidence: 0.60
                    }});

                    hands.onResults((results) => {{
                        if (canvasElement.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {{
                            canvasElement.width = videoElement.videoWidth;
                            canvasElement.height = videoElement.videoHeight;
                        }}
                        drawHandSkeleton(results);
                        processRecognitionResults(results);
                    }});

                    dbgMpStatus.textContent = "READY";
                    await startWebcam();
                }} catch (err) {{
                    console.error("MediaPipe initialization error:", err);
                    dbgMpStatus.textContent = "ERROR: " + (err.message || "Failed to load");
                    statusBadge.className = "status-badge error";
                    statusText.textContent = "MediaPipe Load Error. Check network connection.";
                }}
            }}

            // --- WEBCAM STREAM ---
            async function startWebcam() {{
                try {{
                    dbgCamera.textContent = "REQUESTING ACCESS...";
                    statusBadge.className = "status-badge ready";
                    statusText.textContent = "Requesting Camera Access...";

                    const stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: "user" }},
                        audio: false
                    }});

                    videoElement.srcObject = stream;
                    await videoElement.play();

                    dbgCamera.textContent = "CONNECTED";
                    statusBadge.className = "status-badge ready";
                    statusText.textContent = "Ready - Show your hand";

                    requestAnimationFrame(continuousRenderLoop);
                }} catch (err) {{
                    console.error("Webcam access error:", err);
                    dbgCamera.textContent = "DENIED / ERROR";
                    statusBadge.className = "status-badge error";
                    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {{
                        statusText.textContent = "Camera Permission Denied";
                    }} else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {{
                        statusText.textContent = "No Camera Connected";
                    }} else {{
                        statusText.textContent = "Camera Error: " + (err.message || "Failed to start");
                    }}
                }}
            }}

            function toggleCamera() {{
                if (isCameraRunning) {{
                    isCameraRunning = false;
                    toggleCameraBtn.innerHTML = '<span>▶️ Resume Camera</span>';
                    toggleCameraBtn.className = 'btn';
                    statusBadge.className = "status-badge warning";
                    statusText.textContent = "Camera Paused";
                    dbgCamera.textContent = "PAUSED";
                }} else {{
                    isCameraRunning = true;
                    toggleCameraBtn.innerHTML = '<span>⏹️ Pause Camera</span>';
                    toggleCameraBtn.className = 'btn btn-primary';
                    statusBadge.className = "status-badge ready";
                    statusText.textContent = "Camera Resumed";
                    dbgCamera.textContent = "CONNECTED";
                }}
            }}

            function restartCamera() {{
                startWebcam();
            }}

            // ===================================================
            // 🧠 TEACH YOUR OWN SIGN WORKFLOW
            // ===================================================
            function onCustomMeaningChanged() {{
                const meaning = customMeaningInput.value.trim();
                if (meaning && !customNameInput.dataset.userEdited) {{
                    const shortName = meaning.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 20);
                    customNameInput.value = shortName;
                }}
            }}

            customNameInput.addEventListener('input', () => {{
                customNameInput.dataset.userEdited = "true";
            }});

            function startCustomSignLearning() {{
                const meaning = customMeaningInput.value.trim();
                const name = customNameInput.value.trim();

                conflictErrorMsg.style.display = 'none';

                if (!meaning) {{
                    alert("Please enter what this sign means first (e.g., 'Good Morning').");
                    customMeaningInput.focus();
                    return;
                }}

                const cleanName = (name || meaning).toUpperCase().replace(/[^A-Z0-9_]+/g, '_');
                if (RESERVED_NAMES.has(cleanName)) {{
                    alert("This sign name already exists in the 24 standard signs. Choose another name.");
                    customNameInput.focus();
                    return;
                }}

                isCustomLearning = true;
                customRecordingBuffer = [];
                startTeachBtn.style.display = 'none';
                cancelTeachBtn.style.display = 'inline-flex';
                learningSummaryBox.style.display = 'none';
                customProgressFill.style.width = '0%';
                customSampleCountText.textContent = `0 / ${{CUSTOM_TARGET_SAMPLES}} samples`;
                document.getElementById('teachCard').classList.add('highlight-recording');
            }}

            function cancelCustomSignLearning() {{
                isCustomLearning = false;
                customRecordingBuffer = [];
                startTeachBtn.style.display = 'inline-flex';
                cancelTeachBtn.style.display = 'none';
                document.getElementById('teachCard').classList.remove('highlight-recording');
                customSampleCountText.textContent = `0 / ${{CUSTOM_TARGET_SAMPLES}} samples`;
                customProgressFill.style.width = '0%';
            }}

            function finishCustomSignLearning() {{
                isCustomLearning = false;
                startTeachBtn.style.display = 'inline-flex';
                cancelTeachBtn.style.display = 'none';
                document.getElementById('teachCard').classList.remove('highlight-recording');

                const meaning = customMeaningInput.value.trim();
                const rawName = customNameInput.value.trim() || meaning;
                const cleanName = rawName.toUpperCase().replace(/[^A-Z0-9_]+/g, '_');

                summarySignName.textContent = cleanName;
                summarySignMeaning.textContent = `"${{meaning}}"`;
                learningSummaryBox.style.display = 'flex';
            }}

            function saveCustomSign() {{
                const meaning = customMeaningInput.value.trim();
                const rawName = customNameInput.value.trim() || meaning;
                const cleanName = rawName.toUpperCase().replace(/[^A-Z0-9_]+/g, '_');

                conflictErrorMsg.style.display = 'none';

                if (!meaning) {{
                    conflictErrorMsg.textContent = "Please enter what this sign means.";
                    conflictErrorMsg.style.display = 'block';
                    return;
                }}

                if (RESERVED_NAMES.has(cleanName)) {{
                    conflictErrorMsg.textContent = "This sign name already exists. Choose another name.";
                    conflictErrorMsg.style.display = 'block';
                    return;
                }}

                if (CUSTOM_SIGNS.some(s => s.name === cleanName)) {{
                    conflictErrorMsg.textContent = `A custom sign named '${{cleanName}}' already exists.`;
                    conflictErrorMsg.style.display = 'block';
                    return;
                }}

                if (customRecordingBuffer.length < 10) {{
                    conflictErrorMsg.textContent = "Not enough valid samples captured. Please record again.";
                    conflictErrorMsg.style.display = 'block';
                    return;
                }}

                const centroid = calculateCentroid(customRecordingBuffer);
                const newSign = {{
                    id: 'custom_' + Date.now().toString(36) + '_' + Math.random().toString(36).substr(2, 4),
                    name: cleanName,
                    meaning: meaning,
                    emoji: '🧠',
                    category: 'Custom Learned',
                    sample_count: customRecordingBuffer.length,
                    template: centroid,
                    samples: customRecordingBuffer,
                    created_at: new Date().toISOString()
                }};

                CUSTOM_SIGNS.push(newSign);

                try {{
                    localStorage.setItem('sign_custom_user_signs_v1', JSON.stringify(CUSTOM_SIGNS));
                }} catch(e) {{}}

                syncCustomSignsIntoVocabulary();
                renderCustomSignsList();
                renderVocabCards();

                learningSummaryBox.style.display = 'none';
                customMeaningInput.value = '';
                customNameInput.value = '';
                customNameInput.dataset.userEdited = '';
                customProgressFill.style.width = '0%';
                customSampleCountText.textContent = `0 / ${{CUSTOM_TARGET_SAMPLES}} samples`;

                alert(`✅ Sign learned successfully!\n\nMeaning: "${{meaning}}"\nSamples: ${{newSign.sample_count}}\n\nPerform this gesture to the camera anytime to trigger speech!`);
            }}

            function deleteCustomSign(id) {{
                if (!confirm("Are you sure you want to delete this custom sign?")) return;
                CUSTOM_SIGNS = CUSTOM_SIGNS.filter(s => s.id !== id);
                try {{
                    localStorage.setItem('sign_custom_user_signs_v1', JSON.stringify(CUSTOM_SIGNS));
                }} catch(e) {{}}
                syncCustomSignsIntoVocabulary();
                renderCustomSignsList();
                renderVocabCards();
            }}

            function renderCustomSignsList() {{
                customSignsGrid.innerHTML = '';
                if (CUSTOM_SIGNS.length === 0) {{
                    customSignsGrid.innerHTML = `
                        <div style="grid-column: 1 / -1; padding: 24px; text-align: center; color: #64748B; background: #0B121E; border: 1px dashed #23324A; border-radius: 12px; font-size: 0.88rem;">
                            🧠 No custom signs taught yet. Use the <b>"Teach Your Own Sign"</b> section above to teach the AI your first custom gesture!
                        </div>
                    `;
                    return;
                }}

                CUSTOM_SIGNS.forEach(sign => {{
                    const card = document.createElement('div');
                    card.className = 'custom-sign-item';
                    card.innerHTML = `
                        <div class="custom-sign-top">
                            <span class="custom-sign-name">${{sign.name}}</span>
                            <span style="font-size: 1.2rem;">🧠</span>
                        </div>
                        <div class="custom-sign-meaning">"${{sign.meaning}}"</div>
                        <div class="custom-sign-meta">
                            <span>${{sign.sample_count || 25}} samples</span>
                            <span>Active ✓</span>
                        </div>
                        <div class="custom-sign-actions">
                            <button class="btn btn-purple" style="flex: 1; justify-content: center; font-size: 0.76rem; padding: 6px;" onclick="speakPhrase('${{sign.meaning}}')">
                                <span>🔊 Test</span>
                            </button>
                            <button class="btn btn-danger" style="font-size: 0.76rem; padding: 6px 10px;" onclick="deleteCustomSign('${{sign.id}}')">
                                <span>🗑️</span>
                            </button>
                        </div>
                    `;
                    customSignsGrid.appendChild(card);
                }});
            }}

            // --- BUILT-IN SIGNS CALIBRATION STUDIO ---
            function populateCalibrationDropdown() {{
                calibSignSelect.innerHTML = '';
                BUILTIN_VOCABULARY.forEach(v => {{
                    const opt = document.createElement('option');
                    opt.value = v.id;
                    opt.textContent = `${{v.emoji}} ${{v.name}} (${{v.category}})`;
                    calibSignSelect.appendChild(opt);
                }});
            }}

            function startSampleCollection() {{
                targetRecordingSign = calibSignSelect.value;
                currentRecordingBuffer = [];
                isRecordingSamples = true;
                recordSamplesBtn.disabled = true;
                calibLabelText.textContent = `Recording ${{targetRecordingSign}}...`;
                recordSamplesBtn.innerHTML = `<span>⏳ Recording ${{targetRecordingSign}}... Hold!</span>`;
                sampleProgressFill.style.width = '0%';
                sampleCountText.textContent = `0 / ${{TARGET_SAMPLE_COUNT}} samples`;
            }}

            function finishSampleCollection() {{
                isRecordingSamples = false;
                recordSamplesBtn.disabled = false;
                recordSamplesBtn.innerHTML = '<span>✅ Samples: 50/50 ✓ Saved</span>';
                sampleCountText.textContent = `50 / ${{TARGET_SAMPLE_COUNT}} samples (Ready)`;
                alert(`Captured ${{currentRecordingBuffer.length}} samples for ${{targetRecordingSign}}! Now click 'Train / Save Sign' to apply.`);
            }}

            function trainUpdatedModel() {{
                if (currentRecordingBuffer.length === 0 && !targetRecordingSign) {{
                    alert("Please record 50 samples for a sign first!");
                    return;
                }}
                const numFeatures = currentRecordingBuffer[0].length;
                const centroid = new Array(numFeatures).fill(0);
                currentRecordingBuffer.forEach(s => {{
                    for (let i = 0; i < numFeatures; i++) centroid[i] += s[i];
                }});
                for (let i = 0; i < numFeatures; i++) centroid[i] /= currentRecordingBuffer.length;

                CUSTOM_MODEL_PROTOTYPES[targetRecordingSign] = centroid;
                try {{
                    localStorage.setItem('sign_custom_prototypes_24', JSON.stringify(CUSTOM_MODEL_PROTOTYPES));
                }} catch(e) {{}}

                alert(`Successfully calibrated '${{targetRecordingSign}}'! Custom ML model is now active.`);
            }}

            function resetModelToDefaults() {{
                try {{
                    localStorage.removeItem('sign_custom_prototypes_24');
                }} catch(e) {{}}
                CUSTOM_MODEL_PROTOTYPES = {{}};
                sampleProgressFill.style.width = '0%';
                sampleCountText.textContent = `0 / ${{TARGET_SAMPLE_COUNT}} samples`;
                alert("Restored factory default recognition!");
            }}

            // --- SPEECH SYNTHESIS ---
            let synth = window.speechSynthesis;
            let voices = [];

            function populateVoices() {{
                if (!synth) return;
                voices = synth.getVoices();
                voiceSelect.innerHTML = '<option value="">Default Voice</option>';
                voices.forEach((v, idx) => {{
                    if (v.lang.startsWith('en') || v.lang.startsWith('hi')) {{
                        const opt = document.createElement('option');
                        opt.value = idx;
                        opt.textContent = `${{v.name}} (${{v.lang}})`;
                        voiceSelect.appendChild(opt);
                    }}
                }});
            }}

            if (synth) {{
                populateVoices();
                if (speechSynthesis.onvoiceschanged !== undefined) {{
                    speechSynthesis.onvoiceschanged = populateVoices;
                }}
            }}

            function speakPhrase(text) {{
                if (!synth || !text) return;
                try {{
                    synth.cancel();
                    const utter = new SpeechSynthesisUtterance(text);
                    const selectedIdx = voiceSelect.value;
                    if (selectedIdx !== "" && voices[selectedIdx]) {{
                        utter.voice = voices[selectedIdx];
                    }}
                    utter.rate = 1.0;
                    synth.speak(utter);
                }} catch (err) {{
                    console.error("SpeechSynthesis error:", err);
                }}
            }}

            function speakCurrentSign() {{
                if (currentActiveSign && currentActiveSign.phrase) {{
                    speakPhrase(currentActiveSign.phrase);
                }} else if (sentenceList.length > 0) {{
                    speakPhrase(sentenceList.join(". "));
                }} else {{
                    speakPhrase("Hello!");
                }}
            }}

            function clearSentence() {{
                sentenceList = [];
                currentActiveSign = null;
                lastSpokenSign = null;
                updateSentenceDisplay();
                resetDetectedCard();
                highlightVocabCard(null);
            }}

            function updateSentenceDisplay() {{
                if (sentenceList.length === 0) {{
                    sentenceContent.innerHTML = '<span style="color: #64748b; font-style: italic;">Detected spoken phrases will accumulate here...</span>';
                    wordCount.textContent = '0 phrases';
                }} else {{
                    sentenceContent.textContent = sentenceList.join("  •  ");
                    wordCount.textContent = `${{sentenceList.length}} phrase${{sentenceList.length > 1 ? 's' : ''}}`;
                    sentenceContent.scrollTop = sentenceContent.scrollHeight;
                }}
            }}

            function resetDetectedCard() {{
                detectedDisplay.className = 'detected-display';
                signEmoji.textContent = "✋";
                signName.textContent = "Waiting for sign...";
                signPhrase.textContent = "Show your hand clearly to the camera";
                signCategory.className = "sign-category";
                signCategory.textContent = "Ready";
                confidenceBar.style.width = "0%";
                confidenceText.textContent = "0%";
            }}

            function updateSettings() {{
                confidenceThreshold = parseInt(confSlider.value) / 100.0;
                threshVal.textContent = `${{confSlider.value}}%`;
            }}

            // --- VOCABULARY CARDS & CATEGORY FILTERING ---
            let activeCategory = 'ALL';

            function filterCategory(cat, tabEl) {{
                activeCategory = cat;
                document.querySelectorAll('.cat-tab').forEach(t => t.classList.remove('active'));
                if (tabEl) tabEl.classList.add('active');
                renderVocabCards();
            }}

            function renderVocabCards() {{
                const grid = document.getElementById('vocabGrid');
                grid.innerHTML = '';
                const filtered = activeCategory === 'ALL' ? VOCABULARY : VOCABULARY.filter(v => v.category === activeCategory);

                filtered.forEach(item => {{
                    const card = document.createElement('div');
                    card.className = 'vocab-card';
                    card.id = 'card_' + item.id;
                    const isCustom = item.isCustom || item.category === 'Custom Learned';
                    const customBadge = isCustom ? '<span style="color:#A78BFA; font-size:0.68rem; font-weight:700;">LEARNED</span>' : '';
                    card.innerHTML = `
                        <div class="vocab-top">
                            <span class="vocab-name">${{item.name}}</span>
                            <div class="vocab-icon-box">${{item.emoji}}</div>
                        </div>
                        <div class="vocab-desc">${{item.desc}}</div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-size:0.75rem; color:${{isCustom ? '#A78BFA' : '#60A5FA'}}; font-weight:600;">"${{item.phrase}}"</div>
                            ${{customBadge}}
                        </div>
                    `;
                    card.onclick = () => {{
                        speakPhrase(item.phrase);
                        highlightVocabCard(item.id, isCustom);
                    }};
                    grid.appendChild(card);
                }});
            }}

            function highlightVocabCard(id, isCustom = false) {{
                document.querySelectorAll('.vocab-card').forEach(c => {{
                    c.classList.remove('highlight');
                    c.classList.remove('highlight-custom');
                }});
                if (id) {{
                    const el = document.getElementById('card_' + id);
                    if (el) {{
                        el.classList.add(isCustom ? 'highlight-custom' : 'highlight');
                    }}
                }}
            }}

            // --- WINDOW ONLOAD ---
            window.addEventListener('DOMContentLoaded', () => {{
                syncCustomSignsIntoVocabulary();
                renderCustomSignsList();
                renderVocabCards();
                populateCalibrationDropdown();
                initMediaPipeHands();
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=1480, scrolling=True)


def render_sign_to_speech_page():
    """Renders the full Sign-to-Speech page with matching header and documentation."""
    
    # Top Product Bar
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 18px; background: #0D1420; border: 1px solid #23324A; border-radius: 10px; margin-bottom: 18px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 0.82rem; font-weight: 800; color: #F8FAFC; letter-spacing: 0.04em;">
                    🤟 SIGNBRIDGE AI
                </span>
                <span style="color: #64748B; font-size: 0.80rem;">|</span>
                <span style="font-size: 0.78rem; color: #94A3B8; font-weight: 500;">
                    Two-Way AI Accessibility Platform
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.35); padding: 4px 10px; border-radius: 999px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #22C55E; display: inline-block;"></span>
                <span style="font-size: 0.74rem; font-weight: 700; color: #22C55E; letter-spacing: 0.04em;">AI SYSTEM ONLINE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sign to Speech Hero Card
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #111A28 0%, #162235 100%);
            border: 1px solid #23324A;
            border-radius: 16px;
            padding: 22px 26px;
            margin-bottom: 22px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.20);
        ">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                <div style="max-width: 700px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span style="font-size: 1.3rem;">🤟</span>
                        <span style="font-size: 0.78rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase;">SIGN TO SPEECH</span>
                    </div>
                    <h1 style="margin: 0; font-size: 1.85rem; color: #F8FAFC; font-weight: 800; letter-spacing: -0.02em;">
                        Turn gestures into spoken communication
                    </h1>
                    <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.93rem; line-height: 1.5;">
                        Real-time MediaPipe hand tracking converts recognized signs directly into live captions and natural text-to-speech voice.
                    </p>
                </div>
                <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                    <span style="background: #172F4A; color: #60A5FA; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.80rem; border: 1px solid #214B73;">⚡ 24 Signs</span>
                    <span style="background: #172F4A; color: #A78BFA; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.80rem; border: 1px solid #7C3AED;">🧠 Teach Signs</span>
                    <span style="background: #172F4A; color: #22D3EE; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.80rem; border: 1px solid #214B73;">📷 MediaPipe Hands</span>
                    <span style="background: #172F4A; color: #38BDF8; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.80rem; border: 1px solid #214B73;">🤖 ML Hybrid Engine</span>
                    <span style="background: #172F4A; color: #22C55E; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 0.80rem; border: 1px solid #214B73;">🔊 Web Speech TTS</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    render_sign_to_speech_component()

