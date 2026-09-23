# 🤟 SignBridge AI

**Speak. Sign. Understand.**

SignBridge AI is an accessibility-focused platform that converts spoken or typed language into meaningful **Indian Sign Language (ISL) concepts** and presents them through a 3D human avatar.

## 🚀 Overview

```text
Audio / Text
    ↓
Speech-to-Text / Text Processing
    ↓
Language & Meaning Detection
    ↓
ISL Concept Mapping
    ↓
Sign Database
    ↓
3D Avatar Animation
```

The current prototype uses **single-sign animation**. A sentence is converted into meaningful ISL concepts and the avatar plays the available signs sequentially.

Example:

```text
"Hi, what are you doing?"
        ↓
HI → WHAT → YOU → DO
        ↓
Individual sign animations
        ↓
3D avatar
```

## ✨ Current Features

- 🎙️ Voice recording and audio upload
- 📝 Text input
- 🌍 Multilingual speech/text processing
- 🤟 ISL concept mapping
- 📖 CSV-based ISL sign database
- 🧑‍💻 Real-time Three.js/WebGL 3D avatar
- ▶️ Sequential sign playback
- ⏯️ Play/pause, replay, previous/next and speed controls
- 🗂️ Translation history and saved phrases
- ⚠️ Separate tracking of vocabulary and animation availability

## 🛠️ Technology Stack

### Application
- Python
- Streamlit
- Three.js
- WebGL

### AI / Speech
- OpenAI Whisper
- SpeechRecognition
- Google Translator integration
- Semantic ISL mapping

### Backend
- Node.js
- Express.js

### Data
- `sign_database.csv`
- Planned migration to Supabase PostgreSQL

### Development
- Antigravity IDE
- Git
- GitHub

## 📁 Project Structure

```text
signbridge-ai-main/
├── app.py
├── server.js
├── sign_database.csv
├── requirements.txt
├── package.json
├── .gitignore
├── sanket/
│   └── MaleModelSankit.fbx
├── ui_components/
├── custom_signs.py
├── custom_signs.json
├── emotion_engine.py
├── action_response_registry.py
├── extracted_signs.js
├── clean_signs_library.js
└── README.md
```

## 📖 ISL Sign Database

The sign database is the source of truth for vocabulary and animation mapping.

Planned schema:

```text
sign_id
canonical_gloss
word
synonyms
category
animation_id
animation_available
source
notes
```

Example:

```csv
S001,HI,hi,"hey|hello",Greeting,HI,true,ISLRTC,Core greeting
S002,WHAT,what,,Question,WHAT,true,ISLRTC,Question sign
S003,WHEN,when,,Question,WHEN,false,ISLRTC,Animation pending
```

### Important

Vocabulary availability and animation availability are separate.

```text
SIGN EXISTS
    ≠
ANIMATION EXISTS
```

The application must never replace an unavailable sign with an unrelated animation.

## 🎬 Single-Sign Mode

The current development direction is:

```text
Input
 ↓
Meaningful concepts
 ↓
Canonical ISL glosses
 ↓
Sign database
 ↓
Individual animations
 ↓
Existing avatar
```

Examples:

```text
Hi
→ HI
→ HI animation
```

```text
Where is the hospital?
→ HOSPITAL → WHERE
→ individual animations
```

The avatar itself is treated as a fixed component while the vocabulary and translation system are expanded.

## 🎧 Audio Pipeline

```text
Audio
 ↓
Audio preprocessing
 ↓
Speech recognition
 ↓
Transcript
 ↓
Meaning extraction
 ↓
ISL concepts
 ↓
Sign database
 ↓
Avatar animation
```

Text and audio are intended to use the same ISL mapping pipeline.

## ⚠️ Animation Availability

The UI should clearly distinguish:

```text
HI
✓ Animated
```

from:

```text
WHEN
⚠ Animation pending
```

The system should not:

- play a random animation
- use `HELLO` as a fallback for another sign
- claim an unavailable animation is complete

## 📈 Roadmap

### Phase 1 — Foundation
- [x] Speech input
- [x] Text input
- [x] Speech-to-text
- [x] Initial sign database
- [x] 3D avatar
- [x] Single-sign playback

### Phase 2 — Sign Database
- [ ] Clean existing vocabulary
- [ ] Expand toward 500+ useful ISL concepts
- [ ] Add synonyms and categories
- [ ] Track animation availability
- [ ] Add database validation

### Phase 3 — AI Translation
- [ ] Improve sentence meaning extraction
- [ ] Improve ISL concept ordering
- [ ] Improve multilingual semantic mapping
- [ ] Add context-aware translation

### Phase 4 — Production Database
- [ ] Supabase PostgreSQL
- [ ] User profiles
- [ ] Translation history
- [ ] Saved phrases
- [ ] Learning progress

### Phase 5 — Deployment
- [ ] Production frontend
- [ ] Production backend
- [ ] Secure environment variables
- [ ] Cloud deployment
- [ ] Performance optimization

## ▶️ Run Locally

### Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

### Node / Express

```bash
npm install
npm run start
```

Backend:

```text
http://localhost:5000
```

## 🔐 Security

Never commit:

```text
.env
API keys
passwords
tokens
private credentials
```

Use environment variables for secrets.

## 🧪 Basic Tests

Try:

```text
Hi
What?
When?
How?
Where?
I need water
Help me
Where is the hospital?
```

Verify:

- correct transcript
- correct ISL concept
- correct animation lookup
- correct sequence count
- correct avatar playback
- unavailable animations are reported honestly

## 📌 Project Status

**Active Development**

SignBridge AI is currently a working prototype. The vocabulary database, animation coverage, semantic translation, and production database architecture are being expanded.

A vocabulary count should not be treated as the same number of completed avatar animations.

## 🎯 Vision

SignBridge AI aims to make communication more accessible by connecting:

```text
Any supported spoken language
        ↓
Speech / Text
        ↓
Meaning
        ↓
Indian Sign Language
        ↓
3D Human Avatar
```

## 🔗 GitHub

https://github.com/prathikbg27-eng/signbridge-ai
