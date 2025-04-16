# 📬 Intelligent Voice-Based Email Assistant for Visually Impaired Users

This project is an accessible, intelligent voice-controlled email assistant designed for visually impaired users. It integrates voice commands, biometric voice authentication, gesture recognition, and a user-friendly web interface to provide a seamless email experience.

---

## 🔧 1. Tech Stack Overview

### 👨‍💻 Backend
- **Frameworks**: Python, FastAPI / Flask
- **Models**: Speech Recognition, NLP, Voice Biometrics
- **ML Libraries**: TensorFlow, PyTorch
- **Database**: PostgreSQL / MongoDB (User data, logs, voiceprints)

### 🌐 Frontend (Web App)
- **UI Framework**: React.js
- **Styling**: Tailwind CSS / Material UI
- **Voice/Gesture Tools**:
  - Web Speech API / JavaScript SpeechRecognition
  - MediaPipe / TensorFlow.js
- **Accessibility**:
  - ARIA labels
  - Keyboard navigation
  - High-contrast mode

---

## 🎯 2. Core Features & Build Guide

### ✅ A. Voice-Controlled Email System
**Goal**: Enable voice-based email interactions (send, read, compose).

- Uses `speech_recognition`, `pyttsx3`, `smtplib`
- Shift to backend APIs using FastAPI/Flask:
  - `POST /send-email`
  - `GET /get-speech-input`

---

### 🔐 B. Voice Authentication (Biometric Verification)

**Voice Enrollment**
- Record 3–5 phrases
- Extract MFCCs or spectrograms (via `librosa`)
- Store embeddings using Siamese Network or pre-trained models like `Resemblyzer` / `DeepSpeaker`

**Authentication**
- Capture voice on login
- Match embeddings using cosine similarity

**Libraries**:
- `librosa`, `Resemblyzer`, `speechbrain`
- (Optional) Train on `VoxCeleb` dataset

---

### 🧠 C. NLP & Context Awareness

- Understands vague commands (e.g., “email mom”)
- Context memory for follow-ups (e.g., “Yes, send it.”)
- Autocorrect and intent/entity recognition

**Tools**:
- `spaCy`
- `transformers` (BERT, T5)
- `SymSpell` / `jamspell` for spellcheck

---

### 🤖 D. Gesture Recognition (Assist Partial Vision)

**Use Case**: Trigger voice commands via hand gestures

- ✋ Open Hand → Pause
- 👈 Point → Back
- ✌️ Two Fingers → Compose

**Tools**:
- `MediaPipe`
- `TensorFlow.js` (real-time detection)

---

### ✨ E. Web Interface (Accessibility-First)

- High-contrast toggle
- Font resizing
- Full keyboard support
- Voice readouts using `speechSynthesis` API

**Libraries**:
- `react-aria`, `chakra-ui`, `Tailwind CSS`
- (Optional) Integrate with ChromeVox or NVDA

---

## ☁️ F. Deployment & Storage

- **Backend**: Render / Railway / Heroku
- **Frontend**: Vercel / Netlify
- **Storage**:
  - Voiceprints encrypted/hashed
  - Use SQLite for dev, PostgreSQL in prod

---

## 🧩 Bonus Add-ons

- 📥 Inbox voice reader (reads unread mails aloud)
- 🛑 Wake word detection ("Hey Mansi")
- 🎙️ Voice note → Email draft
- 📦 PWA for offline support

---

## 📸 System Architecture Diagram


![image](https://github.com/user-attachments/assets/59ac9ed2-2a12-4fcb-9be6-63401425d1ef)


---
