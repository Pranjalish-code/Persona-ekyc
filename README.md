# Persona: AI-Based eKYC Verification System

## Overview

Persona is an AI-powered Electronic Know Your Customer (eKYC) verification system designed to automate identity verification through facial recognition and liveness detection. The system verifies whether a user presenting an identity document is physically present and matches the face shown on the uploaded ID.

The application combines ArcFace-based face recognition, anti-spoofing liveness detection, and a modern Streamlit interface to provide a secure and user-friendly verification experience.

---

## Features

* **Government ID Verification**

  * Upload passport, Aadhaar card, driving license, or other identity documents.
  * Automatic face detection and extraction from ID images.

* **Face Recognition**

  * ArcFace-based facial embeddings.
  * Cosine similarity matching between ID photo and live user face.

* **Liveness Detection**

  * Detects spoofing attempts using recorded videos, photos, or screens.
  * Analyzes facial movements and video frames to ensure real user presence.

* **Live Camera Support**

  * Real-time webcam capture using WebRTC.
  * Supports both live recording and uploaded video verification.

* **GPU Acceleration**

  * Optional ONNX Runtime GPU support for faster inference.

* **Interactive Dashboard**

  * Built using Streamlit.
  * Real-time verification status and confidence scores.

---

## System Architecture

```text
ID Image
    │
    ▼
Face Detection & Extraction
    │
    ▼
ArcFace Embedding Generation
    │
    ├───────────────┐
    │               │
    ▼               ▼
Live Video      Liveness Detection
    │               │
    ▼               ▼
Face Embedding   LIVE / SPOOF
    │
    ▼
Cosine Similarity
    │
    ▼
Final Verification Decision
```

---

## Tech Stack

### Frontend

* Streamlit
* Streamlit WebRTC

### Computer Vision

* OpenCV
* Pillow
* NumPy

### Deep Learning

* ArcFace
* ONNX Runtime
* InsightFace

### Backend

* Python

---

## Project Structure

```text
persona-ekyc/
│
├── app.py                 # Main Streamlit application
├── arcface.py             # ArcFace model and embedding generation
├── idphotoextract.py      # ID face extraction module
├── newlivecheck.py        # Liveness detection module
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd persona-ekyc
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will launch at:

```text
http://localhost:8501
```

---

## Verification Workflow

### Step 1: Upload ID

* Upload a government-issued identity document.
* The system detects and extracts the face.
* ArcFace generates a facial embedding.

### Step 2: Liveness Verification

Choose one of:

* Live webcam capture
* Upload a selfie video

The system analyzes facial movements and liveness signals.

### Step 3: Face Matching

* Face embeddings are generated from the verification video.
* Cosine similarity is computed against the ID embedding.
* Verification passes only if:

  * Liveness = LIVE
  * Similarity ≥ Threshold

---

## Performance

* Real-time verification
* GPU acceleration support
* Robust face matching using ArcFace
* Anti-spoofing protection against replay attacks

---

## Future Enhancements

* OCR-based ID data extraction
* Multi-document support
* Deepfake detection
* Cloud deployment
* Audit logging and compliance dashboard
* API integration for enterprise KYC workflows

---

## Author

**Pranjal Gupta**

B.Tech, Motilal Nehru National Institute of Technology (MNNIT) Allahabad

Interested in Machine Learning, Computer Vision, Generative AI, and Intelligent Identity Verification Systems.

---

## License

This project is intended for educational, research, and demonstration purposes.
