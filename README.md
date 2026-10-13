# 🧠 Brain Tumor Detection System with AI Medical Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-91.16%25-brightgreen.svg)]()
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-blueviolet.svg)]()

An end-to-end Deep Learning system that detects and classifies brain tumors from MRI scans with **91.16% accuracy**, integrated with a **Google Gemini LLM chatbot** and **voice assistant** (STT + TTS) for intelligent patient interaction and accessibility.

---

## 🎯 Overview

Brain tumors are life-threatening conditions where **early detection significantly improves survival rates**. Manual MRI analysis takes **15–30 minutes per scan** and requires specialized radiologists who are in short supply, especially in rural areas.

This project provides an **AI-powered diagnostic support system** that:

- ✅ Detects brain tumors from MRI scans in **2–3 seconds**
- ✅ Classifies tumors into **4 categories** (Glioma, Meningioma, Pituitary, No Tumor)
- ✅ Achieves **91.16% accuracy** using Transfer Learning (MobileNetV2)
- ✅ Provides **AI-powered medical Q&A** through Google Gemini LLM
- ✅ Enables **hands-free interaction** via voice assistant (STT + TTS)
- ✅ Generates **automated clinical reports**


## ✨ Features

### 🧠 Deep Learning Detection
- 4-class tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- Transfer Learning with MobileNetV2 (pre-trained on ImageNet)
- Real-time inference in 2–3 seconds per scan
- Confidence scores for every prediction

### 🤖 LLM-Powered Medical Chatbot
- Google Gemini API integration
- Answers patient questions about tumor types, symptoms, and treatments
- Personalized advice based on detected tumor
- Suggests hospitals and specialists based on user location

### 🎤 Voice Assistant
- **Speech-to-Text (STT)** using SpeechRecognition
- **Text-to-Speech (TTS)** using Google Text-to-Speech (gTTS)
- Hands-free navigation and voice-read diagnosis results
- Accessibility enhancement for visually impaired users

### 📊 Clinical Dashboard
- Patient information management
- Interactive visualizations (confusion matrix, probability gauges)
- Automated clinical report generation and download
- HIPAA-compliant design principles

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.13, Keras, MobileNetV2 |
| **Image Processing** | OpenCV, Pillow (PIL), NumPy |
| **Data Handling** | Pandas, Scikit-learn |
| **LLM / GenAI** | Google Gemini API |
| **Voice AI** | SpeechRecognition (STT), gTTS (TTS), pyttsx3 |
| **Web Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Development** | Google Colab, Jupyter Notebook |
| **Version Control** | Git, GitHub |
| **Deployment** | Streamlit Cloud, Ngrok, Serveo |

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | **91.16%** |
| **Precision** | 91.98% |
| **Recall** | 90.09% |
| **F1-Score** | 0.89 |
| **Inference Time** | 2–3 seconds |
| **Training Time** | 52 minutes |
### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.92 | 0.91 | 0.91 |
| Meningioma | 0.91 | 0.88 | 0.89 |
| No Tumor | 0.93 | 0.94 | 0.93 |
| Pituitary | 0.90 | 0.89 | 0.89 |

### Dataset Split

| Set | Samples | Percentage |
|-----|---------|------------|
| Training | 1,445 | 70% |
| Validation | 310 | 15% |
| Testing | 310 | 15% |
| **Total** | **2,065** | **100%** |


