# Real-Time Contactless Heart Rate Monitoring Using Facial Recognition

## Project Overview

This project implements a real-time, non-contact heart rate monitoring system using **facial recognition**, **Eulerian Video Magnification**, and **machine learning**. The system analyzes subtle skin tone variations caused by blood flow, captured through a video feed, and estimates the subject's heart rate. It eliminates the need for physical contact, making it ideal for telemedicine, fitness tracking, and smart healthcare applications.

---

## Key Features

- Real-time face detection using Haar Cascade
- Motion stabilization and noise reduction preprocessing
- Eulerian Video Magnification to enhance subtle skin color changes
- Fast Fourier Transform (FFT) for frequency signal analysis
- Heart rate prediction using ML models (SVR, Decision Tree, AdaBoost, etc.)
- Model evaluation against ECG ground truth data

---

## System Architecture

```mermaid
graph TD
    A[Video Input] --> B[Preprocessing]
    B --> C[Face Detection]
    C --> D[Eulerian Video Magnification]
    D --> E[Signal Extraction & FFT]
    E --> F[Heart Rate Prediction]
    F --> G[Model Evaluation]
