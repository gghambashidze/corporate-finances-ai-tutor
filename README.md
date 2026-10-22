# Corporate Finances AI Tutor

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) system engineered for zero-trust financial environments. This solution demonstrates how organizations can leverage high-performance financial intelligence locally on edge hardware—eliminating data privacy risks, third-party API dependencies, and operational overhead.

## Executive Summary

Modern corporate finance departments handle sensitive internal data, valuation models, and proprietary reporting. Transporting this data to external cloud-hosted LLM APIs poses severe compliance, privacy, and cost challenges. 

This repository presents a fully local, self-contained architecture that brings advanced AI tutoring, document retrieval, and visual reasoning directly to edge hardware. By operating entirely offline, it guarantees absolute data sovereignty while delivering deterministic precision for complex financial analysis.

## Core Architectural Pillars

- Zero-Trust Data Sovereignty: Runs 100% locally using lightweight LLM engines (`llama3.2:1b`) and embeddings (`nomic-embed-text`) managed via Ollama. Ensures zero external network requests and $0 recurring API costs.
- Deterministic Arithmetic Verification: Features an independent math validation layer that verifies calculations prior to output generation, preventing LLM hallucinations in financial computations.
- Multi-Modal Ingestion & OCR: Integrates Tesseract OCR to digitize physical financial worksheets, handwritten notes, and printed statements into indexed digital assets.
- Visual Modeling & Multimodal Feedback: Generates dynamic financial charts via Matplotlib alongside local text-to-speech feedback via `gTTS` for active learning.
- Persistent State Architecture: Native JSON session handling and ChromaDB vector persistence ensure seamless session retention and contextual continuity across restarts.

## System Architecture & Technical Stack

- Interface: Streamlit
- Local Intelligence & Embeddings: Ollama (`llama3.2:1b`, `nomic-embed-text`)
- Vector Storage: ChromaDB (`./chroma_db`)
- Optical Character Recognition: Tesseract (`pytesseract`)
- Audio Synthesis & Plotting: `gTTS` / Matplotlib / NumPy

## Deployment & Setup

1. Clone Repository:
   ```bash
   git clone [https://github.com/gghambashidze/corporate-finances-ai-tutor.git](https://github.com/gghambashidze/corporate-finances-ai-tutor.git)
   cd corporate-finances-ai-tutor

