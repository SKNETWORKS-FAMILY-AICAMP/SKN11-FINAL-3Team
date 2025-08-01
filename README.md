# 🤖 genbot Web Chatbot

AI 기반 상담형 챗봇 서비스 입니다  

---

## 📚 Contents

1. [👥 Introduce Team](#-introduce-team)
2. [🧩 Project Overview](#-project-overview)
3. [🛠️ Technology Stack & Models](#-technology-stack--models)
4. [✨ Key Features](#-key-features)
5. [🔁 Model Workflow](#-model-workflow)
6. [🚀 Deployment Pipeline](#-deployment-pipeline)
7. [🎬 Live Demo](#-live-demo)

---

## 👥 Introduce Team

|[@김성지](https://github.com/kimseoungji0801)|[@김장수](https://github.com/js-kkk)|[@김정원](https://github.com/Kimjeongwon12)|[@현유경](https://github.com/yugyeongh)|
|------|------|------|------|
| <img src="https://github.com/user-attachments/assets/a1ac1e5d-1ebf-4a76-b415-60b718b26c8c" width="200"/> | <img src="https://github.com/user-attachments/assets/86a6b099-43f7-4b43-86f0-e0738ac5b7a0" width="200"/> | <img src="https://github.com/user-attachments/assets/6aa27c80-d308-4ed1-a3ee-526f17e149be" width="200"/> | <img src="https://github.com/user-attachments/assets/dbb903a4-d079-4671-b257-606920967396" width="200"/> | <img src="https://github.com/user-attachments/assets/a4554b6a-e7ef-44d2-9799-09cc7bb78402" width="200"/> |

---

## 🧩 Project Overview

> “AI 챗봇 상담사를 누구나 쉽게 만든다”

본 프로젝트는 소상공인과 기업을 위한 **AI 상담형 챗봇**입니다.  
고객의 반복 질문에 대해 **빠르고 정확하게 응답**하며, 음성 및 텍스트를 모두 지원합니다.  

---

## 🛠️ Technology Stack & Models

| 구성 요소 | 사용 기술 |
|-----------|-----------|
| Frontend  | React, CSS, Web Speech API |
| Backend   | FastAPI, LangGraph|
| Vector DB | FAISS (HuggingFace Embeddings) |
| Model     | `K-intelligence/Midm-2.0`, QLoRA 튜닝 |
| Infra     | RunPod, S3, EFS, EC2 (with CUDA/GPU) |

---

## ✨ Key Features

- 🔄 **멀티턴 챗봇 인터페이스** (LangGraph 기반)
- 🎤 **음성 입력(STT)** 및 TTS 출력
- 📄 **PDF 기반 RAG 검색**
- 🧠 **친절한 말투로 튜닝된 상담사 모델**
- 📦 실시간 DB 저장 및 세션 관리
- 📱 반응형 UI, 모바일 지원

---

## 🔁 Model Workflow

```mermaid
graph LR
A[User Input] --> B[STT / Text Parsing]
B --> C[LangGraph: Retriever Node]
C --> D[FAISS DB 검색]
D --> E[LangGraph: Generator Node]
E --> F[sLLM 응답 생성 (Midm-2.0)]
F --> G[Response to User + DB 저장]
