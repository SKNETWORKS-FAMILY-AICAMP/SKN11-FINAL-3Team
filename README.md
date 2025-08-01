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
"고객 상담을 더 빠르고 정확하게, 그리고 친절하게"

GenBot은 사용자의 질문을 자연스럽고 친절하게 처리하는 AI 상담형 챗봇입니다.
기존 챗봇이 대화의 흐름을 이해하고 사용자 맞춤형 응대를 제공하도록 설계되었습니다.

프로젝트의 Web 챗봇은 다음과 같은 목표를 가지고 개발되었습니다:

📌 목표

1. 기업 및 소상공인을 위한 자동 상담 솔루션 제공

2. 텍스트/음성 기반 양방향 커뮤니케이션 지원

3. 반복되는 고객 문의에 대해 즉각적이고 친절한 응답 생성

4. 도메인 문서 기반 RAG(문서 기반 검색+응답) 처리

🚀 특징

1. 자연스러운 멀티턴 대화 흐름 지원 (LangGraph 활용)

2. 사용자가 말하면 즉시 응답하는 실시간 STT/TTS 연동

3. 도메인 문서를 벡터화하여 고객 맞춤형 정보 제공

4. 소형 SLLM을 QLoRA로 튜닝하여 저비용 운영

---

## 🛠️ Technology Stack & Models

| 구성 요소           | 사용 기술 / 모델                                          | 설명 |
|--------------------|------------------------------------------------------------|------|
| **Frontend**       |   <img src="https://img.shields.io/badge/react-61DAFB?style=for-the-badge&logo=react&logoColor=white" style="display: inline-block; margin: 5px;"> <img src="https://img.shields.io/badge/css-663399?style=for-the-badge&logo=css&logoColor=white" style="display: inline-block; margin: 5px;">  | 사용자 질문 입력 UI, 음성(STT), API 통신 등 웹 인터페이스 구현 |
| **Backend**        |   <img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white" style="display: inline-block; margin: 5px;">         | 챗봇 응답 처리, LangGraph 파이프라인, 세션/채팅 DB 저장 |
| **sLLM 모델**      | <img src="https://img.shields.io/badge/K_intelligence/MiDM_2.0_Base_instruct-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" style="display: inline-block; margin: 5px;">           | 친절한 말투로 튜닝된 경량화 모델 (4bit) |
| **벡터 검색**      | `FAISS`, `intfloat/multilingual-e5-large-instruct`         | PDF 기반 RAG 검색, 유사 문서 검색 |
| **Storage**        | `S3`, `EFS`, `local SSD`                                   | 문서 및 모델 저장, RAG용 벡터 DB, 모델 로딩 |
| **Infrastructure** | `RunPod`, `EC2 (CUDA 환경)`, `Docker`, `ngrok`             | 모델 서빙 환경 (GPU), 배포/테스트 인프라 구성 |
| **로깅 및 추적**   | `wandb`, `logging`, `DB(chatlog / voicelog)`               | 학습 로깅 및 실시간 세션 기록 저장 |

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
