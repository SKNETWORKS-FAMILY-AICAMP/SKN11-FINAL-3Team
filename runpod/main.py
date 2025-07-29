import os
import uuid
import boto3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_upstage import UpstageGroundednessCheck
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import cohere
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from tempfile import TemporaryDirectory
import botocore.exceptions

# 환경변수 설정 (S3, Cohere API 키)
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
AWS_S3_REGION = os.environ["AWS_S3_REGION"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_S3_BUCKET_NAME = os.environ["AWS_S3_BUCKET_NAME"]
s3_client = boto3.client(
    's3',
    region_name=AWS_S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    # endpoint_url='https://s3api-us-ks-2.runpod.io'
)


# 글로벌 변수
sessions = {}
co = cohere.Client(COHERE_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class LoadRequest(BaseModel):
    bot_id: str
    cs_number: str
    email: str
    detail_id: int

class ChatRequest(BaseModel):
    bot_id: str
    session_id: str
    question: str

# 모델 전역 로딩
_tokenizer, _model, _generation_config = None, None, None

def get_midm_model():
    global _tokenizer, _model, _generation_config
    if _model is None:
        hub_model_id = "seoungji/sllm_midm_model"
        print("🔄 모델 로딩 중...")
        _tokenizer = AutoTokenizer.from_pretrained(hub_model_id, trust_remote_code=True, use_fast=False)
        _model = AutoModelForCausalLM.from_pretrained(
            hub_model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        try:
            _generation_config = GenerationConfig.from_pretrained(hub_model_id)
        except:
            _generation_config = GenerationConfig()
        print("✅ 모델 로딩 완료")
    return _tokenizer, _model, _generation_config

# LangGraph 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "질문"]
    answer: Annotated[str, "답변"]
    score: Annotated[float, "유사도 점수"]
    retriever_docs: Annotated[List[Document], "유사도 상위문서"]
    relevance_check: Annotated[str, "문서-답변 관련성 체크"]
    session_id: Annotated[str, "세션 ID"] 
    need_user_input: Annotated[bool, "사용자 재입력 유도 여부"]

# LLM 답변 생성
def generate_with_midm(question: str, context: str, cs_number: str) -> str:
    tokenizer, model, _ = get_midm_model()
    system_prompt = f"진짜 상담원처럼 친절하고 공손한 말투로 답변해주세요. 전화번호는 사용자가 직접 전화를 하고싶다거나, 어디로 연락하면 되냐는 등의 요청이 있을 때만 {cs_number}를 안내해주세요."
    user_prompt = f"문서: {context}\n질문: {question}\n위 문서를 참고해서 질문에 답변해줘."

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        top_p=0.75,
        top_k=20,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("assistant")[-1].strip()


# LangGraph 노드
def retriever_node(state: GraphState) -> GraphState:
    session = sessions[state["session_id"]]
    hybrid_docs = session["retriever"].invoke(state["question"])
    hybrid_docs = [doc for doc in hybrid_docs if doc.page_content]

    rerank_docs = [
        {
            "text": doc.page_content.strip(),
            "metadata": doc.metadata,
            "id": doc.metadata.get("source", f"doc_{i}")
        }
        for i, doc in enumerate(hybrid_docs)
    ]

    rerank_result = co.rerank(
        model="rerank-multilingual-v3.0",
        query=state["question"],
        documents=rerank_docs,
        top_n=3,
        return_documents=True
    )
    reranked_docs = [
        Document(page_content=res.document.text, metadata={**res.document.metadata, "score": res.relevance_score})
        for res in rerank_result.results
    ]
    score = max([doc.metadata.get("score", 0) for doc in reranked_docs], default=0)
    return GraphState(score=score, retriever_docs=reranked_docs, question=state["question"], answer="", relevance_check="", session_id=state["session_id"])

def llm_answer_node(state: GraphState) -> GraphState:
    session = sessions[state["session_id"]]
    context = "\n---\n".join(doc.page_content for doc in state["retriever_docs"])
    answer = generate_with_midm(state["question"], context, session["cs_number"])
    return {**state, "answer": answer, "relevance_check": ""}

def ask_user_again_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": "질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 부탁드리겠습니다!",
        "need_user_input": True
    }

def hallucination_check_node(state: GraphState) -> GraphState:
    groundedness_check = UpstageGroundednessCheck()
    response = groundedness_check.invoke({"context": state["retriever_docs"], "answer": state["answer"]})
    return {**state, "relevance_check": response}

def decide_to_generate(state: GraphState) -> str:
    return "llm_answer" if state["score"] >= 0.02 else "ask_user_again"

def relevance_check(state: GraphState) -> str:
    return END if state["relevance_check"] == "grounded" else END

workflow = StateGraph(GraphState)
workflow.set_entry_point("retriever")
workflow.add_node("retriever", retriever_node)
workflow.add_node("llm_answer", llm_answer_node)
workflow.add_node("hallucination_check", hallucination_check_node)
workflow.add_edge("llm_answer", "hallucination_check")
workflow.add_node("ask_user_again", ask_user_again_node)

workflow.add_conditional_edges("retriever", decide_to_generate, {
    "llm_answer": "llm_answer",
    "ask_user_again": "ask_user_again"
})
workflow.add_conditional_edges("hallucination_check", relevance_check, {END: END})
graph_app = workflow.compile()


# 일단 임시로 세션 아이디 통일, 추후 해결
latest_session_id = None

@app.post("/load")
async def load_model(req: LoadRequest):
    global latest_session_id
    session_id = str(uuid.uuid4())
    latest_session_id = session_id 
    session_dir = f"./sessions/{session_id}"  # 원하는 폴더
    os.makedirs(session_dir, exist_ok=True)   # 폴더 생성

    email_folder = f"bot_{req.email}_{req.detail_id}/vectordb/"
    bucket = AWS_S3_BUCKET_NAME
    
    # 다운로드 파일 경로 정의
    faiss_key = f"{email_folder}index.faiss"
    pkl_key = f"{email_folder}index.pkl"
    local_faiss = os.path.join(session_dir, "index.faiss")
    local_pkl = os.path.join(session_dir, "index.pkl")

    try:
        s3_client.download_file(bucket, faiss_key, local_faiss)
        s3_client.download_file(bucket, pkl_key, local_pkl)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return {"status": "error", "message": f"파일 없음: {e.response['Error']['Message']}"}
        else:
            raise

    # 벡터 DB 불러오기
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    vectorstore = FAISS.load_local(session_dir, embeddings, allow_dangerous_deserialization=True)
    
    
    # 리트리버 구성
    bm25 = BM25Retriever.from_documents(vectorstore.docstore._dict.values())
    faiss = vectorstore.as_retriever(search_kwargs={"k": 5})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=[0.5, 0.5]
    )

    sessions[session_id] = {"retriever": ensemble, "cs_number": req.cs_number}
    
    return {"status": "ready", "session_id": session_id}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    state = GraphState(question=req.question, answer="", score=0.0, retriever_docs=[], relevance_check="", session_id=latest_session_id,need_user_input=False)
    response = graph_app.invoke(state)
    return {
        "answer": response["answer"],
        "need_user_input": response.get("need_user_input", False)
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=7860)
