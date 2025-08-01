# !pip list
# !pip install -qU langchain-core langchain-upstage

from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import cohere
from langchain_upstage import UpstageGroundednessCheck # langchain_upstage==0.1.3
import getpass

# 환경 변수 설정
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")  

# FAISS 벡터 DB 불러오기
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vectorstore = FAISS.load_local("card_QA_faiss_db", embedding_model,allow_dangerous_deserialization=True)

# os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
COHERE_API_KEY = os.environ["COHERE_API_KEY"]
# print("Cohere API Key:", COHERE_API_KEY)  
co = cohere.Client(COHERE_API_KEY)

# # os.getenv("UPSTAGE_API_KEY")
# os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Upstage API Key:")
# Upstage 키
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
if not UPSTAGE_API_KEY:
    raise RuntimeError("`.env`에 UPSTAGE_API_KEY가 설정되어 있지 않습니다.")
# 필요하다면 환경변수에도 등록
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# 전체 문서 (BM25용)
all_docs = vectorstore.docstore._dict.values()

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 5

# faiss retriever
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Hybrid (ensemble) retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.5, 0.5]
)

# 1. State 정의
class GraphState(TypedDict):
    question: Annotated[str, "질문"]
    answer: Annotated[str, "답변"]
    score: Annotated[float, "유사도 점수"]
    retriever_docs: Annotated[List[Document], "유사도 상위문서"]
    relevance_check: Annotated[str, "문서-답변 관련성 체크"]

# 2. 노드 정의

def user_input_node(state: GraphState) -> GraphState:
    """
    사용자로부터 질문을 입력받는 노드.
    """
    question = input()
    return GraphState(question=question, answer="", score=0.0, retriever_docs=[])

def retriever_node(state: GraphState) -> GraphState:
    hybrid_docs = ensemble_retriever.invoke(state["question"])
    # print("앙상블리트리버에서 invoke 로 가져온 결과\n",hybrid_docs)
    hybrid_docs = [doc for doc in hybrid_docs if doc.page_content]  # 빈 문서 제거
 
    # Cohere Rerank용 문서 구성
    rerank_docs = [
        {
            "text": doc.page_content.strip(),
            "metadata": doc.metadata,
            "id": doc.metadata.get("source", f"doc_{i}")
        }
        for i, doc in enumerate(hybrid_docs)
        if doc.page_content.strip()
    ]

    rerank_result = co.rerank(
        model="rerank-multilingual-v3.0",
        query=state["question"],
        documents=rerank_docs,
        top_n=3,
        return_documents=True
    )

    reranked_docs = [
        Document(
            page_content=res.document.text,
            metadata={**res.document.metadata, "score": res.relevance_score}
        )
        for res in rerank_result.results
    ]

    score = max([doc.metadata.get("score", 0) for doc in reranked_docs], default=0)
    print("max_score:", score)
    print("-"*10000)
    print("[retriever_node] reranked_docs", reranked_docs)

    return GraphState(score=score, retriever_docs=reranked_docs)

# def grade_documents_node(state: GraphState) -> GraphState:
#     return GraphState()

def llm_answer_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_template(
        """
        문서: {docs}
        질문: {question}
        위 문서들을 참고해서 질문에 답변해줘.
        진짜 상담원 처럼 답변해주고, 친절한 말투로 일관되게 답변해줘.
        그리고 직접 전화를 하고 싶다거나, 직접 상담을 원하거나, 문의사항이 있다고 직접적으로 전화번호를 요구할 경우에만 010-1111-2345 라는 전화번호를 안내해줘.
        *중요한 것 : 전화,상담,문의,전화번호 등 꼭 직접적으로 사용자가 요청하는 경우에만 알려줘*
        """
    )
    docs_content = "\n---\n".join([doc.page_content for doc in state["retriever_docs"]])
    chain = prompt | ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
    answer = chain.invoke({"docs": docs_content, "question": state["question"]}).content
    print("\n[llm_answer_node] 생성된 답변:", answer)
    return GraphState(answer=answer)

def hallucination_check_node(state: GraphState) -> GraphState:
    """
    생성된 답변과 문서의 관련성을 확인하고 2가지로 분기:
    1. 관련성 높은 경우 - llm_answer로 이동
    2. 관련성이 불확실한 경우 - user_input으로 이동
    3. 관련성이 매우 낮은 경우 - user_input으로 이동
    """
    groundedness_check = UpstageGroundednessCheck()
 
    request_input = {
        "context": state["retriever_docs"],
        "answer": state["answer"],
    }
    response = groundedness_check.invoke(request_input)

    print("\n[hallucination_check_node] 신뢰성 확인:", response)
    return GraphState(relevance_check=response)

# def query_rewrite_node(state: GraphState) -> GraphState:
#     prompt = ChatPromptTemplate.from_template(
#         """
#         원본 질문: {question}
#         위 질문의 핵심은 유지하면서, 유사 문서를 더 잘 찾을 수 있도록 질문을 다시 써줘.
#         """
#     )
#     chain = prompt | ChatOpenAI(model="gpt-4.1-mini-2025-04-14")
#     new_question = chain.invoke({"question": state["question"]}).content
#     print("\n[query_rewrite_node] :", new_question)
#     return GraphState(question=new_question)

# 3. 노드 분기 함수 정의
def decide_to_generate(state: GraphState) -> str:
    """
    질문-문서의 유사도 점수에 따라 다음 노드를 결정 
    - score가 0.02 이상이면 'llm_answer'로 이동
    - score가 0.02 미만이면 'query_rewrite'로 이동
    """
    if state["score"] >= 0.02:
        return "llm_answer"
    else:
        print("해당 질문에 대해서는 안내해드리기 어렵습니다! 다른 질문을 부탁드리겠습니다.")
        return "user_input"

def relevance_check(state: GraphState) -> str:
    """
    문서-답변의 관련성에 따라 다음 노드를 결정
    - grounded 이면 'llm_answer'로 이동
    - not sure, not grounded 이면 'user_input'로 이동
    """
    if state["relevance_check"] == 'grounded':
        return END
    else:
        print("해당 질문에 대해서는 안내해드리기 어렵습니다! 다른 질문을 부탁드리겠습니다.")
        return "user_input"


# 4. LangGraph 구성 및 연결
workflow = StateGraph(GraphState)
workflow.set_entry_point("user_input")
workflow.add_node("user_input", user_input_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("llm_answer", llm_answer_node)
workflow.add_node("hallucination_check", hallucination_check_node)
# workflow.add_node("query_rewrite", query_rewrite_node)

workflow.add_edge("user_input", "retriever")
workflow.add_conditional_edges(
    "retriever",
    decide_to_generate,
    {
        "user_input": "user_input",
        "llm_answer": "llm_answer",
    },
)
workflow.add_edge("llm_answer","hallucination_check")
workflow.add_conditional_edges(
    "hallucination_check",
    relevance_check,
    {
        END: END,
        "user_input": "user_input"
    }
)

# 실행 
app = workflow.compile()
response = app.invoke({})
print("\n[최종 답변]:", response["answer"])
