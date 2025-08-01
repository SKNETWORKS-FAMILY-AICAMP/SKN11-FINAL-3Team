from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from uuid import uuid4
from datetime import datetime
import httpx
from models.session import Session as SessionModel
from models.chatlog import ChatLog
from schemas.request.session import CreateSessionRequest, SendMessageRequest
from schemas.response.session import (
    CreateSessionResponse, SendMessageResponse,
    MessageItemResponse, MessageListResponse, EndSessionResponse
)
from services.session.utils import get_next_turn
from services.session.runpod import start_runpod_pod


client = httpx.AsyncClient()

# 1. 세션 생성
async def create_session_service(bot_id: str, db: AsyncSession) -> CreateSessionResponse:
    print('>>>> 세션 생성 함수 진입 ')
    while True:
        session_id = f"session_{uuid4().hex[:8]}"
        result = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
        existing = result.scalar_one_or_none()
        if not existing:
            break
    
    print(session_id)
    print('>>>> SessionModel 저장 시작 ')

    session = SessionModel(
        session_id=session_id,
        bot_id=bot_id
    )
    db.add(session)
    await db.commit()

    print('>>>> SessionModel 저장 완료 ')

    try:
        start_runpod_pod()
        print("✅ RunPod Pod 실행 완료")
    except Exception as e:
        print(f"❌ RunPod Pod 실행 실패: {e}")

    return CreateSessionResponse(session_id=session_id)

# 2. 메시지 전송 (텍스트)
async def send_message_service(session_id: str, request: SendMessageRequest, db: AsyncSession) -> SendMessageResponse:
    result = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    next_turn = get_next_turn(session_id, sender="user")

    user_msg = ChatLog(
        session_id=session_id,
        turn=next_turn,
        role="user",
        content=request.message,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(user_msg)
    await db.commit()
    await db.refresh(user_msg)

    # SLLM 호출 (실제 답변 받아오기)
    assistant_reply = await req_sllm_service(session_id, request.message, next_turn)

    bot_response = MessageItemResponse(
        sender="bot",
        message=assistant_reply.content,
        timestamp=datetime.utcnow(),
        message_type="text"
    )

    return SendMessageResponse(session_id=session_id, message=bot_response)


# 3. 세션 기록 전체 조회
async def get_session_messages_service(session_id: str, db: AsyncSession) -> MessageListResponse:
    result = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(select(ChatLog).where(ChatLog.session_id == session_id).order_by(ChatLog.turn))
    chatlogs = result.scalars().all()

    messages = [
        MessageItemResponse(
            sender=log.role,
            message=log.content,
            timestamp=log.created_at,
            message_type="text"
        ) for log in chatlogs
    ]

    return MessageListResponse(session_id=session_id, messages=messages)


# 4. 세션 종료
async def end_session_service(session_id: str, db: AsyncSession) -> EndSessionResponse:
    result = await db.execute(select(SessionModel).where(SessionModel.session_id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()

    return EndSessionResponse(success=True, message="세션이 성공적으로 종료되었습니다.")

# 5. SLLM 요청
async def req_sllm_service(session_id: str, message: str, turn: int):
    resp = await client.post(
        f"http://localhost:3000/{session_id}/sllm",
        json={
            "session_id": session_id,
            "content": message,
            "turn": turn
        }
    )
    resp.raise_for_status()
    result = resp.json()
    answer = result.get("content", "")
    return answer
