from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# 사용자 정보
class UserInfo(BaseModel):
    user_id: int = Field(..., example="123")
    social_id: str = Field(..., example="social_123")
    nickname: Optional[str] = Field(None, example="유경", description="사용자의 닉네임")
    profile_image: Optional[str] = Field(None, example="https://cdn.example.com/profile.jpg")
    provider: str = Field(..., example="kakao", description="소셜 로그인 제공자 (kakao, google, naver)")
    created_at: Optional[datetime] = Field(None, example="2025-07-14T10:00:00Z")
    updated_at: Optional[datetime] = Field(None, example="2025-07-14T10:05:00Z")

# 로그인 응답
class LoginResponse(BaseModel):
    access_token: str = Field(..., example="access_token")
    refresh_token: str = Field(..., example="refresh_token")
    user: UserInfo

# 로그아웃 응답
class LogoutResponse(BaseModel):
    success: bool = Field(..., example=True)
    message: str = Field(..., example="성공적으로 로그아웃되었습니다.")

# 회원 정보 조회 응답
class UserMeResponse(BaseModel):
    user: UserInfo

# 회원 탙퇴 응답  
class UserDeleteResponse(BaseModel):
    success: bool = Field(..., example=True)
    message: str = Field(..., example="회원 탈퇴가 완료되었습니다.")
