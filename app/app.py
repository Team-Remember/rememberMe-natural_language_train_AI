from fastapi import FastAPI
from logging import getLogger
from app.configurations import APIConfigurations
from app.routers import router

logger = getLogger(__name__)

app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(router, prefix="", tags=[''])

# 서버 실행시
# uvicorn app.app:app --reload --host=0.0.0.0 --port=8001
# http://127.0.0.1:8001/docs#/
# remember 로 가상환경 진행중.
