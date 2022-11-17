from logging import getLogger
from fastapi import APIRouter, File, UploadFile, Request
from typing import List
import time, os

from ml.preprocess import open_and_preprocess_kakao_file, make_model_input_form, preprocess_db_data
from ml.train import embedding_csv, insert_chatdata_es

logging = getLogger(__name__)
router = APIRouter()


# 챗봇 카카오톡 데이터 입력시 학습시키기
@router.post("/chat_bot_train_kakao")
async def chatbot_train(memberId: str, weId: str, files: List[UploadFile] = File(...)):
    start = time.time()

    # 카카오톡 파일 전처리
    my_katalk_df = open_and_preprocess_kakao_file(files)

    # input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result_csv_name = embedding_csv(result_dataframe, memberId, weId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, weId)
    result = time.time()
    print('training 시간', result - start)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)
    return {"message": "success!"}


# 챗봇 데이터 베이스 데이터 입력시 학습시키기
@router.post("/chat_bot_train_db")
async def chatbot_database_train(request: Request):
    request_list = await request.json()

    # 전처리
    memberId = request_list[0]['memberId']
    opponentId = request_list[0]['opponentId']
    print(memberId, opponentId)

    # format
    embedding_result_df = preprocess_db_data(request_list)

    # 임베딩
    embedding_result_csv_name = embedding_csv(embedding_result_df, memberId, opponentId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, opponentId)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)
    return "성공!"