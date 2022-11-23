import asyncio
from logging import getLogger
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from typing import List
import time, os
from preprocess import open_and_preprocess_kakao_file, make_model_input_form, preprocess_db_data
from train import embedding_csv, insert_chatdata_es

logging = getLogger(__name__)
app = FastAPI()


# uvicorn app.main:app --reload --host=0.0.0.0 --port=8002
# 챗봇 카카오톡 데이터 입력시 학습시키기
@app.post("/chat_bot_train_kakao")
async def chatbot_train(background_tasks: BackgroundTasks, memberId: str, weId: str,
                        files: List[UploadFile] = File(...)):
    background_tasks.add_task(chatbot_train_kakao_after_return, memberId, weId, files)
    return {"message": "success!"}


def chatbot_train_kakao_after_return(memberId, weId, files):
    start = time.time()
    # 카카오톡 파일 전처리
    my_katalk_df = open_and_preprocess_kakao_file(files)

    # input 데이터프레임으로 변형
    result_dataframe = make_model_input_form(my_katalk_df)

    # 임베딩
    embedding_result_csv_name = embedding_csv(result_dataframe, memberId, weId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, weId)
    insert_chatdata_es(embedding_result_csv_name, weId, memberId)
    result = time.time()
    print('kakao_training 시간', result - start)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)


# 챗봇 데이터 베이스 데이터 입력시 학습시키기
@app.post("/chat_bot_train_db")
async def chatbot_database_train(background_tasks: BackgroundTasks, request: Request):
    request_list = await request.json()

    background_tasks.add_task(chatbot_train_db_after_return, request_list)

    return {"message": "success!"}


def chatbot_train_db_after_return(request_list):

    memberId = request_list[0]['memberId']
    weId = request_list[0]['opponentId']
    print(memberId, weId)

    start = time.time()
    # format
    embedding_result_df = preprocess_db_data(request_list)

    # 임베딩
    embedding_result_csv_name = embedding_csv(embedding_result_df, memberId, weId)

    # es 데이터 insert
    insert_chatdata_es(embedding_result_csv_name, memberId, weId)
    insert_chatdata_es(embedding_result_csv_name, weId, memberId)

    # es 데이터 insert 후 csv 삭제
    os.remove(embedding_result_csv_name)
    result = time.time()
    print('db_training 시간', result - start)

