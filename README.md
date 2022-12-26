## 챗봇 학습 파이프라인
![voice pipeline](https://github.com/Team-Remember/rememberMe-natural_language_train_AI/blob/main/img/nl%20pipeline.png)
- 모델 : Bert
- 문장 생성 모델인 lstm이나 GPT등 다양한 모델을 시도해보았으나, 데이터가 부족하여 완전한 문장을 추론하지 못하므로 Bert 모델을 선택하게 되었습니다.
- 개인별 챗봇 데이터를 추론을 빠르게 하기 위하여 elasticsearch에 챗봇 문자 데이터와 임베딩 데이터를 저장합니다.
- 플랫폼 내에서 사용한 문자 채팅과 즉각복원을 위한 외부 데이터(카카오톡 데이터)를 통해서 학습이 가능합니다.
