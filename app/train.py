from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

from datetime import datetime
import numpy as np
import pandas as pd
import ssl, certifi, time

## ssl
context = ssl.create_default_context(cafile=certifi.where())
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 지수로그 없애기 위해 소수점 6자리까지만
np.set_printoptions(precision=6, suppress=True)

url = 'https://34.64.69.252:9200/'
# url = 'http://localhost:9200/'


def embedding_csv(dataframe, member_id, we_id):
    embeding_list = []
    for temp in tqdm(dataframe['Q']):
        embed_temp = model.encode(temp)
        embeding_list.append(embed_temp)

    # df_embeding = pd.DataFrame(embeding_list)
    dataframe['chatvector'] = embeding_list
    dataframe.to_csv(f'embeding_result_{member_id}_{we_id}.csv', index=None)
    return f'embeding_result_{member_id}_{we_id}.csv'


def insert_chatdata_es(embedding_result_csv_name, member_id, we_id):
    es = Elasticsearch(hosts=[url], http_auth=('elastic', 'uYaVMCVhxfZe6IPAA7zT'), ssl_context=context, request_timeout=30, verify_certs=False)
    # es = Elasticsearch(hosts=[url], basic_auth=('elastic', 'rlagksgh'))
    df = pd.read_csv(embedding_result_csv_name)
    index = "chat_bot"
    count = 0
    for temp1, temp, temp2 in zip(df['A'], df['Q'], df['chatvector']):
        # chatvector 에 값을 넣기 위해서 str > replace > list > float 으로 변환.
        list_of_string = temp2.replace('[', '').replace(']', '').split()[0:512]
        list_of_float = list(map(float, list_of_string))
        doc = {
            "member_id": member_id,
            "we_id": we_id,
            "Q": temp,
            "A": temp1,
            "chatvector": list_of_float,
            "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')
        }

        count = count + 1

        es.index(index=index, body=doc)
        # 1000개당 10초 쉬기.
        if count % 1000 == 0:
            time.sleep(10)
