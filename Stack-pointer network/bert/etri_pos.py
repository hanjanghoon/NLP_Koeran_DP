from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

import urllib3
import json

class SentenceFeature(object):
    def __init__(self, raw_sentence, index=None, head=None, phrase=None, morp=None):
        self.raw_sentence = raw_sentence    # 1993/06/08 19 등과 같은 raw sentence
        #self.length = None  # 한 문장을 구성하는 어절의 개수
        self.index = index  # 한 문장 내에qs서 각 어절이 몇번째 어절인지를 저장
        self.head = head    # 한 문장 내에서 각 어절의 head의 index를 가르킴
        self.phrase = phrase  # 한 문장 내에서 해당 어절의 phrase(NP, VP, 등)
        self.morp = morp    # 한 문장 내에서 해당 어절의 기존 형태소 분석 결과가 저장 지금은 안쓸듯??
        self.etri_result = None # 한 문장을 형태소 분석한 전체 결과가 dictionary로 저장

def do_lang(openapi_key, text):
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

    requestJson = {"access_key": openapi_key, "argument": {"text": text, "analysis_code": "morp"}}

    http = urllib3.PoolManager()
    response = http.request("POST", openApiURL, headers={"Content-Type": "application/json; charset=UTF-8"},
                            body=json.dumps(requestJson))

    json_data = json.loads(response.data.decode('utf-8'))
    json_result = json_data["result"]

    if json_result == -1:
        json_reason = json_data["reason"]
        if "Invalid Access Key" in json_reason:
            logger.info(json_reason)
            logger.info("Please check the openapi access key.")
            sys.exit()
        return "openapi error - " + json_reason
    else:
        json_data = json.loads(response.data.decode('utf-8'))

        json_return_obj = json_data["return_object"]

        return_result = ""
        json_sentence = json_return_obj["sentence"]
        return json_sentence
        # for json_morp in json_sentence:
        #     for morp in json_morp["morp"]:
        #         return_result = return_result + str(morp["lemma"]) + "/" + str(morp["type"]) + " "
        #
        # return return_result, json_return_obj

if __name__ == '__main__':
    api_key = '6f8cd509-8bf8-46bf-8c48-033400dc037a'
    #text = "엠마누엘 웅가로 /\n의상서 실내 장식품으로…\n디자인 세계 넓혀"
    #result, json_return_obj = do_lang(openapi_key=api_key, text=text)
    #print("1")
    read_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/modified_sejong.ppos2.train.txt'
    write_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/etri.train.pkl'
    whole_data = []
    with open(read_file_path, 'r') as f:
        for sentence in f.read().split('\n\n'):
            if not sentence:
                continue
            index = []
            head = []
            phrase = []
            morp = []
            for line in sentence.split('\n'):
                if line[0] == ';':
                    raw_sentence = line.split(';', 1)[1].strip()
                    #raw_sentence = line.strip()
                else:
                    i, h, p, m = line.split('\t')
                    index.append(i.strip())
                    head.append(h.strip())
                    phrase.append(p.strip())
                    morp.append(m.strip())
            whole_data.append(SentenceFeature(raw_sentence=raw_sentence, index=index, head=head, phrase=phrase, morp=morp))

    input_for_etri_pos = ""
    len_max = 9500
    #sent_count = 0
    #index_for_whole_data = 0
    etri_result = []
    for i, data in enumerate(whole_data):
        #input_for_etri_pos += ' | '+data.raw_sentence
        input_for_etri_pos += '(' + data.raw_sentence +')\n'
        #sent_count += 1
        if len(input_for_etri_pos) < len_max:
            continue
        else:
            json_sentences = do_lang(api_key, input_for_etri_pos)
            #assert sent_count == len(json_sentences)
            # for etri_result in json_sentences:
            #     whole_data[index_for_whole_data].etri_result = etri_result
            #     index_for_whole_data += 1
            # sent_count = 0
            etri_result += json_sentences
            input_for_etri_pos = ""
    # 맨 마지막에 남은 문장들도 전부 추가
    if input_for_etri_pos:
        json_sentences = do_lang(api_key, input_for_etri_pos)
        #assert sent_count == len(json_sentences)
        # for etri_result in json_sentences:
        #     whole_data[index_for_whole_data].etri_result = etri_result
        #     index_for_whole_data += 1
        etri_result += json_sentences

    with open(write_file_path, 'wb') as f:
        pickle.dump(whole_data, f)
        pickle.dump(etri_result, f)