from bert.etri_pos import *

class Etri_Feature(object):
    def __init__(self, raw_sentence=None, morp_sentence=None):
        self.raw_sentence = raw_sentence    #str
        self.morp_sentence = morp_sentence    #list of str 형태
# etri 결과를 구분자 || 기준으로 나눠서 전체 raw input과 형태소 단위 input으로 저장

# 문장의 형태소 분석 결과를 [감기/NNG|는/JK, 있/VV|다/?] 형식의 list of string으로 저장
def return_morp(etri_object):
    etri_morp_list = []
    for eojeol in etri_object['word']:
        etri_morp = ""
        for i in range(int(eojeol['begin']), int(eojeol['end'] + 1)):
            etri_morp += etri_object['morp'][i]['lemma'] + '/' + etri_object['morp'][i]['type'] + '|'
        etri_morp = etri_morp[:-1]  # 맨 마지막 | 제거
        etri_morp_list.append(etri_morp)

        return etri_morp_list

def organize_etri_result(whole_data, etri_result):
    processing_morp = ""
    processing_sent = ""
    etri_idx = 0
    data_idx = 0
    while True:
        if data_idx == len(whole_data) or etri_idx == len(etri_result):
            break
        if not processing_sent:  # 현재 남아있는 현재 처리중인 문장이 없을 경우, 즉 바로 앞까지 정상적으로 배정되었을 경우 새로 추가
            processing_sent = etri_result[etri_idx]['text'] #처리할 문장
        if processing_sent.count("||") == 0:    # 더 이상 ||로 나뉘지 않는 경우, 즉 문장의 시작이 아닐 경우에는 앞에다가 갖다 붙여야함
            pass
if __name__ == '__main__':
    read_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/etri.dev.pkl'
    write_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/etri.dev.conllx'

    with open(read_file_path, 'rb') as f:
        whole_data = pickle.load(f)
        etri_result = pickle.load(f)

    #organize_etri_result(whole_data, etri_result)
    # 전체 데이터의 index 하나를 두고, etri result용 하나를 둔 다음 만약에 etri result가 하나 이상으로 구분해놓았을 경우
    # etri index만 추가시켜서 계속 돌면서 탐색

    whole_data_idx = 0
    etri_idx = 0
    etri_morp_list = []  # 하나의 element가 남/NNG|과/JC 형태로 저장될 list of str
    etri_raw_sentence = ""
    while(True):
        if whole_data_idx == len(whole_data) or etri_idx == len(etri_result):
            break
        etri_object = etri_result[etri_idx]
        whole_data_object = whole_data[whole_data_idx]
        etri_raw_sentence = etri_object['text']    #나중에 원 raw_sentence와 비교하기 위해서 etri 버전의 raw sentence를 계속 저장해둠 ex) 보인다...... => 보인다...  ... 방지

        for eojeol in etri_object['word']:
            etri_morp = ""
            for i in range(int(eojeol['begin']), int(eojeol['end']+1)):
                etri_morp += etri_object['morp'][i]['lemma'] + '/' + etri_object['morp'][i]['type'] + '|'
            etri_morp = etri_morp[:-1]   # 맨 마지막 | 제거
            etri_morp_list.append(etri_morp)

        # 맨 앞 (와 맨 뒤 ) 삭제
        etri_morp_list[0] = etri_morp_list[0][5:]
        etri_morp_list[-1] = etri_morp_list[-1][:-5]

        if not whole_data_object.etri_result:
            whole_data_object.etri_result = etri_morp_list
        else:
            whole_data_object.etri_result += etri_morp_list
        etri_morp_list = []

        # 문장이 1대1로 제대로 parsing됐을 경우 양쪽 다 하나씩 올리고 아닐 경우 etri것만 다음 문장 참고
        # 1. 문장이 ...... => ...  ... 처럼 한 어절이 분리가 되어버리는 경우
        # 2. 어절 분리는 아니지만 어쨋든 sentence가 분리가 되는 경우

        if len(whole_data_object.etri_result) == len(whole_data_object.index):      # etri result로 저장된 결과, 즉 형태소 분석 결과가 전체 index 개수와 같을 경우 1번만 체크
            whole_data_idx += 1
            etri_idx += 1
        # 아직은 모든 문장이 포함되지 않았을 때, 그런데 형태소 단위로 잘렸을지 아닐지를 알 수가 없다.
        else:
            # 원 문장은 정상적으로 들어갔는데 어절 개수가 다를 경우
            if whole_data_object.raw_sentence == etri_object['text'][1:-1]:
                whole_data_idx += 1
                etri_idx += 1
            else:
                whole_data_idx += 1
                etri_idx += 1



    with open(write_file_path, 'w') as fw:
        for data in whole_data:
            for idx in range(len(data.head)):
                fw.write(str(data.index[idx])+'\t')
                fw.write(str(data.etri_result[idx]+'\t'))
                fw.write('_\t_\t')
                pos_set = "+".join([morp.split('/')[-1] for morp in data.etri_result[idx].split('|')])
                fw.write(str(pos_set)+'\t')
                fw.write("_\t")
                fw.write(str(data.head[idx])+'\t')
                fw.write(str(data.phrase[idx])+'\t')
                fw.write('_\t_\n')
            fw.write('\n')