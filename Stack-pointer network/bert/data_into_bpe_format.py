from bert.tokenization_kor_bert import BertTokenizer

class BPEFeature(object):
    def __init__(self, original_index, eojeol, pos_set, original_head, phrase):
        self.original_index = original_index
        self.eojeol = eojeol
        self.pos_set = pos_set
        self.original_head = original_head
        self.phrase = phrase
        self.new_tokens = None
        self.new_index = None

if __name__ == '__main__':
    myTokenizer = BertTokenizer.from_pretrained("./", do_lower_case=False)
    #api_key = '6f8cd509-8bf8-46bf-8c48-033400dc037a'
    #text = "엠마누엘 웅가로 /\n의상서 실내 장식품으로…\n디자인 세계 넓혀"
    #result, json_return_obj = do_lang(openapi_key=api_key, text=text)
    #print("1")
    read_file_path = '/home/nlpgpu4/yeongjoon/BERT_Biaffine/data/etri_data/etri.train.conllx'
    write_file_path = '/home/nlpgpu4/yeongjoon/BERT_Biaffine/data/etri_data/bpe.etri.train.conllx'
    fw = open(write_file_path, 'w')
    with open(read_file_path, 'r') as f:
        for sentence in f.read().split('\n\n'):
            if not sentence:
                continue
            FeatureList = []
            index = []
            eojeol = []
            pos_set = []
            head = []
            phrase = []
            for line in sentence.split('\n'):
                i, e, _, _, pos, _, h, p, _, _ = line.split('\t')
                FeatureList.append(BPEFeature(i, e, pos, h, p))

            bpe_dict = {}
            bpe_dict[str(0)] = str(0)
            bpe_idx = 1
            for feature in FeatureList:
                tokens = myTokenizer.tokenize(" ".join(feature.eojeol.split('|')))
                feature.new_tokens = tokens
                for i, token in enumerate(tokens):
                    if i == 0:
                        bpe_dict[feature.original_index] = bpe_idx
                    bpe_idx += 1

            eojeol_idx = 1
            for feature in FeatureList:
                for i, token in enumerate(feature.new_tokens):
                    fw.write(str(eojeol_idx) + '\t')
                    fw.write(str(token) + '\t')
                    fw.write('_' + '\t')
                    fw.write('_' + '\t')
                    if '/' in str(token):
                        fw.write(str(token).split('/')[-1][:-1]+'\t')
                    else:
                        fw.write("X"+'\t')

                    fw.write('_' + '\t')
                    if i == 0:
                        fw.write(str(bpe_dict[feature.original_head]) + '\t')
                        fw.write(feature.phrase)
                    else:
                        fw.write('0'+'\t')
                        fw.write('X'+'\t')
                    fw.write('_' + '\t')
                    fw.write('_' + '\n')
                    eojeol_idx += 1

            fw.write('\n')




