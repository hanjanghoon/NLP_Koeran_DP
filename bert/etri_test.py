from bert.etri_pos import *

read_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/modified_sejong.ppos2.train.txt'
#write_file_path = '/home/nlpgpu4/yeongjoon/SPEMoBERT/data/etri_data/modified_sejong.ppos2.train.txt'
#fw = open(write_file_path, 'w')
with open(read_file_path, 'r') as f:
    for sentence in f.read().split('\n\n'):
        if not sentence:
            continue
        phrase_count = 0
        for line in sentence.split('\n'):
            if line[0] == ';':
                #result_line = line.strip().replace("\'", " \' ").replace('\"', ' \" ').replace('`',' ` ').replace("  "," ")+'\n'
                result_line = line.strip()+'\n'
                eojeol_count = len(result_line.split(" "))-1
            else:
                phrase_count += 1
                result_line = line.strip()+'\n'
            #fw.write(result_line)
        if eojeol_count != phrase_count:
            print(1)
        #fw.write('\n')