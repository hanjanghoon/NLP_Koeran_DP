from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np
import random

#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#from tqdm import tqdm

#from pytorch_transformers import BertModel, BertConfig
#from src_tokenizer.tokenization_morp import BertTokenizer
#from src_examples.run_classifier_morp import *
#import argparse
#import glob
#import logging

#logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class BertForEmbedding(nn.Module):
    def __init__(self, bert):
        super(BertForEmbedding, self).__init__()
        self.bert = bert

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        last_hidden, cls_token, output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)
        #seojun : bert 모델의 12개의 encoder layer 중 마지막 4개의 layer의 합을 embedding 값으로 리턴한다.
        embedding = output[-1] + output[-2] + output[-3] + output[-4]
        return last_hidden

class BertForEncoder(nn.Module):
    def __init__(self, bert):
        super(BertForEncoder, self).__init__()
        self.bert = bert

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        last_hidden, cls_token, output = self.bert(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask, position_ids=position_ids,
                                                   head_mask=head_mask)
        #seojun : bert 모델의
        return last_hidden
# bert에 적절한 형태로 input을 만들어준다.
# 이 때 필요한건 original sentence와 tokenizer
# output으로는 bert에 input으로 넣을 알맞은 꼴, 어절 단위 문장, 어절 각각 몇 개의 morp로 이루어져 있는지 길이
# morp 단위 문장, morp 각각이 몇 word piece인지

def make_bert_input(original_sentences, tokenizer):
    each_eojeol_lengths = []
    each_morp_lengths = []
    bert_inputs = []
    for original_sentence in original_sentences:
        ##### bert 자르는거 testing 중 #####
        morp_tokenized_sent = []
        each_eojeol_length = []
        each_morp_length = []
        for word in original_sentence:
            each_eojeol_length.append(len(word))
            for morp in word:
                morp_tokenized_sent.append(morp)
        for morp in morp_tokenized_sent:
            if morp == '_ROOT': # root 추가
                each_morp_length.append(1)
                continue
            bert_token = tokenizer.tokenize(morp)
            each_morp_length.append(len(bert_token))
        # morp_tokenized_sentences.append(morp_tokenized_sent)
        each_eojeol_lengths.append(each_eojeol_length)
        each_morp_lengths.append(each_morp_length)
        bert_input = tokenizer.tokenize(" ".join(morp_tokenized_sent[1:]))
        #why = " ".join(morp_tokenized_sent)
        bert_inputs.append(["_ROOT"]+bert_input)

    return bert_inputs, each_morp_lengths, each_eojeol_lengths

def resize_bert_output(bert_output, each_morp_lengths, each_eojeol_lengths, max_eojeol_length=20, output_dim=2304, use_first_token=True, bert_word_feature_ids=None, bert_morp_feature_ids=None):
    ## @TODO 우선 bert 제대로 들어가는 지확인해보는 용으로 다르게 짜봄
    # eojeol_vectors_tensor = torch.zeros([bert_output.size(0), max_eojeol_length, output_dim], dtype=torch.float32).cuda()
    # batch_size, max_seq_len, _ = bert_output.size()
    # for i in range(batch_size):
    #     jj = -1
    #     for j in range(max_seq_len):
    #         if bert_word_feature_ids[i][j] == 1 and bert_morp_feature_ids[i][j] == 3:
    #             jj += 1
    #             eojeol_vectors_tensor[i][jj] = bert_output[i][j]
    #
    # return _, eojeol_vectors_tensor
    morp_vectors_list = []
    padding_start_idx = []
    # 우선은 각 BPE token => SUM
    for i, each_morp_length in enumerate(each_morp_lengths):
        start_idx = 0  # 맨 앞의 [CLS] 를 root로 대체 embedding 구성
        morp_vectors = []
        for morp_len in each_morp_length:
            end_idx = start_idx + morp_len
            morp_vec_flag = False
            ############ 1. start_idx부터 end_idx까지 모두 더하기
            # for vec in bert_output[i][start_idx:end_idx]:
            #     if morp_vec_flag:
            #         morp_vec += vec
            #     else:
            #         morp_vec = vec
            #         morp_vec_flag = True

            ############ 2. start_idx부터 end_idx까지 mean pooling 사용 아니면 맨 앞 token만 사용
            if use_first_token:
                morp_vec = bert_output[i][start_idx]
            else:
                morp_vec = torch.mean(bert_output[i][start_idx:end_idx], 0)

            start_idx = end_idx
            morp_vectors.append(morp_vec)
        padding_start_idx.append(end_idx)
        morp_vectors_list.append(morp_vectors)

    eojeol_vectors_tensor = torch.zeros([bert_output.size(0), max_eojeol_length, output_dim],
                                        dtype=torch.float32).cuda()
    # eojeol_vectors_list = []
    for i, each_eojeol_length in enumerate(each_eojeol_lengths):
        start_idx = 0
        # 어절 단위일 때는 맨 앞에꺼 + 맨 뒤에꺼
        for j, eojeol_len in enumerate(each_eojeol_length):
            end_idx = start_idx + eojeol_len
            ############ 0. 어절 내의 형태소 중 맨 앞에 있는것만 쓰는 방법
            # eojeol_vec = morp_vectors_list[i][start_idx]
            ############ 1. 어절 내의 각 형태소에 대해서는 맨 처음과 맨 끝만 concat 하는 방법
            eojeol_vec = torch.cat((morp_vectors_list[i][start_idx], morp_vectors_list[i][end_idx - 1],morp_vectors_list[i][end_idx - 1]), dim=-1)

            # morp_vector_tensor = torch.stack([morp_vectors_list[i][start_idx:end_idx]])
            ############ 2. 어절 내의 각 형태소에 대해서 모든 형태소를 sum 하여 사용
            # eojeol_vec_flag = False
            # for k in range(start_idx, end_idx):
            #     if eojeol_vec_flag:
            #         eojeol_vec += morp_vectors_list[i][k]
            #     else:
            #         eojeol_vec = morp_vectors_list[i][k]
            #         eojeol_vec_flag = True

            start_idx = end_idx
            eojeol_vectors_tensor[i, j] = eojeol_vec
        # 나머지 어절 뒷 부분을 최대 어절 길이 만큼 bert padding 값을 채운다.
        # TODO 최대 어절 길이 - 각 어절 길이 만큼 패딩을 bert output에서 끌어다가 채운다. 됐을까?
        # eojeol_vectors_tensor[i, len(each_eojeol_length):] += torch.zeros([max_eojeol_length-len(each_eojeol_length), bert_dim]).cuda()

    return morp_vectors_list, eojeol_vectors_tensor

def convert_sentence_into_features(bert_inputs, tokenizer, max_seq_length):
    features = []
    for bert_input in bert_inputs:
        without_root_bert_input = bert_input[1:]
        if len(without_root_bert_input) > max_seq_length - 2:
            without_root_bert_input = without_root_bert_input[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + without_root_bert_input + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        zeros = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += zeros
        segment_ids += zeros

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

    return features

class BertFeatures(object):
    def __init__(self, word=None, morp=None):
        self.word = word
        self.morp = morp
# max_seq_length는 bert의 input의 max_seq_length
def convert_into_bert_feature_indices(each_eojeol_lengths, each_morp_lengths, max_seq_length):
    bert_feature_indices = []
    word_feature_to_idx ={"B-word": 1, "I-word":2}
    morp_feature_to_idx = {"B-morp": 1, "I-morp": 2}
    for each_eojeol_length, each_morp_length in zip(each_eojeol_lengths, each_morp_lengths):
        word_feature_index = []
        morp_feature_index = []
        token_idx = 0
        for eojeol_length in each_eojeol_length:
            for i in range(eojeol_length):
                for j in range(each_morp_length[token_idx]):
                    if i == 0 and j == 0:
                        word_feature_index.append(word_feature_to_idx["B-word"])
                        morp_feature_index.append(morp_feature_to_idx["B-morp"]) # 어절의 시작이자 형태소의 시작
                    elif i == 0 and j != 0:
                        word_feature_index.append(word_feature_to_idx["B-word"])
                        morp_feature_index.append(morp_feature_to_idx["I-morp"])
                        #어절의 시작이지만 형태소의 시작은 아님
                    elif i != 0 and j == 0:
                        word_feature_index.append(word_feature_to_idx["I-word"])
                        morp_feature_index.append(morp_feature_to_idx["B-morp"])
                        #어절의 시작은 아니지만 형태소의 시작
                    else:
                        word_feature_index.append(word_feature_to_idx["I-word"])
                        morp_feature_index.append(morp_feature_to_idx["I-morp"])
                        # 어절의 시작도 아니고 형태소의 시작도 아님
                token_idx += 1

        #assert len(token_idx) == len(bert_feature_index)
        # 맨 마지막 token들에 각각 패딩 자질 추가??
        word_feature_index += [0] * (max_seq_length - len(word_feature_index))  # 뒤엔 padding
        morp_feature_index += [0] * (max_seq_length - len(morp_feature_index))  # 뒤엔 padding
        bert_feature_indices.append(BertFeatures(word_feature_index, morp_feature_index))

    # word_feature_ids = torch.tensor([f.word for f in bert_feature_indices], dtype=torch.long).cuda()
    # morp_feature_ids = torch.tensor([f.morp for f in bert_feature_indices], dtype=torch.long).cuda()

    return bert_feature_indices

# def main():
#     parser = argparse.ArgumentParser()
#
#     ## Required parameters
#     parser.add_argument("--openapi_key", default=None, type=str, required=True,
#                         help="The openapi accessKey. Please go to this site(http://aiopen.etri.re.kr/key_main.php).")
#
#     parser.add_argument("--data_dir",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
#     parser.add_argument("--task_name", default=None, type=str, required=True,
#                         help="The name of the task to train.")
#     parser.add_argument("--output_dir",
#                         default=None,
#                         type=str,
#                         required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")
#
#     ## Other parameters
#     parser.add_argument("--config_name", default="", type=str,
#                         help="Pretrained config name or path if not the same as model_name")
#     parser.add_argument("--tokenizer_name", default="", type=str,
#                         help="Pretrained tokenizer name or path if not the same as model_name")
#     parser.add_argument("--cache_dir", default="", type=str,
#                         help="Where do you want to store the pre-trained models downloaded from s3")
#     parser.add_argument("--max_seq_length", default=512, type=int,
#                         help="The maximum total input sequence length after tokenization. Sequences longer "
#                              "than this will be truncated, sequences shorter will be padded.")
#     parser.add_argument("--do_train", action='store_true',
#                         help="Whether to run training.")
#     parser.add_argument("--do_eval", action='store_true',
#                         help="Whether to run eval on the dev set.")
#     parser.add_argument("--evaluate_during_training", action='store_true',
#                         help="Rul evaluation during training at each logging step.")
#     parser.add_argument("--do_lower_case", action='store_true',
#                         help="Set this flag if you are using an uncased model.")
#
#     parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
#                         help="Batch size per GPU/CPU for training.")
#     parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
#                         help="Batch size per GPU/CPU for evaluation.")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--learning_rate", default=5e-5, type=float,
#                         help="The initial learning rate for Adam.")
#     parser.add_argument("--weight_decay", default=0.0, type=float,
#                         help="Weight deay if we apply some.")
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
#     parser.add_argument("--num_train_epochs", default=3.0, type=float,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
#     parser.add_argument("--warmup_steps", default=0, type=int,
#                         help="Linear warmup over warmup_steps.")
#
#     parser.add_argument('--logging_steps', type=int, default=50,
#                         help="Log every X updates steps.")
#     parser.add_argument('--save_steps', type=int, default=50,
#                         help="Save checkpoint every X updates steps.")
#     parser.add_argument("--eval_all_checkpoints", action='store_true',
#                         help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
#     parser.add_argument("--no_cuda", action='store_true',
#                         help="Avoid using CUDA when available")
#     parser.add_argument('--overwrite_output_dir', action='store_true',
#                         help="Overwrite the content of the output directory")
#     parser.add_argument('--overwrite_cache', action='store_true',
#                         help="Overwrite the cached training and evaluation sets")
#     parser.add_argument('--seed', type=int, default=42,
#                         help="random seed for initialization")
#
#     parser.add_argument('--fp16', action='store_true',
#                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
#     parser.add_argument('--fp16_opt_level', type=str, default='O1',
#                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
#                              "See details at https://nvidia.github.io/apex/amp.html")
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="For distributed training: local_rank")
#     parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
#     parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
#     args = parser.parse_args()
#
#
#     print("Hello")
#     if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
#         raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
#
#     # Setup distant debugging if needed
#     if args.server_ip and args.server_port:
#         # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
#         import ptvsd
#         print("Waiting for debugger attach")
#         ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
#         ptvsd.wait_for_attach()
#
#     # Setup CUDA, GPU & distributed training
#     if args.local_rank == -1 or args.no_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#         args.n_gpu = torch.cuda.device_count()
#     else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         torch.distributed.init_process_group(backend='nccl')
#         args.n_gpu = 1
#     args.device = device
#
#     # Setup logging
#     logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                         datefmt = '%m/%d/%Y %H:%M:%S',
#                         level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
#     logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
#                     args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
#
#     # Set seed
#     set_seed(args)
#
#     processors = {
#         "cola": ColaProcessor,
#         "mrpc": MrpcProcessor,
#     }
#
#     num_labels_task = {
#         "cola": 2,
#         "mrpc": 2,
#     }
#
#     if args.local_rank == -1 or args.no_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#         n_gpu = torch.cuda.device_count()
#     else:
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         n_gpu = 1
#         # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#         torch.distributed.init_process_group(backend='nccl')
#
#     config = BertConfig.from_pretrained("/home/nlpgpu4/yeongjoon/kor_bert/newversion_bert_morp_pytorch/config.json")
#     #요거 필수 추가
#     config.output_hidden_states=True
#     tokenizer = BertTokenizer.from_pretrained("/home/nlpgpu4/yeongjoon/kor_bert/newversion_bert_morp_pytorch/vocab.txt", do_lower_case=False)
#     model = BertModel.from_pretrained("/home/nlpgpu4/yeongjoon/kor_bert/newversion_bert_morp_pytorch/pytorch_model_backup.bin", config=config)
#
#     args.train_batch_size = args.per_gpu_train_batch_size // args.gradient_accumulation_steps
#
#     task_name = args.task_name.lower()
#
#     if task_name not in processors:
#         raise ValueError("Task not found: %s" % (task_name))
#
#     processor = processors[task_name]()
#     if task_name == 'cola':
#         label_list = processor.get_labels(args.data_dir)
#         num_labels = 2
#         #num_labels = len(label_list)
#
#     train_examples = processor.get_train_examples(args.data_dir)
#     num_train_optimization_steps = int(
#         len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
#     if args.local_rank != -1:
#         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
#
#     # model.to(device)
#     # if args.local_rank != -1:
#     #     try:
#     #         from apex.parallel import DistributedDataParallel as DDP
#     #     except ImportError:
#     #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
#     #
#     #     model = DDP(model)
#     # elif n_gpu > 1:
#     #     model = torch.nn.DataParallel(model)
#
#     global_step = 0
#     nb_tr_steps = 0
#     tr_loss = 0
#
#     original_sentences = [[["그러/VV","던/ETM"], ["어느/MM"], ["날/NNG"], ["나/NP", "는/JX"], ["하/VV"], ["있/VV", "게/EC"], ["되/VV", "었/EP", "다/EF", "./SF"]]]
#
#     tokenized_example = tokenizer.tokenize("그러/VV 던/ETM 어느/MM 날/NNG 나/NP 는/JX 하/VV 있/VV 게/EC 되/VV 었/EP 다/EF ./SF")
#     # morp_tokenized_sentences = []
#     # each_eojeol_lengths = []
#     # each_morp_lengths = []
#     # bert_inputs = []
#     # for original_sentence in original_sentences:
#     # ##### bert 자르는거 testing 중 #####
#     #     morp_tokenized_sent = []
#     #     each_eojeol_length = []
#     #     each_morp_length = []
#     #     for word in original_sentence:
#     #         each_eojeol_length.append(len(word))
#     #         for morp in word:
#     #             morp_tokenized_sent.append(morp)
#     #     for morp in morp_tokenized_sent:
#     #         bert_token = tokenizer.tokenize(morp)
#     #         each_morp_length.append(len(bert_token))
#     #     #morp_tokenized_sentences.append(morp_tokenized_sent)
#     #     each_eojeol_lengths.append(each_eojeol_length)
#     #     each_morp_lengths.append(each_morp_length)
#     #     bert_input = tokenizer.tokenize(" ".join(morp_tokenized_sent))
#     #     bert_inputs.append(bert_input)
#     bert_inputs, each_morp_lengths, each_eojeol_lengths = make_bert_input(original_sentences, tokenizer)
#     max_seq_length = 20
#     train_features = convert_sentence_into_features(bert_inputs, tokenizer, max_seq_length)
#     embedding_model = BertForEmbedding(model)
#
#     embedding_model.to(device)
#
#
#     all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).to(device)
#     all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).to(device)
#     all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).to(device)
#     bert_output=embedding_model(all_input_ids, attention_mask=all_input_mask, token_type_ids=all_segment_ids)
#
#     _, eojeol_bert = resize_bert_output(bert_output, each_morp_lengths, each_eojeol_lengths)
#     # eojeol_bert = torch.tensor(eojeol_bert, dtype=torch.float32)
#     # morp_vectors_list = []
#     # # 우선은 각 BPE token => SUM
#     # for i, each_morp_length in enumerate(each_morp_lengths):
#     #     start_idx = 0
#     #     morp_vectors = []
#     #     for morp_len in each_morp_length:
#     #         end_idx = start_idx + morp_len
#     #         morp_vec_flag=False
#     #         for vec in bert_output[i][start_idx:end_idx]:
#     #             if morp_vec_flag:
#     #                 morp_vec += vec
#     #             else:
#     #                 morp_vec = vec
#     #                 morp_vec_flag=True
#     #         start_idx = end_idx
#     #         morp_vectors.append(morp_vec)
#     #     morp_vectors_list.append(morp_vectors)
#     #
#     # eojeol_vectors_list = []
#     # for i, each_eojeol_length in enumerate(each_eojeol_lengths):
#     #     start_idx = 0
#     #     eojeol_vectors = []
#     #     # 어절 단위일 때는 맨 앞에꺼 + 맨 뒤에꺼
#     #     for eojeol_len in each_eojeol_length:
#     #         end_idx = start_idx + eojeol_len
#     #         eojeol_vec = morp_vectors_list[i][start_idx] + morp_vectors_list[i][end_idx-1]
#     #         start_idx = end_idx
#     #         eojeol_vectors.append(eojeol_vec)
#     #     eojeol_vectors_list.append(eojeol_vectors)
#
#
#     train_features = convert_examples_to_features(
#         train_examples, label_list, args.max_seq_length, tokenizer, args.openapi_key)
#     if len(train_features) == 0:
#         logger.info("The number of train_features is zero. Please check the tokenization. ")
#         sys.exit()
#
#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_examples))
#     logger.info("  Batch size = %d", args.train_batch_size)
#     logger.info("  Num steps = %d", num_train_optimization_steps)
#     all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
#     all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
#     train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#     if args.local_rank == -1:
#         train_sampler = RandomSampler(train_data)
#     else:
#         train_sampler = DistributedSampler(train_data)
#     train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
#
#
#     if args.max_steps > 0:
#         t_total = args.max_steps
#         args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
#     else:
#         t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
#
#     param_optimizer = list(model.named_parameters())
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
#
#
#     model.train()
#     for _ in trange(int(args.num_train_epochs), desc="Epoch"):
#         tr_loss = 0
#         nb_tr_examples, nb_tr_steps = 0, 0
#         for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
#             batch = tuple(t.to(device) for t in batch)
#             input_ids, input_mask, segment_ids, label_ids = batch
#             outputs = model(input_ids, segment_ids, input_mask, label_ids)
#             loss = outputs[0]
#             if n_gpu > 1:
#                 loss = loss.mean()  # mean() to average on multi-gpu.
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
#
#             if args.fp16:
#                 optimizer.backward(loss)
#             else:
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(),
#                                                1.0)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
#                 scheduler.step()
#
#             tr_loss += loss.item()
#             nb_tr_examples += input_ids.size(0)
#             nb_tr_steps += 1
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 global_step += 1
#
#     # Save a trained model and the associated configuration
#     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#     output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
#     torch.save(model_to_save.state_dict(), output_model_file)
#     output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
#     with open(output_config_file, 'w') as f:
#         f.write(model_to_save.config.to_json_string())
#
#     # Load a trained model and config that you have fine-tuned
#     config = BertConfig(output_config_file)
#     config.num_labels = num_labels
#     model = BertForSequenceClassification(config)
#     model.load_state_dict(torch.load(output_model_file))
#
#     model.to(device)
#
# if __name__ == '__main__':
#     main()