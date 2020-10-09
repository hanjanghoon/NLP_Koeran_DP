# -*- coding=utf-8 -*-
from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs-TreeCRF model for Graph-based dependency parsing.
"""

import sys
import os

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json

import numpy as np
import torch
from neuronlp2.io import get_logger, conllx_stacked_data, conllx_data
from neuronlp2.io import CoNLLXWriter
from neuronlp2.tasks import parser_bpe
from neuronlp2.models import StackPtrNet, BiRecurrentConvBiAffine
from neuronlp2 import utils

uid = uuid.uuid4().hex[:6]


def main():
    args_parser = argparse.ArgumentParser(description='Tuning with stack pointer parser')
    args_parser.add_argument('--parser', choices=['stackptr', 'biaffine'], help='Parser', default='stackptr')
    args_parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)
    # 'models/stack_ptr/92.17/'
    args_parser.add_argument('--model_name', help='name for saving model file.', default='network.pt')
    # 'network.pt'
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--ordered', action='store_true', help='Using order constraints in decoding')
    args_parser.add_argument('--display', action='store_true', help='Display wrong examples')
    args_parser.add_argument('--gpu', action='store_true', help='Using GPU')
    args_parser.add_argument('--pos_embedding', type=int, default=4)

    args = args_parser.parse_args()

    logger = get_logger("Analyzer")

    test_path = args.test
    model_path = args.model_path
    model_name = args.model_name

    punct_set = None
    punctuation = args.punctuation
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    use_gpu = args.gpu

    parser = args.parser
    
    if parser == 'stackptr':
        stackptr(model_path, model_name, test_path, punct_set, use_gpu, logger, args)
    else:
        raise ValueError('Unknown parser: %s' % parser)


def stackptr(model_path, model_name, test_path, punct_set, use_gpu, logger, args):
    pos_embedding = args.pos_embedding
    alphabet_path = os.path.join(model_path, 'alphabets/')
    model_name = os.path.join(model_path, model_name)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_stacked_data.create_alphabets(alphabet_path, None, pos_embedding,data_paths=[None, None], max_vocabulary_size=50000, embedd_dict=None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    beam = args.beam
    ordered = args.ordered
    display_inst = args.display

    def load_model_arguments_from_json():
        arguments = json.load(open(arg_path, 'r'))
        return arguments['args'], arguments['kwargs']

    arg_path = model_name + '.arg.json'
    args, kwargs = load_model_arguments_from_json()

    prior_order = kwargs['prior_order']
    logger.info('use gpu: %s, beam: %d, order: %s (%s)' % (use_gpu, beam, prior_order, ordered))

    data_test = conllx_stacked_data.read_stacked_data_to_variable(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet, pos_embedding,
                                                                  use_gpu=use_gpu, volatile=True, prior_order=prior_order, is_test=True)

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet, pos_embedding)

    logger.info('model: %s' % model_name)
    # kwargs???�로??embedidng 추�?    
    word_path = os.path.join(model_path, 'embedding.txt')
    word_dict, word_dim = utils.load_embedding_dict('NNLM', word_path)
    def get_embedding_table():
        table = np.empty([len(word_dict), word_dim])
        for idx,(word, embedding) in enumerate(word_dict.items()):
            try:
                table[idx, :] = embedding
            except:
                print(word)
        return torch.from_numpy(table)
    word_table = get_embedding_table()
    kwargs['embedd_word'] = word_table
    args[1] = len(word_dict) # word_dim
    network = StackPtrNet(*args, **kwargs)
    # word_embedidng?� ??불러?�기
    model_dict = network.state_dict()
    pretrained_dict = torch.load(model_name)
    model_dict.update({k:v for k,v in pretrained_dict.items()
        if k != 'word_embedd.weight'})
    
    network.load_state_dict(model_dict)

    if use_gpu:
        network.cuda()
    else:
        network.cpu()

    network.eval()

    if not ordered:
        pred_writer.start(model_path + '/tmp/inference.txt')
    else:
        pred_writer.start(model_path + '/tmp/inference_ordered_temp.txt')
    sent = 0
    start_time = time.time()
    for batch in conllx_stacked_data.iterate_batch_stacked_variable(data_test, 1, pos_embedding, type='dev'):
        sys.stdout.write('%d, ' % sent)
        sys.stdout.flush()
        sent += 1

        input_encoder, input_decoder = batch
        word, char, pos, heads, types, masks, lengths = input_encoder
        stacked_heads, children, siblings, stacked_types, skip_connect, mask_d, lengths_d = input_decoder
        heads_pred, types_pred, children_pred, stacked_types_pred = network.decode(word, char, pos, mask=masks, length=lengths, beam=beam, ordered=ordered,
                                                                                   leading_symbolic=conllx_stacked_data.NUM_SYMBOLIC_TAGS)

        stacked_heads = stacked_heads.data
        children = children.data
        stacked_types = stacked_types.data
        children_pred = torch.from_numpy(children_pred).long()
        stacked_types_pred = torch.from_numpy(stacked_types_pred).long()
        if use_gpu:
            children_pred = children_pred.cuda()
            stacked_types_pred = stacked_types_pred.cuda()

        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        lengths = lengths.cpu().numpy()
        heads = heads.data.cpu().numpy()
        types = types.data.cpu().numpy()

        pred_writer.write(word, pos, heads_pred, types_pred, lengths, symbolic_root=True)
    pred_writer.close()

if __name__ == '__main__':
    main()
