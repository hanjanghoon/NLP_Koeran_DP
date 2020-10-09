#-*- coding: utf-8 -*-
__author__ = 'max'

import copy
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..nn import TreeCRF, VarMaskedGRU, VarMaskedRNN, VarMaskedLSTM, VarMaskedFastLSTM
from ..nn import SkipConnectFastLSTM, SkipConnectGRU, SkipConnectLSTM, SkipConnectRNN
from ..nn import Embedding    # 2to3
from ..nn import BiAAttention, BiLinear
from neuronlp2.tasks import parser
from .elmocode import Embedder
#from allennlp.modules.elmo import Elmo

from bert.bert_for_embedding import BertForEmbedding, BertForEncoder, make_bert_input, resize_bert_output,\
                                    convert_sentence_into_features, convert_into_bert_feature_indices, SelfAttentiveModel

## version check yj
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert.tokenization_morp import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel,BertConfig, WEIGHTS_NAME, CONFIG_NAME
from .custom_modeling import BertModel2
#from bert.tokenization_kor_bert import BertTokenizer
#from pytorch_transformers import BertModel, BertConfig

#option_file = "/data/embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "/data/embedding/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class PriorOrder(Enum):
    DEPTH = 0
    INSIDE_OUT = 1
    LEFT2RIGTH = 2


class BiRecurrentConvBiAffine(nn.Module):
    def __init__(self, word_dim, num_words, char_dim, num_chars, pos_dim, num_pos, num_filters, kernel_size, rnn_mode, hidden_size, num_layers, num_labels, arc_space, type_space,
                 embedd_word=None, embedd_char=None, embedd_pos=None, p_in=0.33, p_out=0.33, p_rnn=(0.33, 0.33), biaffine=True, pos=True, char=True,
                 elmo=False, elmo_path=None, elmo_dim=None, bert=False, bert_path=None, bert_dim=None, self_attention=False, sa_path=None):
        super(BiRecurrentConvBiAffine, self).__init__()

        # self.word_embedd = Embedding(num_words, word_dim, init_embedding=embedd_word)
        # self.pos_embedd = Embedding(num_pos, pos_dim, init_embedding=embedd_pos) if pos else None
        # self.char_embedd = Embedding(num_chars, char_dim, init_embedding=embedd_char) if char else None
        # self.conv1d = nn.Conv1d(char_dim, num_filters, kernel_size, padding=kernel_size - 1) if char else None
        # ch 논문 기준
        self.rnn_dropout = nn.Dropout2d(0.1)

        self.dropout_in = nn.Dropout2d(p=p_in)
        self.dropout_out = nn.Dropout2d(p=p_out)

        self.my_dropout = nn.Dropout(p=p_out)
        self.num_labels = num_labels
        self.pos = pos
        self.char = char
        self.bert = bert

        self.self_attention = self_attention
        self.sa_path = sa_path
        #yjyj
        self.bert_dim = bert_dim
        self.elmo = elmo
        self.hidden_size = hidden_size
        #if self.bert:

        #hoon : elmo
        if self.elmo:
            self.elmo_embedd = Embedder(elmo_path)

        # yj 버전 체크
        self.tokenizer = BertTokenizer.from_pretrained(bert_path + "./vocab.txt", do_lower_case=False)
        self.bert_model1 = BertModel.from_pretrained(bert_path)
        self.bert_model2 = BertModel2.from_pretrained(bert_path)

        # config = BertConfig.from_pretrained(bert_path)
        # # 요거 필수 추가
        # #config.output_hidden_states = True
        #
        # self.tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        # # 루트 추가
        # self.tokenizer.add_tokens(["_ROOT"])
        # self.bert_model = BertModel.from_pretrained(bert_path, config=config)
        #bert_base_model.resize_token_embeddings(len(self.tokenizer))
        #bert_base_model.train()
        self.bert_dim = bert_dim
        #self.bert_model = BertForEmbedding(bert_base_model)
        self.bert_word_feature_embedd = Embedding(3, 1600, padding_idx=0)   # (B-word & I-word)
        self.bert_morp_feature_embedd = Embedding(3, 1600, padding_idx=0)   # (B-morp & I-morp)

        if rnn_mode == 'RNN':
            RNN = VarMaskedRNN
        elif rnn_mode == 'LSTM':
            RNN = VarMaskedLSTM
        elif rnn_mode == 'FastLSTM':
            RNN = VarMaskedFastLSTM
        elif rnn_mode == 'GRU':
            RNN = VarMaskedGRU
        else:
            raise ValueError('Unknown RNN mode: %s' % rnn_mode)

        dim_enc = bert_dim*2
        # if pos:
        #     dim_enc += pos_dim
        # if char:
        #     dim_enc += num_filters
        if self_attention:
            self.config = BertConfig(sa_path)
            self.self_attentive_model = SelfAttentiveModel(self.config)  # 이 부분 코드 고쳐야함
            # self.output_dim_change = nn.Linear(dim_enc, self.hidden_size * 2)
        else:
            self.rnn = RNN(bert_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)
        # self.rnn = torch.nn.LSTM(input_size=dim_enc, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        # self.eojeol_rnn = RNN(hidden_size*2, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=p_rnn)

        self.src_dense=nn.Linear(dim_enc,bert_dim)

        out_dim = hidden_size * 2
        #out_dim=bert_dim
        self.arc_h = nn.Linear(out_dim, arc_space)
        self.arc_c = nn.Linear(out_dim, arc_space)
        self.attention = BiAAttention(arc_space, arc_space, 1, biaffine=biaffine)

        self.type_h = nn.Linear(out_dim, type_space)
        self.type_c = nn.Linear(out_dim, type_space)
        self.bilinear = BiLinear(type_space, type_space, self.num_labels)


    def _get_rnn_output(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, input_word_elmo=None, input_word_bert=None):
        # [batch, length, word_dim]
        # word = self.word_embedd(input_word)
        # # apply dropout on input
        # word = self.dropout_in(word)
        #
        # input = word
        #
        # if self.char:
        #     # [batch, length, char_length, char_dim]
        #     char = self.char_embedd(input_char)
        #     char_size = char.size()
        #     # first transform to [batch *length, char_length, char_dim]
        #     # then transpose to [batch * length, char_dim, char_length]
        #     char = char.view(char_size[0] * char_size[1], char_size[2], char_size[3]).transpose(1, 2)
        #     # put into cnn [batch*length, char_filters, char_length]
        #     # then put into maxpooling [batch * length, char_filters]
        #     char, _ = self.conv1d(char).max(dim=2)
        #     # reshape to [batch, length, char_filters]
        #     char = torch.tanh(char).view(char_size[0], char_size[1], -1)
        #     # apply dropout on input
        #     char = self.dropout_in(char)
        #     # concatenate word and char [batch, length, word_dim+char_filter]
        #     input = torch.cat([input, char], dim=2)
        #
        # if self.pos:
        #     # [batch, length, pos_dim]
        #     pos = self.pos_embedd(input_pos)
        #     # apply dropout on input
        #     pos = self.dropout_in(pos)
        #     input = torch.cat([input, pos], dim=2)
        #
        # # output from rnn [batch, length, hidden_size]p
        # output, hn = self.rnn(input, mask, hx=hx)

        bert_inputs, each_morp_lengths, each_eojeol_lengths = make_bert_input(input_word_bert, self.tokenizer)
        max_seq_length = max(
            [len(entry) for entry in
             bert_inputs]) + 1  # Bert tokenizer 기준 max_seq_length, [CLS], [SEP] 추가, _ROOT_ 빼는 걸로 총 + 1
        train_features = convert_sentence_into_features(bert_inputs, self.tokenizer, max_seq_length)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        bert_output, _ = self.bert_model1(all_input_ids, attention_mask=all_input_mask, token_type_ids=all_segment_ids)
        bert_output = bert_output[-1]+bert_output[-2]+bert_output[-3]+bert_output[-4]

        # bert_length = torch.tensor([len(bert_input) for bert_input in bert_inputs]).cuda()
        # packed_src_encoding = nn.utils.rnn.pack_padded_sequence(src_encoding, bert_length, batch_first=True,
        #                                                         enforce_sorted=False)
        # packed_output, hn = self.rnn(packed_src_encoding)
        # rnn_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_seq_length)

        # # bert + 자질 임베딩을 encoder 한 번 통과시키고 난 값을 어절별로 정리한다..
        max_eojeol_length = input_word.size(1)

        #JH
        # output, _ = self.eojeol_rnn(eojeol_vectors_tensor, mask, hx=hx)
        _, output = resize_bert_output(bert_output, each_morp_lengths, each_eojeol_lengths,
                                       max_eojeol_length=max_eojeol_length, output_dim=self.hidden_size * 2,
                                       use_first_token=True)

        '''공사시작.'''
        all_input_emb = F.elu(self.src_dense(output))
        all_input_mask = mask
        bert_output, bert_cls = self.bert_model2(inputs_embeds=all_input_emb, attention_mask=all_input_mask)
        output = bert_output[-4] + bert_output[-3] + bert_output[-2] + bert_output[-1]

        output, _ = self.rnn(output, mask, hx=hx)
        output = self.dropout_out(output.transpose(1, 2)).transpose(1, 2)
        # apply dropout for output
        # [batch, length, hidden_size] --> [batch, hidden_size, length] --> [batch, length, hidden_size]
        # output = self.dropout_out(eojeol_vectors_tensor.transpose(1, 2)).transpose(1, 2)

        # output = self.my_dropout(eojeol_vectors_tensor)
        # packed_src_encoding = nn.utils.rnn.pack_padded_sequence(eojeol_output, length, batch_first=True,
        #                                                         enforce_sorted=False)
        # packed_output, hn = self.eojeol_rnn(packed_src_encoding)
        # output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=max_eojeol_length
        # output = self.my_dropout(output)


        # output size [batch, length, arc_space]
        arc_h = F.elu(self.arc_h(output))
        arc_c = F.elu(self.arc_c(output))

        # output size [batch, length, type_space]
        type_h = F.elu(self.type_h(output))
        type_c = F.elu(self.type_c(output))

        # apply dropout
        # [batch, length, dim] --> [batch, 2 * length, dim]
        arc = torch.cat([arc_h, arc_c], dim=1)
        type = torch.cat([type_h, type_c], dim=1)

        # arc = self.dropout_out(arc.transpose(1, 2)).transpose(1, 2)
        arc = self.dropout_out(arc)
        arc_h, arc_c = arc.chunk(2, 1)

        # type = self.dropout_out(type.transpose(1, 2)).transpose(1, 2)
        type = self.dropout_out(type)
        type_h, type_c = type.chunk(2, 1)
        type_h = type_h.contiguous()
        type_c = type_c.contiguous()

        return (arc_h, arc_c), (type_h, type_c), _, mask, length

    def forward(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, input_word_elmo=None, input_word_bert=None):
        # output from rnn [batch, length, tag_space]
        arc, type, _, mask, length = self._get_rnn_output(input_word, input_char, input_pos, mask=mask, length=length, hx=hx, input_word_elmo=input_word_elmo, input_word_bert=input_word_bert)
        # [batch, length, length]
        out_arc = self.attention(arc[0], arc[1], mask_d=mask, mask_e=mask).squeeze(dim=1)
        return out_arc, type, mask, length

    def loss(self, input_word, input_char, input_pos, heads, types, mask=None, length=None, hx=None, input_word_elmo=None, input_word_bert=None):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx, input_word_elmo=input_word_elmo,  input_word_bert=input_word_bert)
        batch, max_len, _ = out_arc.size()

        if length is not None and heads.size(1) != mask.size(1):
            heads = heads[:, :max_len]
            types = types[:, :max_len]

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type

        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(out_arc.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.data.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # loss_arc shape [batch, length, length]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length, num_labels]
        loss_type = F.log_softmax(out_type, dim=2)

        # mask invalid position to 0 for sum loss
        if mask is not None:
            loss_arc = loss_arc * mask.unsqueeze(2) * mask.unsqueeze(1)
            loss_type = loss_type * mask.unsqueeze(2)
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = mask.sum() - batch
        else:
            # number of valid positions which contribute to loss (remove the symbolic head for each sentence.
            num = float(max_len - 1) * batch

        # first create index matrix [length, batch]
        child_index = torch.arange(0, max_len).view(max_len, 1).expand(max_len, batch)
        child_index = child_index.type_as(out_arc.data).long()
        # [length-1, batch]
        loss_arc = loss_arc[batch_index, heads.data.t(), child_index][1:]
        loss_type = loss_type[batch_index, child_index, types.data.t()][1:]

        return -loss_arc.sum() / num, -loss_type.sum() / num

    def _decode_types(self, out_type, heads, leading_symbolic):
        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, _ = type_h.size()
        # create batch index [batch]
        batch_index = torch.arange(0, batch).type_as(type_h.data).long()
        # get vector for heads [batch, length, type_space],
        type_h = type_h[batch_index, heads.t()].transpose(0, 1).contiguous()
        # compute output for type [batch, length, num_labels]
        out_type = self.bilinear(type_h, type_c)
        # remove the first #leading_symbolic types.
        out_type = out_type[:, :, leading_symbolic:]
        # compute the prediction of types [batch, length]
        _, types = out_type.max(dim=2)
        return types + leading_symbolic

    def decode(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0, input_word_bert=None):
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx, input_word_bert=input_word_bert)
        out_arc = out_arc.data
        batch, max_len, _ = out_arc.size()
        # set diagonal elements to -inf
        out_arc = out_arc + torch.diag(out_arc.new(max_len).fill_(-np.inf))
        # set invalid positions to -inf
        if mask is not None:
            # minus_mask = (1 - mask.data).byte().view(batch, max_len, 1)
            minus_mask = (1 - mask.data).byte().unsqueeze(2)
            out_arc.masked_fill_(minus_mask, -np.inf)

        # compute naive predictions.
        # predition shape = [batch, length]
        _, heads = out_arc.max(dim=1)

        types = self._decode_types(out_type, heads, leading_symbolic)

        return heads.cpu().numpy(), types.data.cpu().numpy()

    def decode_mst(self, input_word, input_char, input_pos, mask=None, length=None, hx=None, leading_symbolic=0, input_word_elmo=None, input_word_bert=None):
        '''
        Args:
            input_word: Tensor
                the word input tensor with shape = [batch, length]
            input_char: Tensor
                the character input tensor with shape = [batch, length, char_length]
            input_pos: Tensor
                the pos input tensor with shape = [batch, length]
            mask: Tensor or None
                the mask tensor with shape = [batch, length]
            length: Tensor or None
                the length tensor with shape = [batch]
            hx: Tensor or None
                the initial states of RNN
            leading_symbolic: int
                number of symbolic labels leading in type alphabets (set it to 0 if you are not sure)

        Returns: (Tensor, Tensor)
                predicted heads and types.

        '''
        # out_arc shape [batch, length, length]
        out_arc, out_type, mask, length = self.forward(input_word, input_char, input_pos, mask=mask, length=length, hx=hx, input_word_elmo=input_word_elmo, input_word_bert=input_word_bert)

        # out_type shape [batch, length, type_space]
        type_h, type_c = out_type
        batch, max_len, type_space = type_h.size()

        # compute lengths
        if length is None:
            if mask is None:
                length = [max_len for _ in range(batch)]
            else:
                length = mask.data.sum(dim=1).long().cpu().numpy()

        type_h = type_h.unsqueeze(2).expand(batch, max_len, max_len, type_space).contiguous()
        type_c = type_c.unsqueeze(1).expand(batch, max_len, max_len, type_space).contiguous()
        # compute output for type [batch, length, length, num_labels]
        out_type = self.bilinear(type_h, type_c)

        # mask invalid position to -inf for log_softmax
        if mask is not None:
            minus_inf = -1e8
            minus_mask = (1 - mask) * minus_inf
            out_arc = out_arc + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # loss_arc shape [batch, length, length]
        loss_arc = F.log_softmax(out_arc, dim=1)
        # loss_type shape [batch, length, length, num_labels]
        loss_type = F.log_softmax(out_type, dim=3).permute(0, 3, 1, 2)
        # [batch, num_labels, length, length]
        energy = torch.exp(loss_arc.unsqueeze(1) + loss_type)

        return parser.decode_MST(energy.data.cpu().numpy(), length, leading_symbolic=leading_symbolic, labeled=True)
