import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .dict_v0 import DictionaryAgent


class Autoencoder(nn.Module):
    def __init__(self, emb_size, hidden_size, unbias_size, content_size, dict_file, dropout, rnn_class, device):
        super().__init__()

        dict_opt = {'dict_file': dict_file}
        self.dict = DictionaryAgent(dict_opt)
        self.NULL_IDX = self.dict.tok2ind[self.dict.null_token]
        self.START_IDX = self.dict.tok2ind[self.dict.start_token]
        self.END_IDX = self.dict.tok2ind[self.dict.end_token]
        self.vocab_size = len(self.dict)
        self.longest_label = 20

        self.encoder = Encoder(
            num_features=self.vocab_size, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=emb_size, hidden_size=hidden_size,
            num_layers=1, dropout=dropout)

        self.h2u = nn.Linear(hidden_size, unbias_size, bias=False)
        self.h2c = nn.Linear(hidden_size, content_size, bias=False)

        self.s2h = nn.Linear(unbias_size+content_size, hidden_size, bias=False)

        self.decoder = Decoder(
            num_features=self.vocab_size, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=emb_size, hidden_size=hidden_size,
            num_layers=1, dropout=dropout)

        self.u_classifier_gender = nn.Linear(unbias_size, 2, bias=True)
        self.c_classifier_gender = nn.Linear(content_size, 2, bias=True)

        self.u_classifier_bow = nn.Linear(unbias_size, self.vocab_size, bias=True)
        self.c_classifier_bow = nn.Linear(content_size, self.vocab_size, bias=True)

        self.device = device

    def forward(self, xs, train=True):
        '''

        :param xs: (batch_size x seq_len)
        :return:
        '''

        bsz = len(xs)

        # enc_hidden: (1 x batch_size x hidden_size)
        enc_out, enc_hidden = self.encoder(xs)

        # enc_hidden: (batch_size x hidden_size)
        enc_hidden = enc_hidden.squeeze(0)

        # unbias_code: (batch_size x unbias_size)
        # content_code: (batch_size x content_size)
        unbias_code = nn.functional.relu(self.h2u(enc_hidden))
        content_code = nn.functional.relu(self.h2c(enc_hidden))

        # y_u_gender: (batch_size x 2)
        # y_c_gender: (batch_size x 2)
        y_u_gender = self.u_classifier_gender(unbias_code)
        y_c_gender = self.c_classifier_gender(content_code)

        # y_u_bow: (batch_size x vocab_size)
        # y_c_bow: (batch_size x vocab_size)
        y_u_bow = self.u_classifier_bow(unbias_code)
        y_c_bow = self.c_classifier_bow(content_code)

        # state: (batch_size x unbias_size+content_size)
        state = torch.cat((unbias_code, content_code), 1)

        # dec_hidden: (1 x batch_size x hidden_size)
        dec_hidden = nn.functional.leaky_relu(self.s2h(state)).unsqueeze(0)

        # starts: (batch_size x 1)
        starts = torch.LongTensor([self.START_IDX] * bsz).unsqueeze(1).to(self.device).detach()

        if train:
            y_in = xs.narrow(1, 0, xs.size(1) - 1)
            xs = torch.cat([starts, y_in], 1)

            # preds: (batch_size x seq_len)
            # score: (batch_size x seq_len x vocab_size)
            preds, score, _ = self.decoder(xs, dec_hidden)
            scores = score

        else:
            done = [False for _ in range(bsz)]
            scores = []
            preds = []
            total_done = 0
            xs = starts
            hidden = dec_hidden

            for _ in range(self.longest_label):
                # generate at most longest_label tokens

                # pred: (batch_size x 1)
                # score: (batch_size x 1 x vocab_size)
                pred, score, hidden = self.decoder(xs, hidden)
                scores.append(score)
                xs = pred
                preds.append(pred)

                # check if we've produced the end token
                for b in range(bsz):
                    if not done[b]:
                        # only add more tokens for examples that aren't done
                        if pred.data[b][0] == self.END_IDX:
                            # if we produced END, we're done
                            done[b] = True
                            total_done += 1
                if total_done == bsz:
                    # no need to generate any more
                    break

            # preds: (batch_size x seq_len)
            # score: (batch_size x seq_len x vocab_size)
            preds = torch.cat(preds, 1)
            scores = torch.cat(scores, 1)

        return preds, scores, y_u_gender, y_c_gender, y_u_bow, y_c_bow

    def forward_encoder(self, xs):
        '''

        :param xs: (batch_size x seq_len)
        :return:
        '''

        # enc_hidden: (1 x batch_size x hidden_size)
        enc_out, enc_hidden = self.encoder(xs)

        # enc_hidden: (batch_size x hidden_size)
        enc_hidden = enc_hidden.squeeze(0)

        # unbias_code: (batch_size x unbias_size)
        # content_code: (batch_size x content_size)
        unbias_code = nn.functional.relu(self.h2u(enc_hidden))
        content_code = nn.functional.relu(self.h2c(enc_hidden))

        # y_u_gender: (batch_size x 2)
        # y_c_gender: (batch_size x 2)
        y_u_gender = self.u_classifier_gender(unbias_code)
        y_c_gender = self.c_classifier_gender(content_code)

        # y_u_bow: (batch_size x vocab_size)
        # y_c_bow: (batch_size x vocab_size)
        y_u_bow = self.u_classifier_bow(unbias_code)
        y_c_bow = self.c_classifier_bow(content_code)

        return y_u_gender, y_c_gender, y_u_bow, y_c_bow

    def forward_encoder_oh(self, xs_oh, x_len):
        '''

        :param xs_oh: (batch_size x seq_len x vocab_size)
        :param x_lens: (batch_size)
        :return:
        '''

        # enc_hidden: (1 x batch_size x hidden_size)
        enc_out, enc_hidden = self.encoder.forward_oh(xs_oh, x_len)

        # enc_hidden: (batch_size x hidden_size)
        enc_hidden = enc_hidden.squeeze(0)

        # unbias_code: (batch_size x unbias_size)
        # content_code: (batch_size x content_size)
        unbias_code = nn.functional.relu(self.h2u(enc_hidden))
        content_code = nn.functional.relu(self.h2c(enc_hidden))

        return unbias_code, content_code


    def get_c_classifier_gender_params(self):

        return self.c_classifier_gender.parameters()

    def get_u_classifier_bow_params(self):

        return self.u_classifier_bow.parameters()

    def get_rest_parameters(self):
        for name, param in self.named_parameters(recurse=True):
            # print(name)
            if not name.startswith('c_classifier_gender') and not name.startswith('u_classifier_bow'):
                # print(name, param)
                yield param


class Encoder(nn.Module):
    def __init__(self, num_features, padding_idx=0, rnn_class=nn.GRU,
                 emb_size=128, hidden_size=128, num_layers=1, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.hsz = hidden_size

        self.lt = nn.Embedding(num_features, emb_size,
                                   padding_idx=padding_idx,
                                   sparse=False)

        self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                                 dropout=dropout, batch_first=True,
                                 bidirectional=False)

    def forward(self, xs):
        # embed input tokens
        # xes = self.dropout(self.lt(xs))
        ori_xes = self.dropout(self.lt(xs))
        try:
            # x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
            # xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            # packed = True
            ori_x_lens = np.asarray([x for x in torch.sum((xs > 0).int(), dim=1).data])
            # print("ori_x_lens: ", ori_x_lens)
            x_lens_arg = np.argsort( - ori_x_lens)
            x_lens_arg_ver = []
            for i in range(len(ori_x_lens)):
                x_lens_arg_ver.append(np.where(x_lens_arg==i)[0][0])
            # x_lens_arg_ver = np.asarray(x_lens_arg_ver)
            # print("x_lens_arg: ", x_lens_arg)
            # print("x_lens_arg_ver: ", x_lens_arg_ver)
            x_lens = ori_x_lens[x_lens_arg].tolist()
            # print("x_lens: ", x_lens)
            xes = ori_xes[x_lens_arg]
            # print("xes: ", xes.size())

            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            # print("xes: ", xes)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        # print("encoder_output: ", encoder_output)
        # print("hidden: ", hidden)
        # print("hidden: ", hidden[0].size(), hidden[1].size())
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        # print("encoder_output: ", encoder_output.size())

        hidden = torch.transpose(torch.transpose(hidden, 0, 1)[x_lens_arg_ver], 0, 1).contiguous()
        # hidden = (torch.transpose(torch.transpose(hidden[0], 0, 1)[x_lens_arg_ver], 0, 1).contiguous(), torch.transpose(torch.transpose(hidden[1], 0, 1)[x_lens_arg_ver], 0, 1).contiguous())
        # print("hidden: ", hidden)
        # print("encoder_output: ", encoder_output)
        encoder_output = encoder_output[x_lens_arg_ver]
        # print("encoder_output: ", encoder_output)

        return encoder_output, hidden

    def forward_oh(self, xs_oh, x_lens):
        '''

        :param xs_oh: (batch_size x seq_len x vocab_size)
        :param x_lens: (batch_size)
        :return:
        '''

        # embed input tokens
        # xes: (batch_size x seq_len x emb_size)
        xes = torch.matmul(xs_oh, self.lt.weight)

        ori_xes = self.dropout(xes)
        try:
            # x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
            # xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            # packed = True
            ori_x_lens = x_lens
            # print("ori_x_lens: ", ori_x_lens)
            x_lens_arg = np.argsort( - ori_x_lens)
            x_lens_arg_ver = []
            for i in range(len(ori_x_lens)):
                x_lens_arg_ver.append(np.where(x_lens_arg==i)[0][0])
            # x_lens_arg_ver = np.asarray(x_lens_arg_ver)
            # print("x_lens_arg: ", x_lens_arg)
            # print("x_lens_arg_ver: ", x_lens_arg_ver)
            x_lens = ori_x_lens[x_lens_arg].tolist()
            # print("x_lens: ", x_lens)
            xes = ori_xes[x_lens_arg]
            # print("xes: ", xes.size())

            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            # print("xes: ", xes)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        # print("encoder_output: ", encoder_output)
        # print("hidden: ", hidden)
        # print("hidden: ", hidden[0].size(), hidden[1].size())
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        # print("encoder_output: ", encoder_output.size())

        hidden = torch.transpose(torch.transpose(hidden, 0, 1)[x_lens_arg_ver], 0, 1).contiguous()
        # hidden = (torch.transpose(torch.transpose(hidden[0], 0, 1)[x_lens_arg_ver], 0, 1).contiguous(), torch.transpose(torch.transpose(hidden[1], 0, 1)[x_lens_arg_ver], 0, 1).contiguous())
        # print("hidden: ", hidden)
        # print("encoder_output: ", encoder_output)
        encoder_output = encoder_output[x_lens_arg_ver]
        # print("encoder_output: ", encoder_output)

        return encoder_output, hidden

class Decoder(nn.Module):
    def __init__(self, num_features, padding_idx=0, rnn_class=nn.LSTM,
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()

        if padding_idx != 0:
            raise RuntimeError('This module\'s output layer needs to be fixed '
                               'if you want a padding_idx other than zero.')

        self.dropout = nn.Dropout(p=dropout)
        print("dropout: ", dropout)
        self.layers = num_layers
        self.hsz = hidden_size
        self.esz = emb_size

        self.lt = nn.Embedding(num_features, emb_size, padding_idx=padding_idx,
                               sparse=False)
        self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)

        # rnn output to embedding
        if hidden_size != emb_size:
            # self.o2e = RandomProjection(hidden_size, emb_size)
            # other option here is to learn these weights
            self.o2e = nn.Linear(hidden_size, emb_size, bias=False)
        else:
            # no need for any transformation here
            self.o2e = lambda x: x
        # embedding to scores, use custom linear to possibly share weights
        shared_weight = None
        self.e2s = Linear(emb_size, num_features, bias=False,
                          shared_weight=shared_weight)
        self.shared = shared_weight is not None


    def forward(self, xs, hidden, topk=1):
        xes = self.dropout(self.lt(xs))
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)

        e = self.dropout(self.o2e(output))
        scores = self.e2s(e)

        # # exclude __UNK__
        # scores[:, :, 3] = -1e8

        # select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
        if topk == 1:
            _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)
        elif topk > 1:
            max_score, idx = torch.topk(F.softmax(scores.narrow(2, 1, scores.size(2) - 1), 2), topk, dim=2, sorted=False)
            probs = F.softmax(scores.narrow(2, 1, scores.size(2) - 1).gather(2, idx), 2).squeeze(1)
            dist = torch.distributions.categorical.Categorical(probs)
            samples = dist.sample()
            idx = idx.gather(-1, samples.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
        preds = idx.add_(1)

        return preds, scores, new_hidden

    def forward_embed(self, xes, hidden, encoder_output, temp, attn_mask=None):
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)

        # output: (batch_size x 1 x hidden_size)
        output, new_hidden = self.rnn(xes, hidden)

        # e: (batch_size x 1 x emb_size)
        e = self.dropout(self.o2e(output))

        # scores: (batch_size x vocab_size)
        scores = self.e2s(e).squeeze(1)

        # y: (batch_size x vocab_size)
        y = F.softmax(scores / temp, dim=1)

        # feature: (batch_size x emb_size)
        feature = torch.matmul(y, self.lt.weight)

        # select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
        # preds: (batch_size)
        _max_score, idx = scores.narrow(1, 1, scores.size(1) - 1).max(1)
        preds = idx.add_(1)

        return preds.unsqueeze(1), feature, new_hidden

    def forward_without_attention(self, xs, hidden):
        xes = self.dropout(self.lt(xs))
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)


        e = self.dropout(self.o2e(output))
        scores = self.e2s(e)

        # select top scoring index, excluding the padding symbol (at idx zero)
        # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
        _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)

        # print("idx: ", idx)

        preds = idx.add_(1)

        # print("preds: ", preds)

        return preds, scores, new_hidden

class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'