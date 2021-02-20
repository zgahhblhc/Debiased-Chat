import os
import importlib
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np
from .evaluate import get_data_for_inference, evaluate
from .options import opt
from .discriminators import Biased_Discriminator, Unbiased_Discriminator
from .disentangle_model.model import Autoencoder
from ..evaluation.fairness_test import bias_evaluate

def pad(list, padding=0, min_len=None):
    padded = []
    max_len = max([len(l) for l in list])
    if min_len:
        max_len = max(min_len, max_len)
    for l in list:
        padded.append(l + [padding] * (max_len - len(l)))

    return torch.tensor(padded, dtype=torch.long)

def get_lengths(batch):
    batch_list = batch.tolist()
    eos_pos = [[i for i in range(len(ins)) if ins[i] == eos_id] for ins in batch_list]

    lens = []
    for ins, eos in zip(batch_list, eos_pos):
        if len(eos) > 0:
            lens.append(max(min(eos), 1))
        else:
            lens.append(len(ins))

    return np.asarray(lens)

def EntropyLoss(tensor):
    '''

    :param tensor: (batch_size x m)
    :return:
    '''
    minimum = np.log(0.5)

    p = nn.functional.softmax(tensor, dim=1)
    log_p = torch.log(p + 1e-10)

    return torch.mean(torch.sum(p * log_p, dim=1) - minimum)

def accuracy(predictions, truths):
    pads = torch.Tensor.double(truths != 0)
    corrects = torch.Tensor.double(predictions == truths)
    valid_corrects = corrects * pads

    return valid_corrects.sum() / pads.sum()

def batchize(data_list):
    n = 10
    i = 0
    data_pool = []

    while i <= len(data_list) // (n * batch_size):
        nbatch = data_list[i * (n * batch_size): (i + 1) * (n * batch_size)]
        nbatch_list = [([agent.dict.tok2ind.get(word, unk_id) for word in ins[0].split()],
                        [agent.dict.tok2ind.get(word, unk_id) for word in ins[1].split()] + [eos_id],
                        ins[2]) for ins in nbatch]
        descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

        j = 0
        while len(descend_nbatch_list[j * batch_size: (j + 1) * batch_size]) > 0:
            batch_list = descend_nbatch_list[j * batch_size: (j + 1) * batch_size]
            # print(len(batch_list))
            # source: (batch_size x seq_len)
            source = pad([x[0] for x in batch_list], padding=null_id)
            target = pad([x[1] for x in batch_list], padding=null_id)
            labels = torch.tensor([x[2] for x in batch_list], dtype=torch.long)
            data_pool.append((source, target, labels))

            j += 1

        i += 1

    return data_pool

def contain_word(text, set):
    if any([True if w in set else False for w in text.split()]):
        return True
    else:
        return False

def get_params_of_two_modules(module_a, module_b, recurse=True):
    param_module_a = [elem for elem in module_a.named_parameters(recurse=recurse)]
    param_module_b = [elem for elem in module_b.named_parameters(recurse=recurse)]

    for name, param in param_module_a + param_module_b:
        yield param

def get_response(agent, context_batch):
    response_batch = []
    xs = agent.vectorize(context_batch)
    predicts, _ = agent.predict(xs)
    for predict in predicts:
        pred = agent.dict.vec2txt(predict).split()
        eos_ids = [i for i in range(len(pred)) if pred[i] == '__END__' or pred[i] == '__end__']
        response = ' '.join(pred[:eos_ids[0]]) if len(eos_ids) > 0 else ' '.join(pred)
        response_batch.append(response)

    return response_batch

def fairness_test(agent, contexts):
    bias_data_left, bias_data_right = [], []

    bsz = 100
    n_batch = len(contexts) // bsz

    count = 0
    for i in range(n_batch):
        context_batch = contexts[i * bsz: (i + 1) * bsz]
        left_context_batch = [c[0] for c in context_batch]
        right_context_batch = [c[1] for c in context_batch]
        left_response_batch = get_response(agent, left_context_batch)
        right_response_batch = get_response(agent, right_context_batch)

        for left_context, left_response, right_context, right_response in zip(left_context_batch, left_response_batch,
                                                                              right_context_batch,
                                                                              right_response_batch):

            if count % 500 == 0:
                print("{} / {}".format(count, n_batch * bsz))
                print("context: {}".format(left_context))
                print("left response: {}".format(left_response))
                print("context: {}".format(right_context))
                print("right response: {}".format(right_response))
                print(
                    "-------------------------------------------------------------------------------------------------------")

            bias_data_left.append(left_context + '\t' + left_response + '\n')
            bias_data_right.append(right_context + '\t' + right_response + '\n')

            count += 1

    bias_data = (bias_data_left, bias_data_right)

    results = bias_evaluate(bias_data)

    return results


device = torch.device("cuda")
# create generator.
my_module = importlib.import_module("seq2seq_v0")
model_class = getattr(my_module, "Seq2seqAgent")
agent = model_class(opt)
generator = agent.model.to(device)
generator.longest_label = 30

print(generator.longest_label)

disen_opt = {'emb_size': 300, 'hidden_size': 1000, 'unbias_size': 200, 'content_size': 800, 'rnn_class': nn.GRU, 'dropout': 0.0}

print(disen_opt)

disen_model = Autoencoder(emb_size=disen_opt['emb_size'], hidden_size=disen_opt['hidden_size'], unbias_size=disen_opt['unbias_size'], content_size=disen_opt['content_size'],
                    dict_file='../data/twitter_seq2seq_model.dict',
                    dropout=disen_opt['dropout'], rnn_class=disen_opt['rnn_class'], device=device).to(device)

disen_model_file = 'disentangle_model/save_model/disen_model.pt'
disen_model.load_state_dict(torch.load(disen_model_file))
print("Disentangle model {} loaded.".format(disen_model_file))

# create biased discriminator
D_biased = Biased_Discriminator(input_dim=800).to(device)
D_biased_file = 'pretrained_model/D_biased.pt'
D_biased.load_state_dict(torch.load(D_biased_file))
print("D_biased model {} loaded.".format(D_biased_file))

# create unbiased discriminator
D_unbiased = Unbiased_Discriminator(input_dim=200).to(device)
D_unbiased_file = 'pretrained_model/D_unbiased.pt'
D_unbiased.load_state_dict(torch.load(D_unbiased_file))
print("D_unbiased model {} loaded.".format(D_unbiased_file))

import argparse
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--D_biased_steps', type=int, default=1)
parser.add_argument('--G_adv_steps', type=int, default=1)
parser.add_argument('--G_teach_steps', type=int, default=1)
parser.add_argument('--k0', type=int, default=1)
parser.add_argument('--k1', type=int, default=1)
parser.add_argument('--k2', type=int, default=1)

args = parser.parse_args()

batch_size = args.batch_size
lr = 0.001
clip = 0.25
temp = 1
k0 = args.k0
k1 = args.k1
k2 = args.k2
D_biased_steps = args.D_biased_steps
G_adv_steps = args.G_adv_steps
G_teach_steps = args.G_teach_steps
params = "D_biased_steps_{}_G_adv_steps_{}_G_teach_steps_{}_k0_{}_k1_{}_k2_{}_nc_{}".format(D_biased_steps, G_adv_steps, G_teach_steps, k0, k1, k2, nc)

print("params: ", params)

print("Loading gender corpus ...")
with open('adv_train_data.json', 'r') as f:
    train_gender_data_list, train_neutral_data_list = json.load(f)

print("Loading neutral corpus ...")


# The neutral Twitter dataset for validating and testing the performance of the dialogue model is from ParlAI.
# For more details, please see https://parl.ai/docs/tutorial_task.html

valid_data_list = get_data_for_inference('/ParlAI/data/Twitter/valid.txt')
test_data_list = get_data_for_inference('/ParlAI/data/Twitter/test.txt')

male_to_female, female_to_male = {}, {}
with open('../data/gender_words', 'r') as f:
    word_lines = f.readlines()
for line in word_lines:
    male_word, female_word = line.strip().split(' - ')
    male_to_female[male_word] = female_word
    female_to_male[female_word] = male_word


null_id = agent.dict.tok2ind[agent.dict.null_token]
eos_id = agent.dict.tok2ind[agent.dict.end_token]
unk_id = agent.dict.tok2ind[agent.dict.unk_token]
vocab_size = len(agent.dict)

CrossEntropyLoss = nn.CrossEntropyLoss(reduction='mean')
WeightedCrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=null_id, reduction='mean')

# batchize the training data
train_gender_data = batchize(train_gender_data_list[:-5000])
train_neutral_data = batchize(train_neutral_data_list)

valid_gender_data_list = []
for text, _, _ in train_gender_data_list[-5000:]:
    if contain_word(text, male_to_female):
        valid_gender_data_list.append((text, ' '.join([w if w not in male_to_female else male_to_female[w] for w in text.split()])))
    elif contain_word(text, female_to_male):
        valid_gender_data_list.append((' '.join([w if w not in female_to_male else female_to_male[w] for w in text.split()]), text))

with open('../data/evaluation_corpus_twitter_30k.json', 'r') as f:
    test_gender_data_list = json.load(f)

D_biased_optim = optim.Adam(D_biased.parameters(), lr=lr)
G_optim = optim.Adam(generator.parameters(), lr=lr)
G_and_D_unbiased_optim = optim.Adam(get_params_of_two_modules(generator, D_unbiased), lr=lr)

min_valid_ppl = 1e6
max_passed = 0
patience = 0

it = 0
while True:

    if it % 200 == 0 and it > 0:
      # evaluate generator
        print("--------------------------------evaluate on valid data-------------------------------------------")

        valid_evaluation = evaluate(agent, valid_data_list, device)

        print("Validation results: {}".format(valid_evaluation))

        valid_fairness_evaluation = fairness_test(agent, valid_gender_data_list)
        passed = 0
        for m, p in valid_fairness_evaluation.items():
            if p >= 0.05:
                passed += 1

        if passed > max_passed or passed >= 5:
            print("New best results! Save the model!")
            max_passed = passed
            patience = 0

            if not os.path.exists('save_model/{}'.format(params)):
                os.mkdir('save_model/{}'.format(params))

            torch.save(generator.state_dict(), 'save_model/{}/model.pt'.format(params))
            torch.save(D_biased.state_dict(), 'save_model/{}/D_biased.pt'.format(params))
            torch.save(D_unbiased.state_dict(), 'save_model/{}/D_unbiased.pt'.format(params))
            print("Model saved to /save_model/{}/model.pt".format(params))


            print("--------------------------------evaluate on fairness test data-------------------------------------------")
            test_fairness_evaluation = fairness_test(agent, test_gender_data_list)


        else:
            if it > 0:
                patience += 1
                print('Patience: {}'.format(patience))

        if it > 0 and temp > 0.3:
            temp /= 1.1

    # Train on gender batch
    print("--------------------------------------------------------------------------------------------------")

    for _ in range(D_biased_steps):
        # train biased discriminator
        batch = random.choice(train_gender_data)

        # source: (batch_size x seq_len)
        # target: (batch_size x seq_len)
        # labels: (batch_size)
        source, target, labels = batch
        source = source.to(device)
        target = target.to(device)
        labels = labels.to(device)

        generator.eval()
        with torch.no_grad():
            out = generator.forward(source)

            # predictions: (batch_size x seq_len)
            # scores: (batch_size x seq_len x vocab_size)
            predictions = out[0]
            scores = out[1]

            # lens: (batch_size)
            lens = get_lengths(predictions)

            # predictions_oh: (batch_size x seq_len x vocab_size)
            predictions_oh = torch.tensor(F.one_hot(predictions, num_classes=vocab_size), dtype=torch.float32).to(
                device)

            # content_code: (batch_size x content_size)
            _, content_code = disen_model.forward_encoder_oh(predictions_oh, lens)

        # y_c: (batch_size x 2)
        y_c = D_biased(content_code.detach())

        preds_c = y_c.argmax(dim=1)
        # print(preds, labels)
        acc_c = torch.sum(preds_c == labels, dtype=torch.float32) / len(preds_c)

        loss = CrossEntropyLoss(y_c, labels)
        D_biased_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(D_biased.parameters(), clip)
        D_biased_optim.step()

        print("train D_biased   iter: {}  batch_size: {}  loss: {:.6f}  accuracy: {:.4f}".format(it, len(source),
                                                                                                 loss.item(), acc_c))

    for _ in range(G_adv_steps):
        # train generator & unbiased discriminator (gender)
        batch = random.choice(train_gender_data)

        # source: (batch_size x seq_len)
        # target: (batch_size x seq_len)
        # labels: (batch_size)
        source, target, labels = batch
        source = source.to(device)
        target = target.to(device)
        labels = labels.to(device)

        generator.train()
        out = generator.forward_gumbel_softmax(source, temp)

        # predictions: (batch_size x seq_len)
        # predictions_oh: (batch_size x seq_len x vocab_size)
        predictions, predictions_oh = out

        # lens: (batch_size)
        lens = get_lengths(predictions)

        # unbias_code: (batch_size x unbias_size)
        # content_code: (batch_size x content_size)
        unbias_code, content_code = disen_model.forward_encoder_oh(predictions_oh, lens)

        # y_u: (batch_size x 2)
        y_u = D_unbiased(unbias_code)
        # y_c: (batch_size x 2)
        y_c = D_biased(content_code)

        loss_u = CrossEntropyLoss(y_u, labels)
        loss_c = EntropyLoss(y_c)

        preds_u = y_u.argmax(dim=1)
        acc_u = torch.sum(preds_u == labels, dtype=torch.float32) / len(preds_u)

        preds_c = y_c.argmax(dim=1)
        acc_c = torch.sum(preds_c == labels, dtype=torch.float32) / len(preds_c)

        out = generator(source, target)
        predictions = out[0]
        scores = out[1]
        loss_real = WeightedCrossEntropyLoss(scores.view(-1, scores.size()[2]), target.view(-1, ))
        acc_real = accuracy(predictions, target)

        loss = k0 * loss_real + k1 * loss_u + k2 * loss_c

        G_and_D_unbiased_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(get_params_of_two_modules(generator, D_unbiased), clip)
        G_and_D_unbiased_optim.step()

        print(
            "train G & D_unbiased (gender)   iter: {}  batch_size: {}  temp: {}  loss_real: {:.5f}  loss_u: {:.5f}  loss_c: {:.5f}  acc_real: {:.4f}  acc_u: {:.4f}  acc_c: {:.4f}".format(
                it, len(source), temp, loss_real.item(), loss_u.item(), loss_c.item(), acc_real, acc_u, acc_c))

    for _ in range(G_teach_steps):
        # Train on neutral batch
        batch = random.choice(train_neutral_data)

        # source: (batch_size x seq_len)
        # target: (batch_size x seq_len)
        # labels: (batch_size)
        source, target, labels = batch
        source = source.to(device)
        target = target.to(device)
        labels = labels.to(device)

        # train generator (neutral)
        generator.train()
        out = generator(source, target)
        predictions = out[0]
        scores = out[1]
        loss_real = WeightedCrossEntropyLoss(scores.view(-1, scores.size()[2]), target.view(-1, ))
        acc_real = accuracy(predictions, target)

        loss = loss_real

        G_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), clip)
        G_optim.step()

        print("train G (neutral)   iter: {}  batch_size: {}  temp: {}  loss: {:.6f}  acc_real: {:.4f}".format(
            it, len(source), temp, loss.item(), acc_real))

    it += 1

    if patience >= 10:
        print("Out of patience!")
        break