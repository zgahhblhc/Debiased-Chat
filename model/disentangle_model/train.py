import math
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from .model import Autoencoder

def pad(list, padding=0, min_len=None):
    padded = []
    max_len = max([len(l) for l in list])
    if min_len:
        max_len = max(min_len, max_len)
    for l in list:
        padded.append(l + [padding] * (max_len - len(l)))
    # print(padded)

    return torch.tensor(padded, dtype=torch.long)

def EntropyLoss(tensor):
    '''

    :param tensor: (batch_size x m)
    :return:
    '''
    # print("tensor: ", tensor)

    p = nn.functional.softmax(tensor, dim=1)
    # print("p: ", p)
    log_p = torch.log(p + 1e-10)
    # print("log_p: ", log_p)

    return torch.mean(torch.sum(p * log_p, dim=1))

def accuracy(predictions, truths):
    pads = torch.Tensor.double(truths != 0)
    corrects = torch.Tensor.double(predictions == truths)
    valid_corrects = corrects * pads

    return valid_corrects.sum() / pads.sum()

def eval_loss(dists, text, pad_token):
    loss = 0
    num_tokens = 0

    for dist, y in zip(dists, text):
        y_len = sum([1 if y_i != pad_token else 0 for y_i in y])
        for i in range(y_len):
            loss -= torch.log(dist[i][y[i]])
            num_tokens += 1

    return loss, num_tokens

def bow_loss(y_bow, bow_labels):
    '''

    :param y_bow: (batch_size x vocab_size)
    :param bow_labels: (batch_size x vocab_size)
    :return:
    '''

    y_bow = nn.functional.softmax(y_bow, dim=1)

    # loss_for_each_batch: (batch_size)
    loss_for_each_batch = torch.sum(- bow_labels * torch.log(y_bow + 1e-10), dim=1)

    return loss_for_each_batch.mean()

def get_bow(batch):
    '''

    :param batch: (batch_size x seq_len)
    :return: (batch_size x vocab_size)
    '''

    bows = []
    for text in batch:
        bow = torch.zeros(vocab_size)
        count = {}
        for i in text:
            if i != 0 and i not in remove_words_id:
                if i not in count:
                    count[i] = 1
                else:
                    count[i] += 1
        len = sum([c for w, c in count.items()])
        # print(count)
        # print('\n\n')
        for w, c in count.items():
            bow[w] = c / len
        bows.append(bow)

    return torch.stack(bows)

stop_words = stopwords.words('english')
gender_words = []
with open('gender_words', 'r') as f:
    word_lines = f.readlines()
for line in word_lines:
    male_word, female_word = line.strip().split(' - ')
    gender_words.append(male_word)
    gender_words.append(female_word)
remove_words = list(set(stop_words + gender_words))


device = torch.device("cuda")

opt = {'emb_size': 300, 'hidden_size': 1000, 'unbias_size': 200, 'content_size': 800, 'rnn_class': nn.GRU, 'dropout': 0.0}

print(opt)

model = Autoencoder(emb_size=opt['emb_size'], hidden_size=opt['hidden_size'], unbias_size=opt['unbias_size'], content_size=opt['content_size'],
                    dict_file='/mnt/home/liuhaoc1/original_ParlAI_twitter_seq2seq_model/twitter_seq2seq_model.dict',
                    dropout=opt['dropout'], rnn_class=opt['rnn_class'], device=device).to(device)

n_epoch = 20
batch_size = 32
lr = 0.001
clip = 0.25
k0 = 1
k1 = 10
k2 = 1
k3 = 1
k4 = 3

null_id = model.dict.tok2ind[model.dict.null_token]
eos_id = model.dict.tok2ind[model.dict.end_token]
unk_id = model.dict.tok2ind[model.dict.unk_token]
vocab_size = len(model.dict)
remove_words_id = set([model.dict.tok2ind.get(word, unk_id) for word in remove_words])

CrossEntropyLoss = nn.CrossEntropyLoss(reduce=True, reduction='mean')
padding_weights = torch.ones(vocab_size).to(device)
padding_weights[null_id] = 0
WeightedCrossEntropyLoss = nn.CrossEntropyLoss(weight=padding_weights, reduce=True, reduction='mean')

with open('../unbias_corpus.json', 'r') as f:
    data_list = json.load(f)
    train_data_list = data_list[:-5000]
    valid_data_list = data_list[-5000:]

n = 10
train_data = []
for i in range(len(train_data_list) // (n * batch_size) + 1):
    nbatch = train_data_list[i * (n * batch_size) : (i+1) * (n * batch_size)]
    nbatch_list = [([model.dict.tok2ind.get(word, unk_id) for word in ins[0].split()],
                    ins[1]) for ins in nbatch]
    descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

    j = 0
    while len(descend_nbatch_list[j * batch_size : (j+1) * batch_size]) > 0:
        batch_list = descend_nbatch_list[j * batch_size : (j+1) * batch_size]
        # text: (batch_size x seq_len)
        text = pad([x[0] for x in batch_list], padding=null_id)
        labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)
        train_data.append((text, labels))
        j += 1

valid_data = []
for i in range(len(valid_data_list) // (n * batch_size) + 1):
    nbatch = valid_data_list[i * (n * batch_size) : (i+1) * (n * batch_size)]
    nbatch_list = [([model.dict.tok2ind.get(word, unk_id) for word in ins[0].split()],
                    ins[1]) for ins in nbatch]
    descend_nbatch_list = sorted(nbatch_list, key=lambda x: len(x[0]), reverse=True)

    j = 0
    while len(descend_nbatch_list[j * batch_size : (j+1) * batch_size]) > 0:
        batch_list = descend_nbatch_list[j * batch_size : (j+1) * batch_size]

        # text: (batch_size x seq_len)
        text = pad([x[0] for x in batch_list], padding=null_id)
        labels = torch.tensor([x[1] for x in batch_list], dtype=torch.long)
        valid_data.append((text, labels))
        j += 1

c_classifier_gender_optim = optim.RMSprop(model.get_c_classifier_gender_params(), lr=lr)
u_classifier_bow_optim = optim.RMSprop(model.get_u_classifier_bow_params(), lr=lr)
overall_optim = optim.Adam(model.get_rest_parameters(), lr=lr)

# min_ppl = 1e6
# patience = 0

for epoch in range(n_epoch):
    for i_batch, batch in enumerate(train_data):
        # text: (batch_size x seq_len)
        # labels: (batch_size)
        text, labels = batch
        text = text.to(device)
        labels = labels.to(device)

        # bow_labels: (batch_size x vocab_size)
        bow_labels = get_bow(text)
        bow_labels = bow_labels.to(device)

        model.train()

        # train c_classifier_gender
        _, y_c_gender, _, _ = model.forward_encoder(text)
        c_loss = CrossEntropyLoss(y_c_gender, labels)
        if torch.isnan(c_loss):
            print("c_loss NAN.")
            print("y_c_gender, labels: ", y_c_gender, labels)
            continue

        c_classifier_gender_optim.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_c_classifier_gender_params(), clip)
        c_classifier_gender_optim.step()

        c_preds = y_c_gender.argmax(dim=1)
        c_acc = torch.sum(c_preds == labels, dtype=torch.float32) / len(c_preds)

        if i_batch % 10 == 0:
            print(
            "train c_classifier_gender epoch: {}  batch: {} / {}  batch_size: {}  c_loss: {:.6f}  acc: {:.4f}".format(
                epoch, i_batch, len(train_data), len(text), c_loss.item(), c_acc))

        # train u_classifier_bow
        _, _, y_u_bow, _ = model.forward_encoder(text)
        u_loss = bow_loss(y_u_bow, bow_labels)
        if torch.isnan(u_loss):
            print("u_loss NAN.")
            print("y_u_bow, bow_labels: ", y_u_bow, bow_labels)
            continue

        u_classifier_bow_optim.zero_grad()
        u_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_u_classifier_bow_params(), clip)
        u_classifier_bow_optim.step()


        if i_batch % 10 == 0:
            print(
                "train u_classifier_bow epoch: {}  batch: {} / {}  batch_size: {}  u_loss: {:.6f}".format(
                    epoch, i_batch, len(train_data), len(text), u_loss.item()))


        # train autoencoder
        preds, scores, y_u_gender, y_c_gender, y_u_bow, y_c_bow = model(text, train=True)
        reconstruction_loss = WeightedCrossEntropyLoss(scores.view(-1, scores.size()[2]), text.view(-1,))
        u_gender_loss = CrossEntropyLoss(y_u_gender, labels)
        c_gender_loss = EntropyLoss(y_c_gender)

        u_bow_loss = EntropyLoss(y_u_bow)
        c_bow_loss = bow_loss(y_c_bow, bow_labels)

        loss = k0 * reconstruction_loss + k1 * u_gender_loss + k2 * c_gender_loss + k3 * u_bow_loss + k4 * c_bow_loss

        if torch.isnan(loss):
            print("loss NAN.")
            print(reconstruction_loss, u_gender_loss, c_gender_loss, u_bow_loss, c_bow_loss)
            continue

        overall_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_rest_parameters(), clip)
        overall_optim.step()

        re_acc = accuracy(preds, text)
        u_preds = y_u_gender.argmax(dim=1)
        u_acc = torch.sum(u_preds == labels, dtype=torch.float32) / len(u_preds)
        c_preds = y_c_gender.argmax(dim=1)
        c_acc = torch.sum(c_preds == labels, dtype=torch.float32) / len(c_preds)

        if i_batch % 10 == 0:
            print(
            "train autoencoder epoch: {}  batch: {} / {}  batch_size: {}  loss: {:.6f}  re_acc: {:.4f}  u_acc: {:.4f}  c_acc: {:.4f}".format(
                epoch, i_batch, len(train_data), len(text), loss.item(), re_acc, u_acc, c_acc))
            print("re_loss: {}  u_gender_loss: {}  c_gender_loss: {} u_bow_loss: {} c_bow_loss: {}".format(reconstruction_loss.item(), u_gender_loss.item(), c_gender_loss.item(), u_bow_loss.item(), c_bow_loss.item()))

        if i_batch % 1000 == 0:
            print("-----------------------------Validation-----------------------------")
            model.eval()
            total_loss = 0
            total_num_tokens = 0
            u_correct_num = 0.0
            c_correct_num = 0.0
            u_num = 0.0
            c_num = 0.0

            r = random.randint(0, len(valid_data) - 1)

            for i_batch, batch in enumerate(valid_data):
                # text: (batch_size x seq_len)
                # labels: (batch_size)
                text, labels = batch
                text = text.to(device)
                labels = labels.to(device)

                # preds: (batch_size x seq_len)
                # scores: (batch_size x seq_len x vocab_size)
                preds, scores, y_u_gender, y_c_gender, y_u_bow, y_c_bow = model(text, train=True)
                dists = nn.functional.softmax(scores, dim=2)

                loss, num_tokens = eval_loss(dists, text, pad_token=null_id)
                total_loss += loss.item()
                total_num_tokens += num_tokens

                # re_acc = accuracy(preds, text)
                u_preds = y_u_gender.argmax(dim=1)
                u_correct_num += torch.sum(u_preds == labels, dtype=torch.float32)
                u_num += len(u_preds)

                c_preds = y_c_gender.argmax(dim=1)
                c_correct_num += torch.sum(c_preds == labels, dtype=torch.float32)
                c_num += len(c_preds)

                if i_batch == r:
                    for truth, pred in zip(text, preds):
                        truth_null_id_list = [i for i in range(len(truth)) if truth[i] == null_id]
                        pred_eos_id_list = [i for i in range(len(pred)) if pred[i] == eos_id]

                        truth_null_id = len(truth) if len(truth_null_id_list) == 0 else truth_null_id_list[0]
                        pred_eos_id = len(pred) if len(pred_eos_id_list) == 0 else pred_eos_id_list[0]

                        print('truth: {}'.format(
                            ' '.join([model.dict.ind2tok.get(idx, '__unk__') for idx in truth[:truth_null_id].tolist()])))
                        print('pred: {}'.format(
                            ' '.join([model.dict.ind2tok.get(idx, '__unk__') for idx in pred[:pred_eos_id].tolist()])))
                        print('------------------------------------------------------------')

            ave_loss = total_loss / total_num_tokens
            ppl = math.exp(ave_loss)

            print("Validation ppl: {}  u_acc: {}  c_acc: {}".format(ppl, u_correct_num / u_num, c_correct_num / c_num))

            torch.save(model.state_dict(), 'save_model/model_with_bow.pt')
            print("Model saved to save_model/model_with_bow.pt")
            print("Model performance: ppl_{:.4f} u_acc_{:.4f} c_acc_{:.4f}".format(ppl, u_correct_num / u_num,
                                                                                   c_correct_num / c_num))