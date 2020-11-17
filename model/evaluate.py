import math
import random
import torch
from .pycocoevalcap.bleu.bleu import Bleu

def eval_ppl(model, xs, ps, ys, pad_token):
    loss = 0
    num_tokens = 0

    predictions, logits = model.forward(xs, ps, ys)  # predictions: (batch_size x seq_len) logits: (batch_size x seq_len x vocab_size)
    for logit, y in zip(logits, ys):
        y_len = sum([1 if y_i != pad_token else 0 for y_i in y])
        for i in range(y_len):
            loss -= logit[i][y[i]]
            num_tokens += 1

    return loss, num_tokens

def calculate_ppl(model, data, device, pad_token):
    total_loss = 0
    total_num_tokens = 0

    for batch in data:
        xs, ps, ys, aspects, _ = batch
        ys = ys.to(device)

        loss, num_tokens = eval_ppl(model, xs, ps, ys, pad_token)

        total_loss += loss
        total_num_tokens += num_tokens

    ave_loss = total_loss / total_num_tokens
    ppl = math.exp(ave_loss)

    return ave_loss, ppl

def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)


def eval_distinct(hypos):
    """
    compute distinct score for the hyps_resp
    :param hypos: instance_num x 1 x str
    :return: average distinct score for 1, 2-gram
    """
    if len(hypos) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hypos[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hypos[0])))
        return

    hyps_resp = [hypo[0].split() for hypo in hypos]
    num_tokens = sum([len(i) for i in hyps_resp])
    gram_1 = count_ngram(hyps_resp, 1)
    gram_2 = count_ngram(hyps_resp, 2)
    print("num_tokens: ", num_tokens)
    print("1grams: ", gram_1)
    print("2grams: ", gram_2)
    dist1 = gram_1 / float(num_tokens)
    dist2 = gram_2 / float(num_tokens)

    return {"distinct-1": dist1, "distinct-2": dist2}


def score(refs, hypos):
    '''
    :param refs: instance_num x refer_num x str
    :param hypos: instance_num x 1 x str
    :return:
    '''
    scorers = [
        (Bleu(3), ["Bleu_1", "Bleu_2", "Bleu_3"])]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(refs, hypos)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score

    final_scores["Distinct"] = eval_distinct(hypos)

    return final_scores

def get_data_for_inference(dir, withlabel=False):
    with open(dir, 'r') as f:
        raw_data = f.readlines()

    idx = 0
    data_list = []
    for line in raw_data:
        idx += 1
        text, response, _ = line.strip().split('\t')
        assert text.startswith('text:') and response.startswith('labels:')
        text = text[5:]
        response = response[7:]

        if withlabel:
            data_list.append((text, response, 2))
        else:
            data_list.append((text, response))

    return data_list

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


def pad(list, padding=0, min_len=None):
    padded = []
    max_len = max([len(l) for l in list])
    if min_len:
        max_len = max(min_len, max_len)
    for l in list:
        padded.append(l + [padding] * (max_len - len(l)))
    # print(padded)

    return torch.tensor(padded, dtype=torch.long)


def evaluate(agent, data, device):
    '''

    :param agent: agent
    :param data: list of (text, response) str
    :return:
    '''
    agent.model.eval()

    null_id = agent.dict.tok2ind[agent.dict.null_token]
    eos_id = agent.dict.tok2ind[agent.dict.end_token]
    unk_id = agent.dict.tok2ind[agent.dict.unk_token]

    batch_size = 50
    n_batch = len(data) // batch_size
    response_list, predicted_list = [], []
    chosen = random.randint(0, n_batch - 1)
    total_loss = 0.
    correct_tokens = 0.
    num_tokens = 0.

    for i in range(n_batch):
        batch = data[i * batch_size: (i + 1) * batch_size]
        context_batch = [c[0] for c in batch]
        response_batch = [c[1] for c in batch]

        # predicted_batch: list of strings
        predicted_batch = get_response(agent, context_batch)

        response_list += [[response] for response in response_batch]
        predicted_list += [[predicted] for predicted in predicted_batch]

        # xs, ys: (batch_size x seq_len)

        xs = pad([[agent.dict.tok2ind.get(word, unk_id) for word in context.split()] for context in context_batch], padding=null_id)
        ys = pad([[agent.dict.tok2ind.get(word, unk_id) for word in response.split()] for response in response_batch], padding=null_id)

        xs = xs.to(device)
        ys = ys.to(device)

        out = agent.model(xs, ys)
        preds = out[0]
        scores = out[1]
        score_view = scores.view(-1, scores.size(-1))
        loss = agent.criterion(score_view, ys.view(-1))
        y_ne = ys.ne(null_id)
        target_tokens = y_ne.long().sum().item()
        correct = ((ys == preds) * y_ne).sum().item()
        total_loss += loss.item()
        correct_tokens += correct
        num_tokens += target_tokens


    scores = score(response_list, predicted_list)

    token_acc = correct_tokens / num_tokens
    ave_loss = total_loss / num_tokens
    ppl = math.exp(ave_loss)


    metrics = scores
    metrics['token_acc'] = token_acc
    metrics['ppl'] = ppl

    return metrics