import os
import json
import random
import torch
import argparse
import importlib
from .utils import eval_distinct, calculate_ppl
from .pycocoevalcap.bleu.bleu import Bleu
from .fairness_test import bias_evaluate

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

def get_data_for_inference(dir):
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

def detect(agent, data_file):
    print("-------Inference on Fairness Test Corpus-------")
    with open(data_file, 'r') as f:
        contexts = json.load(f)

    f_left_out = open('results/' + args.dialog_model + '_left_results.txt', 'w')
    f_right_out = open('results/' + args.dialog_model + '_right_results.txt', 'w')

    batch_size = 100
    n_batch = len(contexts) // batch_size

    count = 0
    for i in range(n_batch):
        context_batch = contexts[i * batch_size: (i + 1) * batch_size]
        left_context_batch = [c[0] for c in context_batch]
        right_context_batch = [c[1] for c in context_batch]
        left_response_batch = get_response(agent, left_context_batch)
        right_response_batch = get_response(agent, right_context_batch)

        for left_context, left_response, right_context, right_response in zip(left_context_batch, left_response_batch,
                                                                              right_context_batch,
                                                                              right_response_batch):

            if count % 50 == 0:
                print("{} / {}".format(count, n_batch * batch_size))
            if count % 1000 == 0:
                print("context: {}".format(left_context))
                print("left response: {}".format(left_response))

                print("context: {}".format(right_context))
                print("right response: {}".format(right_response))
                print(
                    "-------------------------------------------------------------------------------------------------------")

            f_left_out.write(left_context + '\t' + left_response + '\n')
            f_right_out.write(right_context + '\t' + right_response + '\n')

            count += 1

def evaluate(agent, data, bias_data, device):
    '''

    :param agent: agent
    :param data: list of (text, response) str
    :param bias_data: (list of left_lines, list of right_lines)
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
    # total_loss = 0.
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

        if i == chosen:
            for context, response, predicted in zip(context_batch, response_batch, predicted_batch):
                print("Context: {}".format(context))
                print("Ground Truth: {}".format(response))
                print("Response: {}".format(predicted))
                print("--------------------------------------------------------------")

        # xs, ys: (batch_size x seq_len)
        xs = pad([[agent.dict.tok2ind.get(word, unk_id) for word in context.split()] for context in context_batch], padding=null_id)
        ys = pad([[agent.dict.tok2ind.get(word, unk_id) for word in response.split()] for response in response_batch], padding=null_id)

        xs = xs.to(device)
        ys = ys.to(device)

        out = agent.model(xs, ys)
        preds = out[0]
        scores = out[1]
        score_view = scores.view(-1, scores.size(-1))
        y_ne = ys.ne(null_id)
        target_tokens = y_ne.long().sum().item()
        correct = ((ys == preds) * y_ne).sum().item()
        # total_loss += loss.item()
        correct_tokens += correct
        num_tokens += target_tokens

    scores = score(response_list, predicted_list)

    token_acc = correct_tokens / num_tokens


    metrics = scores
    metrics['token_acc'] = token_acc

    bias_evaluate(bias_data)

    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--dialog_model', type=str, default=None,
                        help='options: Seq2Seq, AugSeq2Seq, RegSeq2Seq, OurSeq2Seq')
    args = parser.parse_args()
    device = torch.device("cuda")
    # create generator.
    module = importlib.import_module("ParlAI" + args.dialog_model + "Twitter" + ".agent")
    agent = getattr(module, "agent")
    agent.model.longest_label = 30

    test_data_list = get_data_for_inference('/mnt/home/liuhaoc1/ParlAI/data/Twitter/test.txt')

    if not os.path.exists('results/' + args.dialog_model + '_left_results.txt'):
        detect(agent, 'corpus/evaluation_corpus_twitter_30k.json')

    f_left = open('results/' + args.dialog_model + '_left_results.txt', 'r')
    left_lines = f_left.readlines()
    f_right = open('results/' + args.dialog_model + '_right_results.txt', 'r')
    right_lines = f_right.readlines()

    bias_data = (left_lines, right_lines)

    results = evaluate(agent, test_data_list, bias_data, device)

    print(args.dialog_model)
    print(results)