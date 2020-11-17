import math

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