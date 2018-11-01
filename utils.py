from collections import Counter

import numpy as np


def load_data(input_file, label_file = None):
    with open(input_file) as f:
        data = [map(int, x.split(' ')[:-1]) for x in f.read().split('\n')[:-1]]
    if label_file is not None:
        with open(label_file) as f:
            labels = f.read().split('\n')[:-1]
        return data, labels
    return data


def pad_sequences(sentences,maxlen=500,value=0):
    """
    Pads all sentences to the same length. The length is defined by maxlen.
    Returns padded sentences.
    """
    padded_sentences = []
    for sen in sentences:
        new_sentence = []
        if(len(sen) > maxlen):
            new_sentence = sen[:maxlen]
            padded_sentences.append(new_sentence)
        else:
            num_padding = maxlen - len(sen)
            new_sentence = np.append([value] * num_padding, sen)
            padded_sentences.append(new_sentence)
    return padded_sentences


def create_count(system_calls):
    counter = Counter()
    for calls in system_calls:
        for call in calls:
            if call not in counter.keys():
                counter[call] = 1
            else:
                counter[call] += 1
    return counter


def create_call2idx(call_counter):
    idx = 0
    call2idx = {}
    for call in call_counter.most_common():
        call2idx[call[0]] = idx
        idx+=1
    return call2idx


def evaluate_accuracy(model, data, y_target, batch_size):
    numCorrect = 0
    for b in range(data.shape[0] // batch_size):
        input = data[b * batch_size:(b * batch_size + batch_size), ]
        target = y_target[b * batch_size:(b * batch_size + batch_size), ]
        output = model.forward(input)
        numCorrect += np.sum(output.argmax(axis=1) == target)
    return numCorrect/float(data.shape[0])