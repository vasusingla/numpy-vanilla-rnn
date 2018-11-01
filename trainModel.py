import cPickle as pkl
from sklearn.model_selection import train_test_split
from Criterion import *
from Model import Model
from utils import load_data, pad_sequences, create_count, create_call2idx, evaluate_accuracy
import numpy as np
import argparse
import os

np.random.seed(123)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelName', required=True, help='name of model to dump')
    parser.add_argument('-data', required=True, help='path to train_data.txt')
    parser.add_argument('-target', required=True, help='path to train_labels.txt')
    args = parser.parse_args()
    return args


args = parse_args()


def train_model(args):

    # HYPERPARAMETER CONFIGURATIONS
    # TODO - Move configurations to a config file instead, maybe?

    VOCAB_SIZE = 153  # Size of vocabulary
    EMBEDDING_SIZE = 64
    INPUT_DIMENSION = EMBEDDING_SIZE
    HIDDEN_DIMENSION = 128
    BATCH_SIZE = 10
    MAX_SEQUENCE_LENGTH = 1500  # Sequences larger than specified length will be truncated and smaller ones left padded #
    EPOCHS = 1
    LEARNING_RATE = 0.1
    LR_DECAY_EPOCH = 30  # Decay learning rate by LR_DECAY_FACTOR after these epochs
    LR_DECAY_FACTOR = 5
    NUM_LAYERS = 1  # Number of RNN Layers

    modelName = args.modelName
    data_path = args.data
    target_path = args.target

    modelDumpPath = modelName

    ### LOAD AND PREPROCESS DATA
    # Encode calls to indexes according to
    data_raw, labels = load_data(data_path, target_path)

    calls_counter = create_count(data_raw)
    call2idx = create_call2idx(calls_counter)
    idx2call = {v:k for k,v in call2idx.iteritems()}

    data_converted = [[call2idx[call] for call in calls] for calls in data_raw]

    t_data = [np.array([i if i < (VOCAB_SIZE - 1) else (VOCAB_SIZE - 1) for i in s]) for s in data_converted]

    X_train, X_test, y_train, y_test_set = train_test_split(t_data, labels, test_size=0.3)

    trn = np.array(pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=-1), dtype=np.int32)
    test = np.array(pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, value=-1), dtype=np.int32)
    y_trn = np.array(map(int, y_train))
    y_test = np.array(map(int, y_test_set))

    # Model and loss defined

    model = Model(NUM_LAYERS, HIDDEN_DIMENSION, VOCAB_SIZE, EMBEDDING_SIZE)
    loss = Criterion()

    ### SGD START
    for epoch in range(1, EPOCHS+1):
        totalLoss = 0
        numCorrect_train = 0
        if (epoch%LR_DECAY_EPOCH==0):
            LEARNING_RATE = LEARNING_RATE/LR_DECAY_FACTOR
        for b in range(trn.shape[0] // BATCH_SIZE):
            data = trn[b * BATCH_SIZE:(b * BATCH_SIZE + BATCH_SIZE), ]
            target = y_trn[b * BATCH_SIZE:(b * BATCH_SIZE + BATCH_SIZE), ]
            output = model.forward(data)
            batch_loss = loss.forward(output, target)
            gradOut = loss.backward(output, target)
            model.backward(data, gradOut)
            model.update_parameters(LEARNING_RATE)
            totalLoss += batch_loss
            numCorrect_train+=np.sum(target==output.argmax(axis = 1))
        test_accuracy = evaluate_accuracy(model, trn, y_trn, BATCH_SIZE)
        train_accuracy = evaluate_accuracy(model, test, y_test, BATCH_SIZE)
        print("Epoch %s. Train_loss %s Train_Accuracy %s Test_Accuracy %s" %
              (epoch, totalLoss/(trn.shape[0]//BATCH_SIZE), train_accuracy, test_accuracy))

    # Dumping the trained model, encoding call2idx which encodes data to embedding format and the model class itself.

    dump_dict = {}
    dump_dict['call2idx'] = call2idx
    dump_dict['model'] = model
    dump_dict['VOCAB_SIZE'] = VOCAB_SIZE
    dump_dict['MAX_SEQUENCE_LENGTH'] = MAX_SEQUENCE_LENGTH

    if not os.path.exists(modelDumpPath):
        os.makedirs(modelDumpPath)
    with open(os.path.join(modelDumpPath, modelName+'.pkl'), 'wb') as f:
        pkl.dump(dump_dict, f)


if __name__ == "__main__":
    train_model(args)