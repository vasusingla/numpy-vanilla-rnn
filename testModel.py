from utils import *
import cPickle as pkl
import argparse
import os
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-modelName', required=True, help='name of model to load')
    parser.add_argument('-data', required=True, help='path to test_data.txt')
    args = parser.parse_args()
    return args


args = parse_args()

def test_model(args):
    modelName = args.modelName
    data_path = args.data

    data_raw = load_data(data_path)

    with open(os.path.join(modelName, modelName+'.pkl'),'rb') as f:
        dump_data = pkl.load(f)

    model = dump_data['model']
    call2idx = dump_data['call2idx']
    VOCAB_SIZE = dump_data['VOCAB_SIZE']
    MAX_SEQUENCE_LENGTH = dump_data['MAX_SEQUENCE_LENGTH']

    data_converted = [[call2idx[call] if call in call2idx.keys() else -1 for call in calls] for calls in data_raw]

    t_data = [np.array([i if i < (VOCAB_SIZE - 1) else (VOCAB_SIZE - 1) for i in s]) for s in data_converted]
    test_data_final = np.array(pad_sequences(t_data, maxlen=MAX_SEQUENCE_LENGTH, value=-1), dtype=np.int32)
    test_pred = []
    for i in xrange(test_data_final.shape[0]):
        test_pred+=[model.forward(test_data_final[i, :].reshape((1, -1))).argmax(axis = 1).tolist()]

    dump_str = 'id,label\n'

    for i, pred in enumerate(test_pred):
        dump_str+=str(i)+','+str(pred[0])+'\n'
    dump_str = dump_str[:-1]

    with open('submission_file.txt', 'w') as f:
        f.write(dump_str)


if __name__ == '__main__':
    test_model(args)