import re
import os
from collections import defaultdict
from random import shuffle


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def split_dataset(input_file, output_dir=None, max_examples=5000):
    """
    Splits a dataset in three separate files: train, dev and test
    :return: 
    """
    outcome2label = {'1 leg bye,': 'leg_bye',
                     '1 run,': 'single',
                     '1 wide,': 'wide',
                     '2 runs,': 'double',
                     '3 runs,': 'triple',
                     'FOUR,': 'four',
                     'OUT,': 'out',
                     'SIX,': 'six',
                     'no run,': 'none'}
    examples = defaultdict(list)
    with open(input_file) as inp_f:
        for line in inp_f:
            label, text = line.strip().split('\t')
            examples[outcome2label[label]].append(text)
    for k in examples.keys():
        with open(os.path.join("commentary_data", k), 'w') as key_file:
            samples = examples[k]
            shuffle(samples)
            key_file.write('\n'.join(samples))
