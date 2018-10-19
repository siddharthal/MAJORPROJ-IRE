import pandas
import itertools
import re
import os
from collections import Counter

def load_training_data(train_path,essay_set = 1):
    train_path = train_path
    training_data = pandas.read_excel(train_path, delimiter='\t')
    resolved_score = training_data[training_data['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_data[training_data['essay_set'] == essay_set]['essay_id']
    essays = training_data[training_data['essay_set'] == essay_set]['essay']
    essay_list = []
    for idx, essay in essays.iteritems():
        essay_list.append(clean_tokenize(essay))
    return essay_list, resolved_score.tolist(), essay_ids.tolist()

def clean_tokenize(data):
    data = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", data)
    data = re.sub(r"\'s", " \'s", data)
    data = re.sub(r"\'ve", " \'ve", data)
    data = re.sub(r"n\'t", " n\'t", data)
    data = re.sub(r"\'re", " \'re", data)
    data = re.sub(r"\'d", " \'d", data)
    data = re.sub(r"\'ll", " \'ll", data)
    data = re.sub(r",", " , ", data)
    data = re.sub(r"!", " ! ", data)
    data = re.sub(r"\(", " ( ", data)
    data = re.sub(r"\)", " ) ", data)
    data = re.sub(r"\?", " ? ", data)
    data = re.sub(r"\s{2,}", " ", data)
    return [x.strip() for x in re.split('(\W+)?', data) if x.strip()]

@property
def score_range(self):
    return {"1": (2, 12),"2": (1, 6),"3": (0, 3),"4": (0, 3),"5": (0, 4),"6": (0, 4),"7": (0, 30),"8": (0, 60)}

 def normalize_score(self, essay_set_id, score):
    lo, hi = self.score_range[str(essay_set_id)]
    score = float(score)
    return (score - lo) / (hi - lo)

def convertword2vec(dim=100):
    word2vec = []
    word_idx = {}
    word2vec.append([0]*dim)
    count = 1
    with open('glove.6B.100d.txt') as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = map(float, l[1:])
            word_idx[word] = count
            word2vec.append(vector)
            count += 1
    return word_idx, word2vec

def vectorize_data(data, word_idx, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        for w in essay:
            if w in word_idx:
                wl.append(word_idx[w])
            else:
                wl.append(0)
        wl += [0]*ls
        E.append(wl)
    return E

if __name__ == '__main__':
    essay_set_id = 1
    essay_list, resolved_scores, essay_id = load_training_data('/home/user/Downloads/all/training_set_rel3.xls',essay_set_id)
    word_idx, word2vec = convertword2vec(300)
    vocab_size = len(word_idx) + 1
    print vocab_size
    sent_size_list = map(len, [essay for essay in essay_list])
    max_sent_size = max(sent_size_list)
    vectorized_data = vectorize_data(essay_list, word_idx, max_sent_size)
    labeled_data = zip(vectorized_data, resolved_scores, sent_size_list)

    