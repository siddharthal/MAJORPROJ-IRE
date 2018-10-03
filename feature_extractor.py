import pandas
import itertools
from collections import Counter

def load_training_data(train_path,essay_set = 1):
    train_path = train_path
    training_data = pandas.read_excel(train_path, delimiter='\t')
    resolved_score = training_data[training_data['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_data[training_data['essay_set'] == essay_set]['essay_id']
    essays = training_data[training_data['essay_set'] == essay_set]['essay']
    essay_list = []
    for idx, essay in essays.iteritems():
        essay_list.append(re.split(r'\s+|[,;.-/!?()]\s*', essay.lower()))
    return essay_list, resolved_score.tolist(), essay_ids.tolist()


def build_vocab(sentences, vocab_limit):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_limit)]
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}
    return vocabulary

if __name__ == '__main__':
    essay_set_id = 1
    essay_list, resolved_scores, essay_id = load_training_data('/home/user/Downloads/all/training_set_rel3.xls',essay_set_id)
    max_score = max(resolved_scores)
    min_score = min(resolved_scores)
    if essay_set_id == 7:
        min_score, max_score = 0, 30
    elif essay_set_id == 8:
        min_score, max_score = 0, 60
    score_range = range(int(min_score),int(max_score+1))
    word_idx = build_vocab(essay_list, 2500)
    print(word_idx)




