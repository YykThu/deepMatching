from utils import *


print 'Loading Raw Data...'
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')
print 'train data shape: {}'.format(train_data.shape)
print 'test data shape: {}'.format(test_data.shape)

print 'Filling Null with "empty"...'
train_data = train_data.fillna('empty')
test_data = test_data.fillna('empty')

print 'Converting Strings to Word List...'
train_q1_word_list = process_questions(train_data.question1, prex='train q1s')
train_q2_word_list = process_questions(train_data.question2, prex='train q2s')
test_q1_word_list = process_questions(test_data.question1, prex='test q1s')
test_q2_word_list = process_questions(test_data.question2, prex='test q2s')

print 'Building Dictionary...'
train_q1_words = map(nltk.word_tokenize, train_q1_word_list)
train_q2_words = map(nltk.word_tokenize, train_q2_word_list)
test_q1_words = map(nltk.word_tokenize, test_q1_word_list)
test_q2_words = map(nltk.word_tokenize, test_q2_word_list)
words_list = train_q1_words + train_q2_words + test_q1_words + test_q2_words

words_ = []
for list_ in words_list:
    for word in list_:
        words_.append(word)

words_freq = nltk.FreqDist(words_)

words = []
for word in words_freq:
    if words_freq[word] > config['remain_freq']:
        words.append(word)
print '* remain frequence: {}, words num: {}'.format(config['remain_freq'], len(words))

word2id = OrderedDict()
id2word = OrderedDict()

for id_, word in enumerate(words):
    word2id[word] = id_
    id2word[id_] = word

word2id['UNW'] = len(words)
id2word[len(words)] = 'UNK'

data = [train_q1_words,
        train_q2_words,
        test_q1_words,
        test_q2_words,
        word2id,
        id2word,
        words]

f = open('../data/data.pkl', 'wb')
pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
