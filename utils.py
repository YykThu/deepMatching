from configuration import *


config = gen_config_rn()
floatX = np.float32


# data_related
def text_to_wordlist(text, remove_stop_words=False, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.

    # Convert words to lower case and split them
    # text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"What's", "what is", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    #     text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    #     text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text)
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)
    text = re.sub(r"demonitization", "demonetization", text)
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    #     text = re.sub(r" cs ", " computer science ", text)
    text = re.sub(r" upvotes ", " up votes ", text)
    #     text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text)
    text = re.sub(r"calender", "calendar", text)
    #     text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text)
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text)
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in config['stop_words']]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return text


def process_questions(qs, prex):
    q_list = []
    for q in tqdm(qs, desc='Converting {} to list'.format(prex)):
        q_list.append(text_to_wordlist(q))
    return q_list


def to_word_list(question_list):
    word_list = map(lambda x: nltk.word_tokenize(x), question_list)
    return word_list


def load_data(path='../data/data.pkl'):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


# model_related
# #return prex + name
def pp(prex, basename):
    return '{prex}_{basename}'.format(prex=prex, basename=basename)


# #return params based on param type
def init_param(param_type, param_size, param_name, cf=config):
    if param_type == 'weight':
        dim_in = param_size[-1]
        param = theano.shared(value=np.random.uniform(low=-1/np.sqrt(dim_in),
                                                      high=1/np.sqrt(dim_in),
                                                      size=param_size).astype(dtype=cf['floatX']),
                              name=param_name)
    elif param_type == 'bias':
        param = theano.shared(value=np.zeros(shape=param_size).astype(dtype=cf['floatX']),
                              name=param_name)
    elif param_type == 'embedding_matrix':
        param = theano.shared(value=np.random.randn(param_size[0], param_size[1]).astype(dtype=cf['floatX']))
    else:
        raise TypeError('wrong param type')

    return param


# training_related
def gen_batch_index(train_num, batch_size):
    index_pairs = []
    max_index = 0
    while max_index < train_num:
        index_pairs.append((max_index, min(max_index + batch_size, train_num)))
        max_index += batch_size
    return index_pairs


def init_data_type(data):
    return np.array(data).astype(np.int32).T

if __name__ == '__main__':
    pass
