import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('../wordembed/GoogleNews-vectors-negative300.bin', binary=True)
# if you vector file is in binary format, change to binary=True
sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
for word in sentence:
    try:
        print model[word].shape
    except:
        print word



