from component import *
from theano.tensor import signal


class Encoder(object):
    def __init__(self, dim_in, dim_encode, embed, cf, prex='Encoder'):
        self.dim_in = dim_in
        self.dim_encode = dim_encode
        self.prex = prex
        self.embed = embed
        self.cf = cf
        self._init_params()

    def _init_params(self):
        self.lstml = LSTM(dim_in=self.dim_in,
                          dim_hid=self.dim_encode,
                          embed=self.embed,
                          prex=pp(self.prex, 'lstml'))
        self.lstmr = LSTM(dim_in=self.dim_in,
                          dim_hid=self.dim_encode,
                          embed=self.embed,
                          prex=pp(self.prex, 'lstmr'))
        self.params = self.lstml.params + self.lstmr.params

    def forward(self, id_sequence, mask_sequence):
        batch_size = id_sequence.shape[1]
        n_steps = id_sequence.shape[0]
        id_sequence_reverse = id_sequence[::-1]
        mask_sequence_reverse = mask_sequence[::-1]
        out_info = [T.alloc(np.float32(0.), batch_size, self.dim_encode),
                    T.alloc(np.float32(0.), batch_size, self.dim_encode)]
        [hl, _], _ = theano.scan(fn=self.lstml.step_forward_emb,
                                 sequences=[id_sequence, mask_sequence],
                                 outputs_info=out_info,
                                 n_steps=n_steps)
        [hr, _], _ = theano.scan(fn=self.lstmr.step_forward_emb,
                                 sequences=[id_sequence_reverse, mask_sequence_reverse],
                                 outputs_info=out_info,
                                 n_steps=n_steps)

        temp_encode = T.concatenate([hl, hr], axis=2)
        temp_encode = T.max(temp_encode, axis=0)
        return temp_encode


class CompModel(object):

    def __init__(self, dim_in, dim_hid, cf, prex='CompModel'):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.cf = cf
        self.prex = prex
        self._init_params()

    def _init_params(self):
        self.fc1 = FC(dim_in=self.dim_in,
                      dim_out=self.dim_hid,
                      prex=pp(self.prex, 'layer1'))
        self.fc2 = FC(dim_in=self.dim_hid,
                      dim_out=1,
                      prex=pp(self.prex, 'sigmoid_layer2'))
        self.params = self.fc1.params + self.fc2.params

    def forward(self, inputs):
        hidden = self.fc1.step_forward(inputs)
        prob = self.fc2.step_forward(hidden)
        return prob


class ScoreModel(object):

    def __init__(self, dim_in, dim_word, dim_hid, embed, cf, prex='ScoreModel'):
        self.dim_in = dim_in
        self.dim_word = dim_word
        self.dim_hid = dim_hid
        self.embed = embed
        self.cf = cf
        self.prex = prex
        self._init_params()

    def _init_params(self):
        self.fc1 = FC(dim_in=self.dim_in + self.dim_word,
                      dim_out=self.dim_hid,
                      prex=pp(self.prex, 'fc_layer1'))
        self.fc2 = FC(dim_in=self.dim_hid,
                      dim_out=1,
                      prex=pp(self.prex, 'fc_layer2'))
        self.params = self.fc1.params + self.fc2.params

    def forward(self, share_word, q1_encode, q2_encode):
        share_word_embed = self.embed.apply(share_word)
        x = T.concatenate([q1_encode, q2_encode, share_word_embed], axis=1)
        hidden = self.fc1.step_forward(x)
        score = self.fc2.step_forward(hidden)
        return score

    def generate_word_vector(self, scores, share_word):
        share_word_embed = self.embed.apply(share_word)
        scores = scores.dimshuffle(0, 'x')
        scores = T.repeat(scores, self.dim_word, axis=1)
        word_vector = T.sum(scores * share_word_embed, axis=0)
        return word_vector


class CnnEncoder(object):

    def __init__(self, dim_in, dim_out, embed, cf, prex='CnnEncoder'):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.embed = embed
        self.cf = cf
        self.prex = prex


if __name__ == '__main__':
    id_seq = T.imatrix()
    mask_seq = T.imatrix()
    embed = Embed(config)
    encoder = Encoder(dim_in=300, dim_encode=128, embed=embed, cf=config)
    hs = encoder.forward(id_seq, mask_seq)
    hs_ = theano.function(inputs=[id_seq, mask_seq], outputs=hs)
    id_seq_test = np.random.randint(low=0, high=5, size=[20, 10]).astype(dtype=np.int32)
    mask_seq_test = np.random.randint(low=0, high=2, size=[20, 10]).astype(dtype=np.int32)
    print hs_(id_seq_test, mask_seq_test).shape


    # q1_encode = T.matrix()
    # q2_encode = T.matrix()
    # shared_word = T.ivector()
    # model = ScoreModel(dim_in=512, dim_word=10, dim_hid=64, cf=config)
    # score = model.forward(shared_word, q1_encode, q2_encode)
    # f = theano.function(inputs=[shared_word, q1_encode, q2_encode], outputs=score, on_unused_input='ignore')
    # fake_q1 = np.random.rand(5, 256).astype(dtype=floatX)
    # fake_q2 = np.random.rand(5, 256).astype(dtype=floatX)
    # fake_shared_word = np.random.randint(low=0, high=10, size=[5]).astype(dtype=np.int32)
    # print f(fake_shared_word, fake_q1, fake_q2).shape


