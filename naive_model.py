from model import *
from component import *
from optimizor import *


class Naive_Comp(object):
    def __init__(self, dim_in, dim_encode, cf, prex='Naive_Comp'):
        self.dim_in = dim_in
        self.dim_encode = dim_encode
        self.cf = cf
        self.prex = prex
        self._init_params()
        self._init_computation_graph()

    def _init_params(self):
        self.q1encoder = Encoder(dim_in=self.dim_in,
                                 dim_encode=self.dim_encode,
                                 cf=self.cf,
                                 prex=pp(self.prex, 'Q1_encoder'))
        self.q2encoder = Encoder(dim_in=self.dim_in,
                                 dim_encode=self.dim_encode,
                                 cf=self.cf,
                                 prex=pp(self.prex, 'Q2_question'))
        self.juger = FC(dim_in=4 * self.dim_encode,
                        dim_out=1,
                        linear=False,
                        prex=pp(self.prex, 'juger'))
        self.params = self.q1encoder.params + self.q2encoder.params + self.juger.params

    def _init_computation_graph(self):
        q1_seq = T.imatrix('q1_seq_node')
        q1_mask_seq = T.imatrix('q1_mask_node')
        q2_seq = T.imatrix('q2_seq_node')
        q2_mask_seq = T.imatrix('q2_mask_node')
        label = T.vector('label_node')

        q1_encoded = self.q1encoder.forward(q1_seq, q1_mask_seq)
        q2_encoded = self.q2encoder.forward(q2_seq, q2_mask_seq)
        pair_encoded = T.concatenate([q1_encoded[-1], q2_encoded[-1]], axis=1)
        pred = self.juger.step_forward(pair_encoded).flatten()
        cost = - T.sum(label * T.log(pred) + (1 - label) * T.log(1 - pred))
        rmsupdates = rmsprop(cost, self.params, self.cf)

        self.q1_enc = theano.function(inputs=[q1_seq, q1_mask_seq], outputs=q1_encoded)
        self.q2_enc = theano.function(inputs=[q2_seq, q2_mask_seq], outputs=q2_encoded)
        self.pair_enc = theano.function(inputs=[q1_seq, q1_mask_seq, q2_seq, q2_mask_seq], outputs=pair_encoded)
        self.pred = theano.function(inputs=[q1_seq, q1_mask_seq, q2_seq, q2_mask_seq], outputs=pred)
        self.cost = theano.function(inputs=[q1_seq, q1_mask_seq, q2_seq, q2_mask_seq, label],
                                    outputs=cost,
                                    updates=rmsupdates)

if __name__ == '__main__':
    test_model = Naive_Comp(dim_in=config['dim_emb'], dim_encode=config['dim_encode'], cf=config)
    fake_data = np.random.randint(low=0, high=10, size=(20, 10)).astype(np.int32)
    fake_mask = np.random.randint(low=0, high=2, size=(20, 10)).astype(np.int32)
    fake_label = np.random.randint(low=0, high=2, size=10).astype(np.float32)
    print test_model.pred(fake_data, fake_mask, fake_data, fake_mask).shape
    for i in range(10):
        print test_model.cost(fake_data, fake_mask, fake_data, fake_mask, fake_label)
