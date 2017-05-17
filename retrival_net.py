from model import *
from optimizor import *


class RetrivalNet(object):

    def __init__(self, cf, prex='RetrivalNet'):
        self.cf = cf
        self.prex = prex
        self._init_params()
        self._init_computation_graph()

    def _init_params(self):
        self.embed = Embed(self.cf)
        self.encoderModule = Encoder(dim_in=self.cf['dim_emb'],
                                     dim_encode=self.cf['dim_encode'],
                                     embed=self.embed,
                                     cf=self.cf,
                                     prex=pp(self.prex, 'Encoder')
                                     )
        self.scoreModule = ScoreModel(dim_in=4 * self.cf['dim_encode'],
                                      dim_word=self.cf['dim_word'],
                                      dim_hid=self.cf['score_dim_hid'],
                                      embed=self.embed,
                                      cf=self.cf,
                                      prex=pp(self.prex, 'Score'))
        self.compModule = CompModel(dim_in=4 * self.cf['dim_encode'] + self.cf['dim_word'],
                                    dim_hid=self.cf['comp_dim_hid'],
                                    cf=self.cf,
                                    prex=pp(self.prex, 'Comp'))
        self.params = self.embed.params + self.encoderModule.params + self.scoreModule.params + self.compModule.params

    def forward(self, q1batch, q1mask, q2batch, q2mask, shared_words_batch, word_mask):
        q1batch_encoded = self.encoderModule.forward(q1batch, q1mask)
        q2batch_encoded = self.encoderModule.forward(q2batch, q2mask)
        scores, _ = theano.scan(fn=self.scoreModule.forward,
                                sequences=[shared_words_batch],
                                non_sequences=[q1batch_encoded, q2batch_encoded],
                                outputs_info=[None])
        scores = T.exp(scores).flatten(ndim=2) * word_mask
        score_sum = T.sum(scores, axis=0) + 1e-10
        score_sum = score_sum.dimshuffle(0, 'x')
        score_sum = T.repeat(score_sum, scores.shape[0], axis=1).dimshuffle(1, 0)
        scores /= score_sum

        word_vector, _ = theano.scan(fn=self.scoreModule.generate_word_vector,
                                     sequences=[scores.dimshuffle(1, 0),
                                                shared_words_batch.dimshuffle(1, 0)])

        com_inputs = T.concatenate([q1batch_encoded, q2batch_encoded, word_vector], axis=1)
        pred, _ = theano.scan(fn=self.compModule.forward,
                              sequences=[com_inputs])
        # pred_prob = T.reshape(pred, (pred.shape[0], pred.shape[-1]))
        return pred.flatten()

    def _init_computation_graph(self):
        q1_node = T.imatrix('q1_node')
        q2_node = T.imatrix('q2_node')
        q1_mask = T.imatrix('q1_mask')
        q2_mask = T.imatrix('q2_mask')
        shared_words_node = T.imatrix('word_node')
        word_mask_node = T.imatrix('word_mask')
        label_node = T.ivector('label_node')
        pred_prob = self.forward(q1_node, q1_mask, q2_node, q2_mask, shared_words_node, word_mask_node)
        cost = -T.sum(label_node * T.log(pred_prob) + (1 - label_node) * T.log(1 - pred_prob))
        cost = T.cast(cost, 'float32')
        rmsupdates = rmsprop(cost=cost, params=self.params, config=self.cf)
        #
        self.cost = theano.function(inputs=[q1_node,
                                            q2_node,
                                            q1_mask,
                                            q2_mask,
                                            shared_words_node,
                                            word_mask_node,
                                            label_node],
                                    outputs=cost,
                                    updates=rmsupdates)
        pred = T.switch(pred_prob >= 0.5, 1, 0)
        correct_num = T.sum(T.eq(pred, label_node))
        self.validate = theano.function(inputs=[q1_node,
                                                q2_node,
                                                q1_mask,
                                                q2_mask,
                                                shared_words_node,
                                                word_mask_node,
                                                label_node],
                                        outputs=[cost, pred, correct_num],
                                        on_unused_input='ignore')


if __name__ == '__main__':
    q1_node = T.imatrix('q1_node')
    q2_node = T.imatrix('q2_node')
    q1_mask = T.imatrix('q1_mask')
    q2_mask = T.imatrix('q2_mask')
    shared_words_node = T.imatrix('word_node')
    word_mask = T.imatrix('word_mask')
    label_node = T.ivector('label_node')

    test_model = RetrivalNet(cf=config)

    fake_q1 = np.random.randint(low=0, high=10, size=[20, 20]).astype(dtype=np.int32)
    fake_q2 = np.random.randint(low=0, high=10, size=[10, 20]).astype(dtype=np.int32)
    fake_q1_mask = np.random.randint(low=0, high=2, size=[20, 20]).astype(dtype=np.int32)
    fake_q2_mask = np.random.randint(low=0, high=2, size=[10, 20]).astype(dtype=np.int32)
    shared_words = np.random.randint(low=0, high=2, size=[2, 20]).astype(dtype=np.int32)
    word_mask = np.random.randint(low=0, high=2, size=[2, 20]).astype(dtype=np.int32)
    fake_label = np.random.randint(low=0, high=2, size=20).astype(dtype=np.int32)
    # print test_model.pred(fake_q1,
    #                       fake_q2,
    #                       fake_q1_mask,
    #                       fake_q2_mask,
    #                       shared_words,
    #                       word_mask,
    #                       fake_label)
    for i in range(10):
        print test_model.cost(fake_q1,
                              fake_q2,
                              fake_q1_mask,
                              fake_q2_mask,
                              shared_words,
                              word_mask,
                              fake_label)
        print test_model.validate(fake_q1,
                                  fake_q2,
                                  fake_q1_mask,
                                  fake_q2_mask,
                                  shared_words,
                                  word_mask,
                                  fake_label)
