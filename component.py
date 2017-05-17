from utils import *
import theano.tensor.signal.pool as pool


class Embed(object):
    def __init__(self, cf, prex='Embed'):
        self.cf = cf
        self.prex = prex
        self._init_params()

    def _init_params(self):
        emf = open(self.cf['embed_matrix'], 'rb')
        self.W = theano.shared(value=pickle.load(emf),
                               name=pp(self.prex, 'embedding matrix'))
        emf.close()
        self.params = [self.W]

    def apply(self, index):
        embedded = self.W[index]
        return T.cast(embedded, 'float32')


class FC(object):
    def __init__(self, dim_in, dim_out, linear=False, prex='FC'):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = linear
        self.prex = prex
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=[self.dim_in, self.dim_out],
                            param_name=pp(self.prex, 'W'))
        self.b = init_param(param_type='bias',
                            param_size=self.dim_out,
                            param_name=pp(self.prex, 'b'))
        self.params = [self.W, self.b]

    def step_forward(self, x):
        e = T.dot(x, self.W) + self.b
        if self.linear:
            y = e
        else:
            y = T.nnet.sigmoid(e)
        return y


class Softmax(object):
    def __init__(self, dim_in, n_class, prex='Softmax'):
        self.dim_in = dim_in
        self.n_class = n_class
        self.prex = prex
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=[self.dim_in, self.n_class],
                            param_name=pp(self.prex, 'W'))
        self.b = init_param(param_type='bias',
                            param_size=[self.n_class],
                            param_name=pp(self.prex, 'b'))
        self.params = [self.W]

    def step_forward(self, x):
        e = T.dot(x, self.W) + self.b
        l = T.nnet.softmax(e)
        return l


class LSTM(object):
    def __init__(self, dim_in, dim_hid, embed, prex='LSTM'):
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.prex = prex
        self.embed = embed
        self._init_params()

    def _init_params(self):
        self.W_in = init_param(param_type='weight',
                               param_size=[4, self.dim_in, self.dim_hid],
                               param_name=pp(self.prex, 'W_in'))
        self.W_hid = init_param(param_type='weight',
                                param_size=[4, self.dim_hid, self.dim_hid],
                                param_name=pp(self.prex, 'W_hid'))
        self.b = init_param(param_type='bias',
                            param_size=[4, self.dim_hid],
                            param_name=pp(self.prex, 'bias'))
        self.params = [self.W_in, self.W_hid, self.b]

    def step_forward_emb(self, x, m, htm1, ctm1):
        x_embed = self.embed.apply(x)
        z = T.tanh(T.dot(x_embed, self.W_in[0]) + T.dot(htm1, self.W_hid[0]) + self.b[0])
        i = T.nnet.sigmoid(T.dot(x_embed, self.W_in[1]) + T.dot(htm1, self.W_hid[1]) + self.b[1])
        f = T.nnet.sigmoid(T.dot(x_embed, self.W_in[2]) + T.dot(htm1, self.W_hid[2]) + self.b[2])
        o = T.nnet.sigmoid(T.dot(x_embed, self.W_in[3]) + T.dot(htm1, self.W_hid[3]) + self.b[3])

        c = f * ctm1 + i * z
        c = m[:, None] * c + (1 - m)[:, None] * ctm1
        h = T.tanh(o * c)
        h = m[:, None] * h + (1 - m)[:, None] * htm1

        return T.cast(h, 'float32'), T.cast(c, 'float32')

    def step_forward(self, x, m, htm1, ctm1):
        z = T.tanh(T.dot(x, self.W_in[0]) + T.dot(htm1, self.W_hid[0]) + self.b[0])
        i = T.nnet.sigmoid(T.dot(x, self.W_in[1]) + T.dot(htm1, self.W_hid[1]) + self.b[1])
        f = T.nnet.sigmoid(T.dot(x, self.W_in[2]) + T.dot(htm1, self.W_hid[2]) + self.b[2])
        o = T.nnet.sigmoid(T.dot(x, self.W_in[3]) + T.dot(htm1, self.W_hid[3]) + self.b[3])

        c = f * ctm1 + i * z
        c = m[:, None] * c + (1 - m)[:, None] * ctm1
        h = T.tanh(o * c)
        h = m[:, None] * h + (1 - m)[:, None] * htm1

        return T.cast(h, 'float32'), T.cast(c, 'float32')


class Conv2d(object):

    def __init__(self, fshape, filter_size, pool_size, cf, prex='Conv2D'):
        self.fshape = fshape
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.cf = cf
        self.prex = prex
        self._init_params()

    def _init_params(self):
        self.W = init_param(param_type='weight',
                            param_size=self.filter_size,
                            param_name=pp(self.prex, 'weght'))
        self.b = init_param(param_type='bias',
                            param_size=self.filter_size[0],
                            param_name=pp(self.prex, 'convbias'))
        self.params = [self.W, self.b]

    def step_forward(self, fmap):
        conv_out = T.nnet.conv2d(input=fmap,
                                 filters=self.W,
                                 filter_shape=self.filter_size,
                                 input_shape=self.fshape)
        pooled_out = pool.pool_2d(input=conv_out,
                                           ds=self.pool_size,
                                           ignore_border=True
                                           )
        y = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return y

    def simple_step_forward(self, fmap, image_size):
        conv_out = T.nnet.conv2d(input=fmap,
                                 filters=self.W,
                                 filter_shape=self.filter_size,
                                 input_shape=[1] + image_size)
        pooled_out = pool.pool_2d(input=conv_out,
                                  ds=self.pool_size,
                                  ignore_border=True
                                  )
        y = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return y

if __name__ == '__main__':
    # embed = Embed(vocab_size=10, dim_embed=10)
    # test_vector = T.ivector()
    # test_index = T.argmax(test_vector)
    # test_index = T.cast(test_index, 'int32')
    # emb = embed.apply(test_vector)
    # f = theano.function(inputs=[test_vector], outputs=emb)
    # print f(np.random.randint(low=0, high=10, size=5).astype(dtype=np.int32)).shape
    # test_matrix = T.tensor4()
    # dim_embed = 10
    # test_filter_shape = [1, 1, 21, 10]
    # test_fmap_shape = [1, 1, 60, 10]
    # test_model = Conv2d(fshape=test_fmap_shape,
    #                     filter_size=test_filter_shape,
    #                     pool_size=[1, 1],
    #                     cf=None)
    #
    # result = test_model.step_forward(test_matrix)
    # f = theano.function(inputs=[test_matrix], outputs=result)
    # print f(np.random.randn(1, 1, 60, 10).astype(dtype=np.float32)).shape
    test_embed = Embed(config)
    print test_embed.apply(2).eval()
