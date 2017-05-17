from environment import *


def gen_config_rn():
    config = {}

    # path related
    config['data_folder'] = '../data'
    config['embed_matrix'] = '../wordembed/embeding_matrix.pkl'

    # data related
    config['stop_words'] = ['the','a','an','and','but','if','or','because','as','what','which','this','that','these','those','then',
                            'just','so','than','such','both','through','about','for','is','of','while','during','to','What','Which',
                            'Is','If','While','This']
    config['remain_freq'] = 50
    config['vocab_size'] = 28803

    # model related
    config['floatX'] = np.float32
    config['dim_emb'] = 300
    config['dim_encode'] = 128
    config['dim_word'] = 300
    config['score_dim_hid'] = 64
    config['comp_dim_hid'] = 64

    # optimizor related
    config['learning_rate'] = 1e-3
    config['rho'] = 0.8
    config['epsilon'] = 1e-5

    # training related
    config['train_num'] = 390400
    config['batch_size'] = 256
    config['train_iteration'] = 100

    # validation_related
    config['validate_num'] = 10240
    config['tolerate_step'] = 5
    config['tolerate_acc_up'] = 0.01
    return config

