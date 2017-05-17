from utils import *
from retrival_net import *


load_model_flag = False
model_id = 5
# load_data
f = open('../data/train_data.pkl', 'rb')
data = pickle.load(f)
f.close()
# data_setting
[q1s, q2s, q1_masks, q2_masks, sws, sw_masks, label] = data
label = np.array(label).astype(dtype=np.int32)
# train_setting
index_pairs = gen_batch_index(config['train_num'], config['batch_size'])
# model_initialize
if load_model_flag:
    f = open('../model_built/model{}.pkl'.format(model_id), 'rb')
    model = pickle.load(f)
    f.close()
else:
    model = RetrivalNet(config)
# validation_setting
v_s_ix = config['train_num']
v_e_ix = v_s_ix + config['validate_num']
v_q1s = q1s[:, v_s_ix:v_e_ix]
v_q2s = q2s[:, v_s_ix:v_e_ix]
v_q1_masks = q1_masks[:, v_s_ix:v_e_ix]
v_q2_masks = q2_masks[:, v_s_ix:v_e_ix]
v_sws = sws[:, v_s_ix:v_e_ix]
v_sw_masks = sw_masks[:, v_s_ix:v_e_ix]
v_label = label[v_s_ix:v_e_ix]

p_v_acc = 0
p_v_cost = 1e5
best_model_id = 0
log = []
for epoch in range(config['train_iteration']):
    cost = 0
    for s_ix, e_ix in tqdm(index_pairs):
        c_q1s = q1s[:, s_ix:e_ix]
        c_q2s = q2s[:, s_ix:e_ix]
        c_q1_masks = q1_masks[:, s_ix:e_ix]
        c_q2_masks = q2_masks[:, s_ix:e_ix]
        c_sws = sws[:, s_ix:e_ix]
        c_sw_masks = sw_masks[:, s_ix:e_ix]
        c_label = label[s_ix:e_ix]

        cost += model.cost(c_q1s,
                           c_q2s,
                           c_q1_masks,
                           c_q2_masks,
                           c_sws,
                           c_sw_masks,
                           c_label)
    [v_cost, v_pred, v_correct_num] = model.validate(v_q1s,
                                                     v_q2s,
                                                     v_q1_masks,
                                                     v_q2_masks,
                                                     v_sws,
                                                     v_sw_masks,
                                                     v_label)
    c_v_acc = float(v_correct_num)/config['validate_num']
    v_cost = float(v_cost)/config['validate_num']
    cost = float(cost)/config['train_num']
    print 'iteration: {}'.format(epoch + 1)
    print 'training_cost: {}'.format(cost)
    print 'validation_cost: {}, acc: {}'.format(v_cost, c_v_acc)
    log.append('iteration:{}, '
               'training_cost:{}, '
               'validation_cost:{}, '
               'validation_acc:{}'.format(epoch + 1, cost, v_cost, c_v_acc))

    if c_v_acc > p_v_acc:
        p_v_acc = c_v_acc
        if load_model_flag:
            best_model_id = epoch + 1 + model_id
        else:
            best_model_id = epoch + 1
        print 'new_best_model: model {}'.format(best_model_id)
        print 'saving model...'
        f = open('../model_built/model{}.pkl'.format(epoch + 1), 'wb')
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
    logf = open('../data/log{}.pkl'.format(epoch + 1), 'wb')
    pickle.dump(log, logf)
    logf.close()
