from configuration import *


def rmsprop(cost, params, config):
    lr = config['learning_rate']
    rho = config['rho']
    epsilon = config['epsilon']
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, T.cast(acc_new, 'float32')))
        updates.append((p, T.cast(p - lr * g, 'float32')))
    return updates
