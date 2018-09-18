from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K

class MySGD(Optimizer):
    """Keras中简单自定义SGD优化器
    每隔一定的batch才更新一次参数
    """
    def __init__(self, lr=0.01, steps_per_update=1, **kwargs):
        super(MySGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.steps_per_update = steps_per_update # 多少batch才更新一次

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """主要的参数更新算法
        """
        shapes = [K.int_shape(p) for p in params]
        sum_grads = [K.zeros(shape) for shape in shapes] # 平均梯度，用来梯度下降
        grads = self.get_gradients(loss, params) # 当前batch梯度
        self.updates = [K.update_add(self.iterations, 1)] # 定义赋值算子集合
        self.weights = [self.iterations] + sum_grads # 优化器带来的权重，在保存模型时会被保存
        for p, g, sg in zip(params, grads, sum_grads):
            # 梯度下降
            new_p = p - self.lr * sg / float(self.steps_per_update)
            # 如果有约束，对参数加上约束
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            cond = K.equal(self.iterations % self.steps_per_update, 0)
            # 满足条件才更新参数
            self.updates.append(K.switch(cond, K.update(p, new_p), p))
            # 满足条件就要重新累积，不满足条件直接累积
            self.updates.append(K.switch(cond, K.update(sg, g), K.update(sg, sg+g)))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'steps_per_update': self.steps_per_update}
        base_config = super(MySGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))