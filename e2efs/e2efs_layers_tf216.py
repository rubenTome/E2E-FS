from keras import backend as K, layers, models, initializers, ops
import keras
from keras.constraints import Constraint
import numpy as np
from backend_config import bcknd

ops.cast_to_floatx = lambda x: ops.cast(x, keras.config.floatx())
K.backend = bcknd


class E2EFS_Base(layers.Layer):

    def __init__(self, units,
                 kernel_initializer='truncated_normal',
                 kernel_constraint=None,
                 kernel_activation=None,
                 kernel_regularizer=None,
                 heatmap_momentum=.99999,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(E2EFS_Base, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_constraint = kernel_constraint
        self.kernel_activation = kernel_activation
        self.kernel_regularizer = kernel_regularizer
        self.supports_masking = True
        self.kernel = None
        self.heatmap_momentum = heatmap_momentum

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = np.prod(input_shape[1:])
        kernel_shape = (input_dim, )
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.trainable)
        self.moving_heatmap = self.add_weight(shape=(input_dim, ),
                                              name='heatmap',
                                              initializer='ones',
                                              trainable=False)
        self.stateful = False
        self.built = True

    def e2efs_kernel(self):
        return self.kernel if self.kernel_activation is None else self.kernel_activation(self.kernel)

    def call(self, inputs, training=None, **kwargs):

        kernel = self.kernel
        if self.kernel_activation is not None:
            kernel = self.kernel_activation(kernel)
        kernel_clipped = ops.reshape(kernel, ops.shape(inputs)[1:])

        output = inputs * kernel_clipped

        if training in {0, False}:
            return output

        self._get_update_list(kernel)

        return output

    def _get_update_list(self, kernel):
        self.moving_heatmap.assign(
            self.heatmap_momentum * self.moving_heatmap + (1. - self.moving_heatmap) * ops.sign(kernel)
        )

    def add_to_model(self, model, input_shape, activation=None):
        input = layers.Input(shape=input_shape)
        x = self(input)
        if activation is not None:
            x = layers.Activation(activation=activation)(x)
        output = model(x)
        model = models.Model(input, output)
        model.fs_kernel = self.e2efs_kernel
        model.heatmap = self.moving_heatmap
        return model

    def compute_output_shape(self, input_shape):
        return input_shape


class KernelConstraint(Constraint):

    def __call__(self, w):
        return ops.clip(w, 0., 1.)

class E2EFSSoft(E2EFS_Base):

    def __init__(self, units,
                 dropout=.0,
                 decay_factor=.75,
                 kernel_regularizer=None,
                 kernel_initializer='ones',
                 T=10000,
                 warmup_T=2000,
                 start_alpha=.0,
                 alpha_N=.99,
                 epsilon=.001,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.dropout = ops.cast_to_floatx(dropout)
        self.decay_factor = ops.cast_to_floatx(decay_factor)
        self.T = ops.cast_to_floatx(T * epsilon)
        self.warmup_T = ops.cast_to_floatx(warmup_T * epsilon)
        self.start_alpha = ops.cast_to_floatx(start_alpha)
        self.cont_T = 0
        self.alpha_M = ops.cast_to_floatx(alpha_N)
        self.epsilon = ops.cast_to_floatx(epsilon)
        self.increment = ops.cast_to_floatx((1. - start_alpha) / T)
        super(E2EFSSoft, self).__init__(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=kernel_initializer,
                                        heatmap_momentum= (T - 1.) / T,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False)
        self.moving_increment = self.add_weight(shape=(),
                                            name='moving_increment',
                                            initializer=initializers.constant(self.increment),
                                            trainable=False, dtype='float32')
        self.moving_T = self.add_weight(shape=(),
                                        name='moving_T',
                                        initializer='zeros',
                                        trainable=False)
        self.moving_factor = self.add_weight(shape=(),
                                             name='moving_factor',
                                             initializer=initializers.constant(0.),
                                             trainable=False, dtype='float32')
        self.moving_decay = self.add_weight(shape=(),
                                             name='moving_decay',
                                             initializer=initializers.constant(self.decay_factor),
                                             trainable=False)
        self.cont = self.add_weight(shape=(),
                                    name='cont',
                                    initializer='ones',
                                    trainable=False)

        def kernel_activation(x):
            t = x / ops.max(ops.abs(x))
            s = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x), x)
            # s /= ops.stop_gradient(ops.max(s))
            return s

        self.kernel_activation = kernel_activation

        self.kernel_constraint = KernelConstraint

        def loss_units(x):
            t = x / ops.max(ops.abs(x))
            x = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x, dtype=x.dtype), x)
            m = ops.sum(ops.cast(ops.greater(x, 0.), keras.config.floatx()))
            sum_x = ops.sum(x)
            moving_units = ops.where(ops.less_equal(m, self.units), m,
                                    ops.cast((1. - self.moving_decay) * self.moving_units, x.dtype))
            epsilon_minus = 0.
            epsilon_plus = ops.where(ops.less_equal(m, self.units), ops.cast(self.moving_units, dtype=x.dtype), 0.)
            return ops.relu(moving_units - sum_x - epsilon_minus) + ops.relu(sum_x - moving_units - epsilon_plus)

        # self.kernel_regularizer = lambda x: regularizers.l2(.01)(ops.relu(x))

        super(E2EFSSoft, self).build(input_shape)

        def regularization(x):
            l_units = loss_units(x)
            t = x / ops.max(ops.abs(x) + ops.epsilon())
            p = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x, dtype=x.dtype), x)
            cost = ops.sum(p * (1. - p)) + 2. * l_units
            return cost

        self.regularization_loss = regularization(self.kernel)
        self.regularization_func = regularization

    def _get_update_list(self, kernel):
        super(E2EFSSoft, self)._get_update_list(kernel)
        self.moving_factor.assign(
            ops.where(ops.less(self.moving_T, self.warmup_T),
                     self.start_alpha,
                     ops.minimum(self.alpha_M,
                               self.moving_factor + self.moving_increment))
        )
        self.moving_T.assign_add(self.epsilon)
        self.moving_decay.assign(
            ops.where(ops.less(self.moving_factor, self.alpha_M), self.moving_decay,
                     ops.maximum(.75, self.moving_decay + self.epsilon))
        )
        # self.moving_increment.assign(
        #     ops.where(ops.cast_to_floatx(self.moving_factor + self.moving_increment) - ops.cast_to_floatx(self.moving_factor) < .1 * self.moving_increment, 1e-6 + self.moving_increment,
        #              self.moving_increment)
        # )


class E2EFS(E2EFSSoft):

    def __init__(self, units,
                 dropout=.0,
                 kernel_initializer='ones',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(E2EFS, self).__init__(units=units,
                                    dropout=dropout,
                                    kernel_regularizer=None,
                                    decay_factor=0.,
                                    kernel_initializer=kernel_initializer,
                                    T=10000,
                                    **kwargs)


class E2EFSRanking(E2EFS_Base):

    def __init__(self, units,
                 dropout=.0,
                 kernel_regularizer=None,
                 kernel_initializer='ones',
                 T=20000,
                 warmup_T=2000,
                 start_alpha=.0,
                 speedup=4.,
                 alpha_M =.99,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        self.dropout = dropout
        self.T = T
        self.warmup_T = warmup_T
        self.start_alpha = start_alpha
        self.cont_T = 0
        self.speedup = speedup
        self.alpha_M = alpha_M
        super(E2EFSRanking, self).__init__(units=units,
                                        kernel_regularizer=kernel_regularizer,
                                        kernel_initializer=kernel_initializer,
                                        heatmap_momentum= (T - 1.) / T,
                                        **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.moving_units = self.add_weight(shape=(),
                                            name='moving_units',
                                            initializer=initializers.constant(self.units),
                                            trainable=False, dtype='float32')
        self.moving_T = self.add_weight(shape=(),
                                        name='moving_T',
                                        initializer='zeros',
                                        trainable=False, dtype='float32')
        self.moving_factor = self.add_weight(shape=(),
                                             name='moving_factor',
                                             initializer=initializers.constant([0.]),
                                             trainable=False, dtype='float32')
        self.cont = self.add_weight(shape=(),
                                    name='cont',
                                    initializer='ones',
                                    trainable=False, dtype='float32')

        def apply_dropout(x, rate, refactor=False):
            if 0. < self.dropout < 1.:
                def dropped_inputs():
                    x_shape = ops.int_shape(x)
                    noise = ops.random_uniform(x_shape)
                    factor = 1. / (1. - rate) if refactor else 1.
                    return ops.where(ops.less(noise, self.dropout), ops.zeros_like(x), factor * x)
                return ops.in_train_phase(dropped_inputs, x)
            return x

        def kernel_activation(x):
            x = apply_dropout(x, self.dropout, False)
            t = x / ops.max(ops.abs(x))
            s = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x), x)
            # s /= ops.stop_gradient(ops.max(s))
            return s

        self.kernel_activation = kernel_activation

        def kernel_constraint(x):
            return ops.clip(x, 0., 1.)

        self.kernel_constraint = kernel_constraint

        def loss_units(x):
            t = x / ops.max(ops.abs(x))
            x = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x), x)
            # m = ops.sum(ops.cast(ops.greater(x, 0.), ops.floatx()))
            sum_x = ops.sum(x)
            # moving_units = ops.where(ops.less_equal(m, self.units), m, self.moving_units)
            # epsilon_minus = 0.
            # epsilon_plus = ops.where(ops.less_equal(m, self.units), self.moving_units, 0.)
            return ops.abs(self.moving_units - sum_x)

        # self.kernel_regularizer = lambda x: regularizers.l2(.01)(ops.relu(x))
        # self.kernel_initializer = initializers.constant(max(.05, self.units / np.prod(input_shape[1:])))

        super(E2EFSRanking, self).build(input_shape)

        def regularization(x):
            l_units = loss_units(x)
            t = x / ops.max(ops.abs(x))
            p = ops.where(ops.less(t, ops.epsilon()), ops.zeros_like(x), x)
            cost = ops.cast_to_floatx(0.)
            cost += ops.sum(p) - ops.sum(ops.square(p)) + 2. * l_units
            # cost += ops.sum(p * (1. - p)) + l_units
            # cost += ops.sum(ops.relu(x - 1.))
            return cost

        self.regularization_loss = regularization(self.kernel)
        self.regularization_func = regularization

    def _get_update_list(self, kernel):
        super(E2EFSRanking, self)._get_update_list(kernel)
        self.moving_factor.assign(
            ops.where(ops.less(self.moving_T, self.warmup_T),
                     self.start_alpha,
                     ops.minimum(self.alpha_M,
                               self.start_alpha + (1. - self.start_alpha) * (self.moving_T - self.warmup_T) / self.T))
        )
        self.moving_T.assign_add(1.)
        self.moving_units.assign(
            ops.where(ops.less_equal(self.moving_T, self.warmup_T),
                     (1. - self.start_alpha) * np.prod(ops.int_shape(kernel)),
                     ops.maximum(self.alpha_M,
                               np.prod(ops.int_shape(kernel)) * ops.power(1. / np.prod(ops.int_shape(kernel)),
                                                                    self.speedup * (
                                                                                self.moving_T - self.warmup_T) / self.T)))
        )