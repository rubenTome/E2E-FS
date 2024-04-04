from keras import optimizers
from keras import backend as K
from keras.mixed_precision import LossScaleOptimizer
import tensorflow as tf
from tensorflow.python.training import gen_training_ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.python.framework import ops
from backend_config import bcknd

K.backend = bcknd


def get_e2efs_gradient(self, e2efs_grad):
        """Called in `minimize` to compute gradients from loss."""
        with tf.GradientTape() as e2efs_tape:
            e2efs_loss = self.e2efs_layer.regularization_func(self.e2efs_layer.kernel)
        e2efs_regularizer_grad = e2efs_tape.gradient(e2efs_loss, [self.e2efs_layer.kernel])[0]
        # tf.print(e2efs_regularizer_grad)
        e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + K.epsilon())
        e2efs_grad_corrected = e2efs_grad / (tf.norm(e2efs_grad) + K.epsilon())
        combined_e2efs_grad = (1. - self.e2efs_layer.moving_factor) * e2efs_grad_corrected + \
                              self.e2efs_layer.moving_factor * e2efs_regularizer_grad_corrected
        combined_e2efs_grad = K.sign(
            self.e2efs_layer.moving_factor) * K.minimum(K.cast_to_floatx(self.th), K.max(
            K.abs(combined_e2efs_grad))) * combined_e2efs_grad / K.max(
            K.abs(combined_e2efs_grad) + K.epsilon())
        return combined_e2efs_grad


class E2EFS_SGD(optimizers.SGD):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, e2efs_beta_1=0.5, e2efs_beta_2=0.999, e2efs_epsilon=1e-7, e2efs_amsgrad=False, **kwargs):
        super().__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_lr = e2efs_lr
        self.th = th
        self.e2efs_beta1 = e2efs_beta_1
        self.e2efs_beta2 = e2efs_beta_2
        self.e2efs_amsgrad = e2efs_amsgrad
        self.e2efs_epsilon = e2efs_epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """

        super().build(var_list)
        self._built = False
        self.velocities = []
        self.vhats = []
        for var in var_list:
            if 'e2efs' in var.name:
                self.velocities.append(self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                ))
                if self.e2efs_amsgrad:
                    self.vhats.append(self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    ))
        self._built = True

    def update_step(self, gradient, variable):
        if 'e2efs' in variable.name:
            e2efs_gradient = get_e2efs_gradient(self, gradient)
            self._update_step_e2efs(e2efs_gradient, variable)
        else:
            super().update_step(gradient, variable)

    def _update_step_e2efs(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.e2efs_lr, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1 = self.e2efs_beta1
        beta_2 = self.e2efs_beta2
        beta_1_power = tf.pow(tf.cast(beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]
        v = self.velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - beta_2),
                    gradient.indices,
                )
            )
            if self.e2efs_amsgrad:
                v_hat = self.vhats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.e2efs_epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - beta_2))
            if self.e2efs_amsgrad:
                v_hat = self.vhats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.e2efs_epsilon))


class E2EFS_Adam(optimizers.Adam):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, e2efs_beta_1=0.5, e2efs_beta_2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_lr = e2efs_lr
        self.th = th
        self.e2efs_beta1 = e2efs_beta_1
        self.e2efs_beta2 = e2efs_beta_2

    def build(self, var_list):
        """Initialize optimizer variables.

        Adam optimizer has 3 types of variables: momentums, velocities and
        velocity_hat (only set when amsgrad is applied),

        Args:
            var_list: list of model variables to build Adam variables on.
        """
        super().build(var_list)
        self._built = False
        self._momentums = []
        self._velocities = []
        for var in var_list:
            self._momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self._velocities.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.e2efs_lr if 'e2efs' in variable.name else self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1 = self.e2efs_beta1 if "e2efs" in variable.name else self.beta_1
        beta_2 = self.e2efs_beta2 if "e2efs" in variable.name else self.beta_2
        beta_1_power = tf.pow(tf.cast(beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(beta_2, variable.dtype), local_step)

        if 'e2efs' in variable.name:
            gradient = get_e2efs_gradient(self, gradient)

        var_key = self._var_key(variable)
        m = self._momentums[self._index_dict[var_key]]
        v = self._velocities[self._index_dict[var_key]]

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - beta_2),
                    gradient.indices,
                )
            )
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - beta_2))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))


class E2EFS_RMSprop(optimizers.RMSprop):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, e2efs_beta_1=0.5, e2efs_beta_2=0.999, e2efs_epsilon=1e-7, e2efs_amsgrad=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_lr = e2efs_lr
        self.th = th
        self.e2efs_beta1 = e2efs_beta_1
        self.e2efs_beta2 = e2efs_beta_2
        self.e2efs_amsgrad = e2efs_amsgrad
        self.e2efs_epsilon = e2efs_epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        SGD optimizer has one variable `momentums`, only set if `self.momentum`
        is not 0.

        Args:
          var_list: list of model variables to build SGD variables on.
        """
        super().build(var_list)
        for var in var_list:
            if 'e2efs' in var.name:
                self.e2efs_v = self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
                if self.e2efs_amsgrad:
                    self.e2efs_vhat = self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
        self._built = True

    def update_step(self, gradient, variable):
        if 'e2efs' in variable.name:
            e2efs_gradient = get_e2efs_gradient(self, gradient)
            self._update_step_e2efs(e2efs_gradient, variable)
        else:
            super().update_step(gradient, variable)

    def _update_step_e2efs(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.e2efs_lr, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1 = self.e2efs_beta1
        beta_2 = self.e2efs_beta2
        beta_1_power = tf.pow(tf.cast(beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]
        v = self.e2efs_v

        alpha = lr * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(-m * (1 - beta_1))
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - beta_1), gradient.indices
                )
            )
            v.assign_add(-v * (1 - beta_2))
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - beta_2),
                    gradient.indices,
                )
            )
            if self.e2efs_amsgrad:
                v_hat = self.e2efs_vhat
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.e2efs_epsilon))
        else:
            # Dense gradients.
            m.assign_add((gradient - m) * (1 - beta_1))
            v.assign_add((tf.square(gradient) - v) * (1 - beta_2))
            if self.e2efs_amsgrad:
                v_hat = self.e2efs_vhat
                v_hat.assign(tf.maximum(v_hat, v))
                v = v_hat
            variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.e2efs_epsilon))
