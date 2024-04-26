import keras.config
from keras import optimizers, ops, backend as K
import tensorflow as tf
from backend_config import bcknd

ops.backend = bcknd


def get_e2efs_gradient(self, e2efs_grad):
        """Called in `minimize` to compute gradients from loss."""
        with tf.GradientTape() as e2efs_tape:
            e2efs_loss = self.e2efs_layer.regularization_func(self.e2efs_layer.kernel)
        e2efs_regularizer_grad = e2efs_tape.gradient(e2efs_loss, [self.e2efs_layer.kernel])[0]
        # tf.print(e2efs_regularizer_grad)
        e2efs_regularizer_grad_corrected = e2efs_regularizer_grad / (tf.norm(e2efs_regularizer_grad) + ops.epsilon())
        e2efs_grad_corrected = e2efs_grad / (tf.norm(e2efs_grad) + ops.epsilon())
        combined_e2efs_grad = ops.cast(1. - self.e2efs_layer.moving_factor, dtype=e2efs_regularizer_grad.dtype) * e2efs_grad_corrected + \
                              ops.cast(self.e2efs_layer.moving_factor, dtype=e2efs_regularizer_grad.dtype) * e2efs_regularizer_grad_corrected
        combined_e2efs_grad = ops.sign(
            ops.cast(self.e2efs_layer.moving_factor, dtype=e2efs_regularizer_grad.dtype)) * ops.minimum(
            ops.cast(self.th, dtype=e2efs_regularizer_grad.dtype), ops.max(
            ops.abs(combined_e2efs_grad))) * combined_e2efs_grad / ops.max(
            ops.abs(combined_e2efs_grad) + ops.epsilon())
        return combined_e2efs_grad


class E2EFS_SGD(optimizers.SGD):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, e2efs_beta_1=0.5, e2efs_beta_2=0.999, e2efs_epsilon=1e-3, e2efs_amsgrad=False, **kwargs):
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
        self._velocities = []
        self._vhats = []
        self._momentums = []
        for var in var_list:
            if 'e2efs' in var.path:
                self._momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum"
                    )
                )
                self._velocities.append(self.add_variable_from_reference(
                    reference_variable=var, name="v"
                ))
                if self.e2efs_amsgrad:
                    self._vhats.append(self.add_variable_from_reference(
                        reference_variable=var, name="vhat"
                    ))
        self._built = True

    def update_step(self, gradient, variable, learning_rate):
        if 'e2efs' in variable.name:
            e2efs_gradient = get_e2efs_gradient(self, gradient)
            self._update_step_e2efs(e2efs_gradient, variable)
        else:
            super().update_step(gradient, variable, learning_rate)

    def _update_step_e2efs(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(self.e2efs_lr, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(
            ops.cast(self.e2efs_beta1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.e2efs_beta2, variable.dtype), local_step
        )

        m = self._momentums[0]
        v = self._velocities[0]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.e2efs_beta1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.e2efs_beta2
            ),
        )
        if self.e2efs_amsgrad:
            v_hat = self._vhats[0]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.e2efs_epsilon)
            ),
        )


class E2EFS_Adam(optimizers.Adam):

    def __init__(self, e2efs_layer, th=.1, e2efs_lr=0.01, e2efs_beta_1=0.5, e2efs_beta_2=0.999, **kwargs):
        super().__init__(**kwargs)
        self.e2efs_layer = e2efs_layer
        self.e2efs_lr = e2efs_lr
        self.th = th
        self.e2efs_beta1 = e2efs_beta_1
        self.e2efs_beta2 = e2efs_beta_2

    def update_step(self, gradient, variable, learning_rate):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(self.e2efs_lr if 'e2efs' in variable.name else self.learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        if 'e2efs' in variable.name:
            gradient = get_e2efs_gradient(self, gradient)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1 = self.e2efs_beta1 if "e2efs" in variable.name else self.beta_1
        beta_2 = self.e2efs_beta2 if "e2efs" in variable.name else self.beta_2
        beta_1_power = ops.power(
            ops.cast(beta_1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(beta_2, variable.dtype), local_step
        )

        m = self._momentums[self._get_variable_index(variable)]
        v = self._velocities[self._get_variable_index(variable)]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - beta_1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - beta_2
            ),
        )
        if self.amsgrad:
            v_hat = self._velocity_hats[self._get_variable_index(variable)]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.epsilon)
            ),
        )


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
        self._built = False
        self._velocities = []
        self._vhats = []
        self._momentums = []
        for var in var_list:
            if 'e2efs' in var.path:
                self._momentums.append(
                    self.add_variable_from_reference(
                        reference_variable=var, name="momentum"
                    )
                )
                self._velocities.append(self.add_variable_from_reference(
                    reference_variable=var, name="v"
                ))
                if self.e2efs_amsgrad:
                    self._vhats.append(self.add_variable_from_reference(
                        reference_variable=var, name="vhat"
                    ))
        self._built = True

    def update_step(self, gradient, variable, learning_rate):
        if 'e2efs' in variable.name:
            e2efs_gradient = get_e2efs_gradient(self, gradient)
            self._update_step_e2efs(e2efs_gradient, variable)
        else:
            super().update_step(gradient, variable, learning_rate)

    def _update_step_e2efs(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = ops.cast(self.e2efs_lr, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        local_step = ops.cast(self.iterations + 1, variable.dtype)
        beta_1_power = ops.power(
            ops.cast(self.e2efs_beta1, variable.dtype), local_step
        )
        beta_2_power = ops.power(
            ops.cast(self.e2efs_beta2, variable.dtype), local_step
        )

        m = self._momentums[0]
        v = self._velocities[0]

        alpha = lr * ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)

        self.assign_add(
            m, ops.multiply(ops.subtract(gradient, m), 1 - self.e2efs_beta1)
        )
        self.assign_add(
            v,
            ops.multiply(
                ops.subtract(ops.square(gradient), v), 1 - self.e2efs_beta2
            ),
        )
        if self.e2efs_amsgrad:
            v_hat = self._vhats[0]
            self.assign(v_hat, ops.maximum(v_hat, v))
            v = v_hat
        self.assign_sub(
            variable,
            ops.divide(
                ops.multiply(m, alpha), ops.add(ops.sqrt(v), self.e2efs_epsilon)
            ),
        )
