import tensorflow as tf
from tensorflow import keras


class KerasMLP(keras.Model):
    def __init__(self, *dimensions_activations, weight_init_scale=1):
        super(KerasMLP, self).__init__()
        initializer = keras.initializers.VarianceScaling(
            scale=weight_init_scale,
            mode="fan_avg",
            distribution="uniform",
            seed=None
        )
        self.denses = [
            keras.layers.Dense(
                dim,
                activation=activation,
                kernel_initializer=initializer)
            for dim, activation in dimensions_activations
        ]

    def call(self, inputs):
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
        return outputs


class PolicyType1(KerasMLP):
    def __init__(self, action_space_dim, weight_init_scale):
        super(PolicyType1, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (action_space_dim, None),
            weight_init_scale=weight_init_scale
        )


class CriticType1(KerasMLP):
    def __init__(self, weight_init_scale):
        super(CriticType1, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (1, None),
            weight_init_scale=weight_init_scale
        )


class PolicyType2(KerasMLP):
    def __init__(self, action_space_dim, weight_init_scale):
        super(PolicyType2, self).__init__(
            (400, tf.nn.relu),
            (300, tf.nn.relu),
            (action_space_dim, tf.tanh),
            weight_init_scale=weight_init_scale
        )


class CriticType2(KerasMLP):
    def __init__(self, weight_init_scale):
        super(CriticType2, self).__init__(
            (400, tf.nn.relu),
            (300, tf.nn.relu),
            (1, None),
            weight_init_scale=weight_init_scale
        )


class PolicyType3(KerasMLP):
    def __init__(self, action_space_dim, weight_init_scale):
        super(PolicyType3, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (action_space_dim, tf.tanh),
            weight_init_scale=weight_init_scale
        )


class PolicyType4(KerasMLP):
    def __init__(self, action_space_dim, weight_init_scale):
        super(PolicyType4, self).__init__(
            (100, tf.nn.relu),
            (action_space_dim, tf.tanh),
            weight_init_scale=weight_init_scale
        )


class PolicyType5(KerasMLP):
    def __init__(self, action_space_dim, weight_init_scale):
        super(PolicyType5, self).__init__(
            (action_space_dim, tf.tanh),
            weight_init_scale=weight_init_scale
        )


def get_critic_model(model_params):
    type = model_params["type"]
    weight_init_scale = model_params["weight_init_scale"]
    if type == "type1":
        return CriticType1(weight_init_scale)
    if type == "type2":
        return CriticType2(weight_init_scale)
    else:
        raise ArgumentError("Unrecognized critic model type {}".format(type))


def get_policy_model(model_params, action_space_dim):
    type = model_params["type"]
    weight_init_scale = model_params["weight_init_scale"]
    if type == "type1":
        return PolicyType1(action_space_dim, weight_init_scale)
    if type == "type2":
        return PolicyType2(action_space_dim, weight_init_scale)
    if type == "type3":
        return PolicyType3(action_space_dim, weight_init_scale)
    if type == "type4":
        return PolicyType4(action_space_dim, weight_init_scale)
    if type == "type5":
        return PolicyType5(action_space_dim, weight_init_scale)
    else:
        raise ArgumentError("Unrecognized critic model type {}".format(type))
