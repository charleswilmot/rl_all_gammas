import tensorflow as tf
from tensorflow import keras


class KerasMLP(keras.Model):
    def __init__(self, *dimensions_activations):
        super(KerasMLP, self).__init__()
        self.denses = [
            keras.layers.Dense(dim, activation=activation)
            for dim, activation in dimensions_activations
        ]

    def call(self, inputs):
        outputs = inputs
        for dense in self.denses:
            outputs = dense(outputs)
        return outputs


class PolicyType1(KerasMLP):
    def __init__(self, action_space_dim):
        super(PolicyType1, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (action_space_dim, None)
        )


class CriticType1(KerasMLP):
    def __init__(self):
        super(CriticType1, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (1, None)
        )


class PolicyType2(KerasMLP):
    def __init__(self, action_space_dim):
        super(PolicyType2, self).__init__(
            (400, tf.nn.relu),
            (300, tf.nn.relu),
            (action_space_dim, tf.tanh)
        )


class CriticType2(KerasMLP):
    def __init__(self):
        super(CriticType2, self).__init__(
            (400, tf.nn.relu),
            (300, tf.nn.relu),
            (1, None)
        )


class PolicyType3(KerasMLP):
    def __init__(self, action_space_dim):
        super(PolicyType3, self).__init__(
            (100, tf.nn.relu),
            (100, tf.nn.relu),
            (action_space_dim, tf.tanh)
        )


def get_critic_model(name):
    if name == "type1":
        return CriticType1()
    if name == "type2":
        return CriticType2()
    else:
        raise ArgumentError("Unrecognized critic model name {}".format(name))


def get_policy_model(name, action_space_dim):
    if name == "type1":
        return PolicyType1(action_space_dim)
    if name == "type2":
        return PolicyType2(action_space_dim)
    if name == "type3":
        return PolicyType3(action_space_dim)
    else:
        raise ArgumentError("Unrecognized critic model name {}".format(name))
