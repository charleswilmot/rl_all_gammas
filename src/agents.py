import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from network_models import get_policy_model, get_critic_model
import numpy as np


class Agent(object):
    def __init__(self, environment):
        self.discrete_actions = environment.action_space.dtype == np.int64
        if self.discrete_actions:
            self.action_space_dim = environment.action_space.n
        else:
            self.action_space_dim = environment.action_space.shape[0]
        self.state_space_shape = environment.observation_space.shape

    @staticmethod
    def from_conf(environment, **agent_conf):
        class_in_conf = eval(agent_conf.pop("class"))
        return class_in_conf(environment, **agent_conf)

    @tf.function
    def get_actions(self, states, explore=True):
        pass

    @tf.function
    def get_estimated_returns(self, states, actions):
        pass

    @tf.function
    def get_critic_loss(self, states, actions, target_returns):
        pass

    @tf.function
    def get_actor_loss(self, states):
        pass

    def rewards_to_target_returns(self, rewards, bootstraping_return):
        returns = np.zeros_like(rewards)
        # to reward scale
        previous = bootstraping_return * self.reward_scaling_factor
        last = rewards.shape[0] - 1
        for i, reward in zip(np.arange(last, -1, -1), rewards[::-1]):
            returns[i] = self.gamma * previous + reward
            previous = returns[i]
        return returns / self.reward_scaling_factor

    @tf.function
    def train(self, states, actions, target_returns):
        # something with gradient tape
        pass


class TD3Agent(Agent):
    def __init__(self, environment, policy_model, critic_model,
            critic_learning_rate, actor_learning_rate, gamma, noise_stddev,
            reward_scaling_factor):
        super(TD3Agent, self).__init__(environment)
        self.policy_model = get_policy_model(policy_model, self.action_space_dim)
        self.critic_model_1 = get_critic_model(critic_model)
        self.critic_model_2 = get_critic_model(critic_model)
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.gamma = gamma
        self.noise_stddev = noise_stddev
        self.reward_scaling_factor = reward_scaling_factor
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        self.actor_optimizer = keras.optimizers.Adam(self.actor_learning_rate)
        self._hparams = {
            # hp.HParam('actor_learning_rate', hp.RealInterval(1e-6, 1e-2))
            "critic_learning_rate": self.critic_learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "gamma": self.gamma,
            "noise_stddev": self.noise_stddev,
            "reward_scaling_factor": self.reward_scaling_factor,
            "policy_model": policy_model,
            "critic_model": critic_model,
        }

    def save_model(self, path):
        self.critic_model_1.save_weights(path + "/critic_model_1")
        self.critic_model_2.save_weights(path + "/critic_model_2")
        self.policy_model.save_weights(path + "/policy_model")

    def restore_model(self, path):
        self.critic_model_1.load_weights(path + "/critic_model_1")
        self.critic_model_2.load_weights(path + "/critic_model_2")
        self.policy_model.load_weights(path + "/policy_model")

    @tf.function
    def get_actions(self, states, explore=True, logits=False):
        """Maps the states to the actions, depending on the values of
        self.discrete_actions, explore and logits
        if explore is set, must add noise on the action
        if actions are discrete
            if logit is set
                must return log probs
            else
                must return action indices
        """
        if self.discrete_actions:
            action_logits = self.policy_model(states)
            if explore:
                noise = tf.random.normal(
                    shape=tf.shape(action_logits),
                    stddev=self.noise_stddev
                )
                action_logits += noise
            if logits:
                return action_logits
            actions_distribution = \
                tfp.distributions.Categorical(logits=action_logits)
            return actions_distribution.sample()
        else:  # continuous action control
            actions = self.policy_model(states)
            if explore:
                noise = tf.random.normal(
                    shape=tf.shape(actions),
                    stddev=self.noise_stddev
                )
                actions += noise
            return actions

    @tf.function
    def get_estimated_returns(self, states, actions, mode="minimum"):
        # print("Tracing get_estimated_returns, {} {} {}".format(
        #     type(states),
        #     type(actions),
        #     mode)
        # )
        state_actions = tf.concat([states, actions], axis=-1)
        estimated_returns_1 = self.critic_model_1(state_actions)
        estimated_returns_2 = self.critic_model_2(state_actions)
        if mode == 1:
            return estimated_returns_1
        elif mode == 2:
            return estimated_returns_2
        elif mode == "minimum":
            return tf.minimum(estimated_returns_1, estimated_returns_2)
        else:
            raise ArgumentError("Unrecognized return estimate mode")

    @tf.function
    def get_critic_loss(self, states, actions, target_returns):
        h = keras.losses.Huber(delta=1.0)
        target_returns = tf.reshape(target_returns, (-1, 1))
        estimated_returns_1 = self.get_estimated_returns(states, actions, mode=1)
        estimated_returns_2 = self.get_estimated_returns(states, actions, mode=2)
        loss = h(target_returns, estimated_returns_1)
        loss += h(target_returns, estimated_returns_2)
        return loss / 2

    @tf.function
    def get_actor_loss(self, states):
        actions = self.get_actions(states, explore=False, logits=True)
        return - tf.reduce_mean(
            self.get_estimated_returns(states, actions, mode=1)
        )

    @tf.function
    def train(self, states, actions, target_returns, train_actor=True,
            train_critic=True):
        if train_actor:
            train_op_actor = self.actor_optimizer.minimize(
                lambda: self.get_actor_loss(states),
                var_list=lambda: self.policy_model.variables
            )
        if train_critic:
            train_op_critic = self.critic_optimizer.minimize(
                lambda: self.get_critic_loss(states, actions, target_returns),
                var_list=lambda: \
                    self.critic_model_1.variables + \
                    self.critic_model_2.variables
            )
        critic_loss = self.get_critic_loss(states, actions, target_returns)
        actor_loss = self.get_actor_loss(states)
        return critic_loss, actor_loss
