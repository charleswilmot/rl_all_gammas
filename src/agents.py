import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from network_models import get_policy_model, get_critic_model
import numpy as np
from ornstein_uhlenbeck import OUProcess


class AgentBase(object):
    @staticmethod
    def from_conf(environment, **agent_conf):
        class_in_conf = eval(agent_conf.pop("class"))
        return class_in_conf(environment, **agent_conf)


class Agent(AgentBase):
    def __init__(self, environment, policy_model, critic_model,
            critic_learning_rate, actor_learning_rate, gamma, noise_params,
            reward_scaling_factor, target_computation_params):
        self.discrete_actions = environment.action_space.dtype == np.int64
        if self.discrete_actions:
            self.action_space_dim = environment.action_space.n
        else:
            self.action_space_dim = environment.action_space.shape[0]
        self.state_space_shape = environment.observation_space.shape
        if noise_params["applied_on"] == "input":
            self.noisy_state_space_shape = np.copy(self.state_space_shape)
            self.noisy_state_space_shape[-1] += noise_params["noise_dim"]
            self.noisy_state_space_shape = tuple(self.noisy_state_space_shape)
            self.input_noise_shape = np.copy(self.state_space_shape)
            self.input_noise_shape[-1] = noise_params["noise_dim"]
            self.input_noise_shape = tuple(self.input_noise_shape)
        self.policy_model = get_policy_model(policy_model, self.action_space_dim)
        self.critic_model_1 = get_critic_model(critic_model)
        self.critic_model_2 = get_critic_model(critic_model)
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.gamma = gamma
        self.noise_params = noise_params
        self.reward_scaling_factor = reward_scaling_factor
        self.target_computation_params = target_computation_params
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)
        self.actor_optimizer = keras.optimizers.Adam(self.actor_learning_rate)
        self._hparams = {
            "critic_learning_rate": self.critic_learning_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "gamma": self.gamma,
            "reward_scaling_factor": self.reward_scaling_factor,
            "policy_model": policy_model,
            "critic_model": critic_model,
        }
        self._noise_params_to_hparams()
        self._target_computation_params_to_hparams()

    def _noise_params_to_hparams(self):
        for key, value in self.noise_params.items():
            self._hparams["noise_" + key] = value

    def _target_computation_params_to_hparams(self):
        for key, value in self.target_computation_params.items():
            self._hparams["target_" + key] = value

    def save_model(self, path):
        self.critic_model_1.save_weights(path + "/critic_model_1")
        self.critic_model_2.save_weights(path + "/critic_model_2")
        self.policy_model.save_weights(path + "/policy_model")

    def restore_model(self, path):
        self.critic_model_1.load_weights(path + "/critic_model_1")
        self.critic_model_2.load_weights(path + "/critic_model_2")
        self.policy_model.load_weights(path + "/policy_model")

    @tf.function
    def get_noise(self, shape):
        stddev = self.noise_params["stddev"]
        if self.noise_params["type"] == "normal":
            return tf.random.normal(shape=shape, stddev=stddev)
        elif self.noise_params["type"] == "partial":
            noise_prob = self.noise_params["prob"]
            noise = tf.random.normal(shape=shape, stddev=stddev)
            gate = tf.random.uniform(shape=shape[:1], dtype=tf.float32)
            gate = tf.cast(tf.greater(noise_prob, gate), tf.float32)
            return noise * gate
        elif self.noise_params["type"] == "ornstein_uhlenbeck":
            if not hasattr(self, "ornstein_uhlenbeck"):
                self.ornstein_uhlenbeck = OUProcess(
                    tf.random.normal(
                        shape=shape,
                        stddev=stddev,
                        dtype=tf.float32,
                    ),
                    damping=self.noise_params["ou_damping"],
                    stddev=stddev,
                )
            return self.ornstein_uhlenbeck()
        else:
            raise ValueError("Unrecognized noise type, got {}".format(
                self.noise_params["type"]
            ))

    @tf.function
    def get_noisy_states(self, states, explore=True):
        batch_dim = tf.shape(states)[:1]
        if explore:
            additional_noise_shape = (1,) + self.input_noise_shape  # for OUProc
            noise = self.get_noise(additional_noise_shape)
        else:
            additional_noise_shape = tf.concat(
                [batch_dim, self.input_noise_shape],
                axis=0
            )
            noise = tf.zeros(
                shape=additional_noise_shape,
                dtype=states.dtype
            )
        return tf.concat([states, noise], axis=-1)

    @tf.function
    def get_policy_inputs(self, states, explore=True):
        """Maps the states to the actions, depending on the values of
        self.discrete_actions and explore
        if explore is set, must add noise on the action
        """
        if self.noise_params["applied_on"] == "input":
            return self.get_noisy_states(states, explore=explore)
        else:
            return states

    @tf.function
    def get_actions(self, policy_inputs, explore=True):
        """Maps the states to the actions, depending on the values of
        self.discrete_actions and explore
        if explore is set, must add noise on the action
        """
        if self.noise_params["applied_on"] == "input":
            actions = self.policy_model(policy_inputs)
        else:
            actions = self.policy_model(policy_inputs)
            if explore:
                actions += self.get_noise((1, self.action_space_dim))
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
    def get_bootstraping_return(self, states):
        policy_inputs = self.get_policy_inputs(states, explore=False)
        actions = self.get_actions(policy_inputs, explore=False)
        return self.get_estimated_returns(states, actions)

    @tf.function
    def get_critic_loss(self, states, actions, target_returns,
            batch_weights=None):
        target_returns = tf.reshape(target_returns, (-1, 1))
        estimated_returns_1 = self.get_estimated_returns(states, actions, mode=1)
        estimated_returns_2 = self.get_estimated_returns(states, actions, mode=2)
        h = keras.losses.Huber(delta=1.0, reduction='none')
        loss = h(target_returns, estimated_returns_1)
        loss += h(target_returns, estimated_returns_2)
        if batch_weights is not None:
            loss *= batch_weights
        return tf.reduce_mean(loss) / 2

    @tf.function
    def get_actor_loss(self, states, policy_inputs, batch_weights=None,
            explore=False):
        actions = self.get_actions(policy_inputs, explore=explore)
        if batch_weights is not None:
            return - tf.reduce_mean(
                self.get_estimated_returns(states, actions, mode=1)[:, 0] \
                * batch_weights
            )
        else:
            return - tf.reduce_mean(
                self.get_estimated_returns(states, actions, mode=1)[:, 0]
            )

    @tf.function
    def train_actor(self, states, policy_inputs, batch_weights=None):
        train_op_actor = self.actor_optimizer.minimize(
            lambda: \
                self.get_actor_loss(
                    states,
                    policy_inputs,
                    batch_weights=batch_weights
                ),
            var_list=lambda: self.policy_model.variables
        )

    @tf.function
    def train_critic(self, states, actions, target_returns, batch_weights=None):
        train_op_critic = self.critic_optimizer.minimize(
            lambda: \
                self.get_critic_loss(
                    states, actions, target_returns,
                    batch_weights=batch_weights
                ),
            var_list=lambda: \
                self.critic_model_1.variables + \
                self.critic_model_2.variables
        )

    # @tf.function
    # def train(self, states, actions, target_returns, train_actor=True,
    #         train_critic=True, batch_weights=None):
    #     critic_loss = self.get_critic_loss(
    #         states,
    #         actions,
    #         target_returns,
    #         batch_weights=batch_weights
    #     )
    #     actor_loss = self.get_actor_loss(states, batch_weights=batch_weights)
    #     if train_critic:
    #         self.train_critic(states, actions, target_returns,
    #             batch_weights=batch_weights)
    #     if train_actor:
    #         self.train_actor(states, batch_weights=batch_weights)
    #     return critic_loss, actor_loss

    def train(self, training_data, train_actor=True, train_critic=True,
            train_noise=False):
        states = training_data["states"]
        actions = training_data["actions"]
        target_returns = training_data["targets"]
        if "importance" in training_data.dtype.fields.keys():
            batch_weights = training_data["importance"]
        else:
            batch_weights = None
        if "noisy_states" in training_data.dtype.fields.keys():
            noisy_states = training_data["noisy_states"]
        else:
            noisy_states = None
        return self.train_proxy(
            states=states,
            actions=actions,
            target_returns=target_returns,
            batch_weights=batch_weights,
            noisy_states=noisy_states,
            train_actor=True,
            train_critic=True,
            train_noise=False
        )

    @tf.function
    def train_proxy(self, states, actions,target_returns, batch_weights,
            noisy_states, train_actor=True, train_critic=True,
            train_noise=False):
        policy_inputs = self.get_policy_inputs(states, explore=False)
        critic_loss = self.get_critic_loss(
            states,
            actions,
            target_returns,
            batch_weights=batch_weights
        )
        actor_loss = self.get_actor_loss(
            states,
            policy_inputs,
            batch_weights=batch_weights
        )
        if train_critic:
            self.train_critic(
                states,
                actions,
                target_returns,
                batch_weights=batch_weights
            )
        if train_actor:
            if self.noise_params["applied_on"] == "input" and train_noise:
                self.train_actor(
                    states,
                    noisy_states,
                    batch_weights=batch_weights
                )
            else:
                self.train_actor(
                    states,
                    policy_inputs,
                    batch_weights=batch_weights
                )
        return critic_loss, actor_loss

    def rewards_to_target_returns(self, rewards, *args, **kwargs):
        return self.scaled_rewards_to_target_returns(
            rewards / self.reward_scaling_factor,
            *args,
            **kwargs
        )

    def scaled_rewards_to_target_returns(self, rewards, estimated_returns,
            bootstraping_return):
        # estimated_returns and bootstraping_return are scaled down
        # rewards are the raw rewards from the environment
        rshape = rewards.shape[1:]
        if self.target_computation_params["type"] == "n_steps_strict":
            n_steps = self.target_computation_params["n_steps"]
            final_size = len(rewards) - n_steps + 1
            if final_size > 0:
                targets = np.zeros(shape=(final_size,) + rshape, dtype=np.float32)
                targets[:-1] = estimated_returns[n_steps:]
                targets[-1] = bootstraping_return
                targets *= self.gamma ** n_steps
                for i in range(n_steps):
                    targets += self.gamma ** i * rewards[i:i + final_size]
            else:
                targets = np.zeros(shape=(0,) + rshape, dtype=np.float32)
            return targets
        elif self.target_computation_params["type"] == "n_steps":
            n_steps = self.target_computation_params["n_steps"]
            targets = np.zeros(shape=(len(rewards),) + rshape, dtype=np.float32)
            # strict part
            strict_size = len(rewards) - n_steps + 1
            if strict_size > 0:
                targets[:strict_size - 1] = estimated_returns[n_steps:]
                targets[strict_size - 1] = bootstraping_return
                targets[:strict_size] *= self.gamma ** n_steps
                for i in range(n_steps):
                    tmp = self.gamma ** i * rewards[i:i + strict_size]
                    targets[:strict_size] += tmp
                rewards_rest = rewards[1 - n_steps:]
            else:
                strict_size = 0
                rewards_rest = rewards
            # adaptive n_steps part (plays an important role)
            last = len(rewards_rest) - 1
            previous = bootstraping_return
            for i, reward in zip(np.arange(last, -1, -1), rewards_rest[::-1]):
                targets[i + strict_size] = self.gamma * previous + reward
                previous = targets[i + strict_size]
            return targets
        elif self.target_computation_params["type"] == "max_steps":
            returns = np.zeros_like(rewards)
            previous = bootstraping_return
            last = rewards.shape[0] - 1
            for i, reward in zip(np.arange(last, -1, -1), rewards[::-1]):
                returns[i] = self.gamma * previous + reward
                previous = returns[i]
            return returns
        else:
            raise ValueError(
                "Unrecognized target computation type, got {}".format(
                    self.target_computation_params["type"])
            )

    def get_buffer_data(self, states, actions, rewards,
            estimated_returns, bootstraping_return, **others):
        # must return a dict
        # some keys are mandatory: states, actions, targets, priorities
        targets = self.rewards_to_target_returns(
            rewards,
            estimated_returns,
            bootstraping_return
        )
        n = len(targets)
        ret = {}
        ret["states"] = states[:n]
        ret["actions"] = actions[:n]
        ret["targets"] = targets
        ret["priorities"] = np.abs(targets - estimated_returns[:n])
        for key, value in others.items():
            ret[key] = value[:n]
        return ret
