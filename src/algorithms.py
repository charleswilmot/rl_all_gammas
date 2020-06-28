import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from return_viewer import ReturnViewer



class Algorithm(object):
    def __init__(self, environment, agent, replay_buffer, training_steps=0,
            evaluate_every=0, train_every=100, batch_size=100,
            make_critic_checkpoint=False, restore_from_checkpoint=None,
            train_noise_every=4, return_viewer=False):
        self.env = environment
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.training_steps = training_steps
        self.evaluate_every = evaluate_every
        self.train_every = train_every
        self.batch_size = batch_size
        self.make_critic_checkpoint = make_critic_checkpoint
        self.train_noise_every = train_noise_every
        self.return_viewer = ReturnViewer() if return_viewer else False
        self.summary_writer = tf.summary.create_file_writer("logs")
        if restore_from_checkpoint is not False:
            self.restore_model(restore_from_checkpoint)

    @staticmethod
    def from_conf(environment, agent, replay_buffer, **algorithm_conf):
        class_in_conf = eval(algorithm_conf.pop("class"))
        return class_in_conf(
            environment,
            agent,
            replay_buffer,
            **algorithm_conf
        )

    def __call__(self, training_steps=None, evaluate_every=None):
        self.main_loop(
            self.training_steps if training_steps is None else training_steps,
            self.evaluate_every if evaluate_every is None else evaluate_every
        )
        self.save_model("checkpoints")

    def save_model(self, path):
        self.agent.save_model(path)

    def restore_model(self, path):
        self.agent.restore_model(path)


class OffPolicyAlgorithm(Algorithm):
    def __init__(self, environment, agent, replay_buffer, training_steps=0,
            evaluate_every=0, train_every=100, batch_size=100,
            make_critic_checkpoint=False, restore_from_checkpoint=None,
            train_noise_every=4, return_viewer=False):
        super(OffPolicyAlgorithm, self).__init__(
            environment, agent, replay_buffer, training_steps, evaluate_every,
            train_every, batch_size, make_critic_checkpoint,
            restore_from_checkpoint, train_noise_every, return_viewer
        )
        self.episode_counter = tf.Variable(0, name="episode_counter")
        self.train_step_counter = tf.Variable(0, name="train_step")
        # self.global_step = self.train_step_counter
        self.global_step = self.episode_counter
        self._hparams = {
            "train_every": self.train_every,
            "batch_size": self.batch_size
        }
        # define a few metrics to keep track of
        self.training_episode_length = tf.keras.metrics.Mean(
            "training/episode_length",
            dtype=tf.float32
        )
        self.training_critic_loss = tf.keras.metrics.Mean(
            "training/critic_loss",
            dtype=tf.float32
        )
        self.training_actor_loss = tf.keras.metrics.Mean(
            "training/actor_loss",
            dtype=tf.float32
        )
        self.training_critic_signal_to_noise = tf.keras.metrics.Mean(
            "training/critic_signal_to_noise_db",
            dtype=tf.float32
        )
        self.testing_episode_length = tf.keras.metrics.Mean(
            "testing/episode_length",
            dtype=tf.float32
        )
        self.testing_total_episode_reward = tf.keras.metrics.Mean(
            "testing/total_episode_reward",
            dtype=tf.float32
        )
        self.testing_mean_total_episode_reward = tf.keras.metrics.Mean(
            "testing/mean_total_episode_reward",
            dtype=tf.float32
        )

    def main_loop(self, training_steps, evaluate_every):
        n_done = 0
        n_iterations_collected = 0
        while training_steps > n_done:
            n_iterations_collected += self.collect_training_data()
            while n_done == 0 \
                    or n_iterations_collected / n_done > self.train_every:
                if self.agent.noise_params["applied_on"] == "input":
                    train_actor = not ((n_done + 1) % self.train_noise_every == 0)
                    train_noise = not train_actor
                else:
                    train_actor = True
                    train_noise = False
                if self.make_critic_checkpoint:
                    train_actor = False
                    train_noise = False
                self.train(
                    train_actor=train_actor,
                    train_noise=train_noise,
                )
                n_done += 1
                if n_done % 100 == 0:
                    print("{: 6d}/{: 6d} training steps done".format(
                        n_done, training_steps
                    ))
                if n_done % evaluate_every == 0:
                    self.evaluate()
                if n_done % (evaluate_every * 10) == 0:
                    self.log_summaries(self.global_step.numpy())
                if not training_steps > n_done:
                    break
        print("{} training steps completed, exiting main loop".format(n_done))

    def log_summaries(self, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(
                self.training_episode_length.name,
                self.training_episode_length.result(),
                step=step
            )
            tf.summary.scalar(
                self.training_critic_loss.name,
                self.training_critic_loss.result(),
                step=step
            )
            tf.summary.scalar(
                self.training_actor_loss.name,
                self.training_actor_loss.result(),
                step=step
            )
            tf.summary.scalar(
                self.training_critic_signal_to_noise.name,
                self.training_critic_signal_to_noise.result(),
                step=step
            )
            tf.summary.scalar(
                self.testing_episode_length.name,
                self.testing_episode_length.result(),
                step=step
            )
            tf.summary.scalar(
                self.testing_total_episode_reward.name,
                self.testing_total_episode_reward.result(),
                step=step
            )
            tf.summary.scalar(
                self.testing_mean_total_episode_reward.name,
                self.testing_mean_total_episode_reward.result(),
                step=step
            )
            hparams = {
                **self.agent._hparams,
                **self.replay_buffer._hparams,
                **self._hparams,
                "env_id": self.env.spec.id,
            }
            hp.hparams(hparams)
        self.training_episode_length.reset_states()
        self.training_critic_loss.reset_states()
        self.training_actor_loss.reset_states()
        self.training_critic_signal_to_noise.reset_states()
        self.testing_episode_length.reset_states()
        self.testing_total_episode_reward.reset_states()

    def train(self, train_actor=True, train_critic=True, train_noise=False):
        # sample from buffer
        training_data = self.replay_buffer.sample(self.batch_size)
        # call agent.train
        critic_loss, actor_loss = self.agent.train(
            training_data,
            train_actor=train_actor,
            train_critic=train_critic,
            train_noise=train_noise
        )
        if self.replay_buffer.is_prioritized:
            estimated_returns = self.agent.get_estimated_returns(
                training_data["states"],
                training_data["actions"],
                mode=1
            )
            priorities = np.abs(estimated_returns[:, 0] - training_data["targets"])
            self.replay_buffer.update_priorities(
                training_data["indices"],
                priorities
            )
        self.train_step_counter.assign_add(1)
        # accumulate log data in keras metrics
        self.training_critic_loss(critic_loss)
        self.training_actor_loss(actor_loss)

    def collect_training_data(self):
        # resent env
        print("  collecting", end='\r')
        max_steps = self.env.spec.max_episode_steps
        shape = (max_steps, ) + self.env.observation_space.shape
        states_b = np.zeros(shape=shape, dtype=self.env.observation_space.dtype)
        if self.agent.noise_params["applied_on"] == "input":
            noisy_states_b = np.zeros(
                shape=(max_steps,) + self.agent.noisy_state_space_shape,
                dtype=self.env.observation_space.dtype
            )
        if self.env.action_space.dtype == np.int64:
            shape = (max_steps, self.env.action_space.n)
        else:
            shape = (max_steps, ) + self.env.action_space.shape
        actions_b = np.zeros(shape=shape, dtype=np.float32)
        rewards_b = np.zeros(shape=max_steps, dtype=np.float32)
        estimated_returns_b = np.zeros(shape=max_steps, dtype=np.float32)
        state = self.env.reset()
        n_transitions = 0
        while True:
            states = state[np.newaxis].astype(np.float32)
            policy_inputs = self.agent.get_policy_inputs(states, explore=True)
            actions = self.agent.get_actions(policy_inputs, explore=True)
            action = actions[0]
            estimated_returns = self.agent.get_estimated_returns(
                states,
                actions,
                mode=1
            ).numpy()
            estimated_return = estimated_returns[0]
            next_state, reward, done, _ = self.env.step(action.numpy())
            # store state action reward done in temp buffer
            states_b[n_transitions] = state
            if self.agent.noise_params["applied_on"] == "input":
                noisy_states_b[n_transitions] = policy_inputs[0]
            actions_b[n_transitions] = action
            rewards_b[n_transitions] = reward
            estimated_returns_b[n_transitions] = estimated_return
            state = next_state
            n_transitions += 1
            if done:
                break
        self.training_episode_length(n_transitions)
        self.episode_counter.assign_add(1)
        if done:  # and n_transitions < max_steps:
            bootstraping_return = 0
        else:
            bootstraping_return = self.agent.get_bootstraping_return(
                states,
            ).numpy()[0]  # tensor to numpy
        if self.agent.noise_params["applied_on"] == "input":
            to_buffer = self.agent.get_buffer_data(
                states=states_b[:n_transitions],
                actions=actions_b[:n_transitions],
                rewards=rewards_b[:n_transitions],
                estimated_returns=estimated_returns_b[:n_transitions],
                bootstraping_return=bootstraping_return,
                noisy_states=noisy_states_b[:n_transitions]
            )
        else:
            to_buffer = self.agent.get_buffer_data(
                states=states_b[:n_transitions],
                actions=actions_b[:n_transitions],
                rewards=rewards_b[:n_transitions],
                estimated_returns=estimated_returns_b[:n_transitions],
                bootstraping_return=bootstraping_return,
            )
        self.replay_buffer.register_episode(**to_buffer)
        # display return if needed
        targets = to_buffer["targets"]
        if self.return_viewer:
            self.return_viewer(targets, estimated_returns_b[:n_transitions])
        # log signal to noise ratio
        noise = targets - estimated_returns_b[:len(targets)]
        signal_to_noise = 10 * (
            np.log10(np.var(targets)) - np.log10(np.var(noise))
        )
        self.training_critic_signal_to_noise(signal_to_noise)
        ########################################################################
        # must return number of iteration added to the buffer in order to ensure
        # that the correct number of weight update is performed
        print("            ", end='\r')
        return n_transitions

    def evaluate(self, explore=False):
        print("  evaluation", end='\r')
        state = self.env.reset()
        n_transitions = 0
        total_episode_reward = 0
        while True:
            states = state[np.newaxis].astype(np.float32)
            policy_inputs = self.agent.get_policy_inputs(states, explore=explore)
            actions = self.agent.get_actions(policy_inputs, explore=explore)
            action = actions.numpy()[0]
            state, reward, done, _ = self.env.step(action)
            n_transitions += 1
            total_episode_reward += reward
            if done:
                break
        self.testing_episode_length(n_transitions)
        self.testing_total_episode_reward(total_episode_reward)
        self.testing_mean_total_episode_reward(total_episode_reward)
        print("            ", end='\r')
