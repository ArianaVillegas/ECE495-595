import tensorflow as tf
import numpy as np

from tf_agents.utils import common


class DQN:
    def __init__(self, fc_layer_params, num_actions, optimizer, eps=0.05) -> None:
        self.fc_layer_params = fc_layer_params
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.eps = eps

        self.q_net = self._build_q_net()
        self.target_net = self._build_q_net()

    def _dense_layer(self, num_units) -> tf.keras.layers.Dense:
        return tf.keras.layers.Dense(
                num_units, 
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            )

    def _build_q_net(self) -> tf.keras.Sequential:
        q_net = tf.keras.Sequential()
        [q_net.add(self._dense_layer(num_units)) for num_units in self.fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
                            self.num_actions,
                            activation=None,
                            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
                            bias_initializer=tf.keras.initializers.Constant(-0.2)
                        )
        q_net.add(q_values_layer)
        q_net.compile(optimizer=self.optimizer,
                      loss='mse')
        return q_net

    def update_target_net(self) -> None:
        self.target_net.set_weights(self.q_net.get_weights())

    def random_action(self) -> int:
        idx = np.random.choice(self.num_actions)
        return idx

    def _get_action(self, state) -> int:
        state_tf = tf.convert_to_tensor(state)
        cur_actions = self.q_net(state_tf).numpy()
        opt_action = np.argmax(cur_actions[0], axis=0)
        return opt_action

    def get_action(self, state, random=False) -> int:
        if random or np.random.random() < self.eps:
            return self.random_action()
        return self._get_action(state)

    def train(self, experience_batch, gamma=0.9) -> float:
        states, actions, next_states, rewards, dones = experience_batch
        target_qval = self.q_net(states).numpy()
        next_target_qval = self.target_net(next_states).numpy()
        max_next_qval = np.amax(next_target_qval, axis=1)
        for i, reward in enumerate(rewards):
            if not dones[i]:
                reward += gamma * max_next_qval[i]
            target_qval[i][actions[i]] = reward
        summary = self.q_net.fit(states, target_qval, verbose=0)
        train_loss = summary.history['loss'][0]
        return train_loss

    def name(self) -> str:
        return 'DQN'



class DDQN(DQN):
    def __init__(self, fc_layer_params, num_actions, optimizer, eps=0.05) -> None:
        super().__init__(fc_layer_params, num_actions, optimizer, eps)

    def train(self, experience_batch, gamma=0.9) -> float:
        states, actions, next_states, rewards, dones = experience_batch
        target_qval = self.q_net(states).numpy()
        next_qval = self.q_net(next_states).numpy()
        argmax_next_qval = np.argmax(next_qval, axis=1)
        next_target_qval = self.target_net(next_states).numpy()
        for i, reward in enumerate(rewards):
            if not dones[i]:
                reward += gamma * next_target_qval[i][argmax_next_qval[i]]
            target_qval[i][actions[i]] = reward
        summary = self.q_net.fit(states, target_qval, verbose=0)
        train_loss = summary.history['loss'][0]
        return train_loss

    def name(self) -> str:
        return 'DDQN'
    