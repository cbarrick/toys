import numpy as np
import scipy as sp
import tensorflow as tf

import csb


class ReplayMemory:
    '''A class for storing and sampling experience replay memories.'''

    def __init__(self, cap, obs_shape):
        self.index = 0
        self.len = 0
        self.cap = cap
        self.data = np.zeros(
            cap,
            dtype=[
                ('old_obs', 'f4', obs_shape),
                ('new_obs', 'f4', obs_shape),
                ('action', 'i4'),
                ('reward', 'f4'),
                ('done', 'bool'),
            ])

    def __len__(self):
        return self.len

    def push(self, old_obs, action, new_obs, reward, done):
        if self.len < self.cap:
            self.len += 1
        self.index = (self.index + 1) % self.cap
        row = self.data[self.index]
        row['old_obs'] = old_obs
        row['new_obs'] = new_obs
        row['action'] = action
        row['reward'] = reward
        row['done'] = done

    def sample(self, n):
        return np.random.choice(self.data[:self.len], n)


class DQN(csb.Model):
    def __init__(self,
                 env,
                 memory_size=2**16,
                 minibatch_size=32,
                 max_repeat=32,
                 learn_freq=4,
                 target_update_freq=10000,
                 replay_start=50000,
                 discount_factor=0.99,
                 exploration_initial=1,
                 exploration_final=0.1,
                 exploration_steps=1000000,
                 optimizer=tf.train.AdamOptimizer(0.001),
                 name='dqn'):

        # Hyper-parameters
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.name = name
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.max_repeat = max_repeat
        self.learn_freq = learn_freq
        self.target_update_freq = target_update_freq
        self.replay_start = replay_start
        self.discount_factor = discount_factor
        self.exploration_initial = exploration_initial
        self.exploration_final = exploration_final
        self.exploration_steps = exploration_steps
        self.optimizer = optimizer

        # Runtime state
        self.name = name
        self.prev_action = 0
        self.prev_action_count = 0
        self.memory = ReplayMemory(self.memory_size, self.obs_shape)
        self.sess = self.new_session()
        self.target = self.new_session()

    def act(self, obs):
        exploration = self.sess.run(self.exploration)

        if len(self.memory) < self.memory_size:
            # Play randomly until memory buffer is full.
            action = np.random.randint(self.n_actions)

        elif np.random.random() < exploration:
            # Sometimes perform a random action.
            # The exploration value is annealed over time.
            action = np.random.randint(self.n_actions)

        else:
            # Otherwise choose the best predicted action.
            obs = np.expand_dims(obs, 0)
            q = self.sess.run(self.q, {self.input: obs})
            action = np.argmax(q)

        # Don't repeat actions too often
        if action == self.prev_action:
            self.prev_action_count += 1
            if self.prev_action_count > self.max_repeat:
                while action == self.prev_action:
                    action = np.random.randint(self.n_actions)
                self.prev_action_count = 0
        else:
            self.prev_action_count = 0
        self.prev_action = action

        return action

    def learn(self, obs, action, obs_next, reward, done, info):
        global_step = self.sess.run(self.increment_global_step)

        # Clip the reward.
        if reward > 1: reward = 1.0
        elif reward < -1: reward = -1.0

        # Record the new memory.
        self.memory.push(obs, action, obs_next, reward, done)

        # Update the target network.
        if global_step % self.target_update_freq == 0:
            self.update_target()

        # Do the training, but only after an initial waiting period.
        if self.replay_start < global_step and global_step % self.learn_freq == 0:
            # Get the experiences and predictions to train on.
            replay = self.memory.sample(self.minibatch_size)
            prediction = self.sess.run(self.q, {self.input: replay['old_obs']})

            # Compute the targets.
            future_q = self.target.run(self.q, {self.input: replay['new_obs']})
            future_q = np.amax(future_q, axis=1)
            future_q = future_q * np.invert(replay['done'])
            q_label = replay['reward'] + self.discount_factor * future_q

            # Train!
            self.sess.run(self.train, {
                self.input: replay['old_obs'],
                self.q_label: q_label,
                self.action_label: replay['action'],
            })

    def update_target(self):
        ckpt = self.save()
        self.load(ckpt)

    def save(self, ckpt=None):
        ckpt = ckpt or self.name + '.ckpt'
        return self.saver.save(self.sess, ckpt, global_step=self.global_step)

    def load(self, ckpt):
        if self.target is None:
            self.target = tf.Session(graph=self.graph)
        self.saver.restore(self.sess, ckpt)
        self.saver.restore(self.target, ckpt)

    def load_latest(self, ckpt_dir='.'):
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        self.load(ckpt)

    @csb.graph_property
    def input(self, scope):
        return tf.placeholder(tf.float32, (None, *self.obs_shape))

    @csb.graph_property
    def q(self, scope):
        defaults = {
            'activation': tf.nn.relu,
            'kernel_initializer': tf.contrib.layers.variance_scaling_initializer(),
        }
        y = self.input
        y = tf.layers.conv2d(y, filters=32, kernel_size=8, strides=4, **defaults)
        y = tf.layers.conv2d(y, filters=64, kernel_size=4, strides=2, **defaults)
        y = tf.layers.conv2d(y, filters=64, kernel_size=3, strides=1, **defaults)
        y = tf.contrib.layers.flatten(y)
        y = tf.layers.dense(y, units=512, **defaults)
        y = tf.layers.dense(y, units=self.n_actions, activation=None)
        return y

    @csb.graph_property
    def q_label(self, scope):
        return tf.placeholder(tf.float32, (None, ))

    @csb.graph_property
    def action_label(self, scope):
        return tf.placeholder(tf.int32, (None, ))

    @csb.graph_property
    def loss(self, scope):
        action_label = tf.one_hot(self.action_label, self.n_actions)
        q_label = tf.expand_dims(self.q_label, 1) * action_label
        q = self.q * action_label
        error = tf.losses.mean_squared_error(q_label, q)
        return tf.clip_by_value(error, -1, 1)

    @csb.graph_property
    def global_step(self, scope):
        return tf.train.create_global_step()

    @csb.graph_property
    def increment_global_step(self, scope):
        return tf.assign_add(self.global_step, 1)

    @csb.graph_property
    def exploration(self, scope):
        initial = float(self.exploration_initial)
        global_step = self.global_step
        steps = self.exploration_steps
        final = float(self.exploration_final)
        power = 1.0
        return tf.train.polynomial_decay(initial, global_step, steps, final, power)

    @csb.graph_property
    def train(self, scope):
        return self.optimizer.minimize(self.loss)

    @csb.graph_property
    def saver(self, scope):
        return tf.train.Saver()
