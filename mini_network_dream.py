import numpy as np
import tensorflow as tf
import os
import param as P

from algo.ppo import Policy_net, PPOTrain
from rnn.rnn_dream import reset_graph, ConvVAE, HyperParams, DreamModel

ACTION_SPACE = 10
SIZE_1 = 64                # image latent size
SIZE_2 = 20                # non-image obs feature size

model_rnn_size = 512
model_state_space = 2 # includes C and H concatenated if 2, otherwise just H


class MiniNetwork(object):

    def __init__(self, sess=None, summary_writer=tf.summary.create_file_writer("logs/"), rl_training=False,
                 reuse=False, cluster=None, index=0, device='/gpu:0',
                 ppo_load_path=None, ppo_save_path=None, load_worldmodel=True, ntype='dream-model'):
        self.policy_model_path_load = ppo_load_path + ntype
        self.policy_model_path_save = ppo_save_path + ntype
        self.ntype = ntype

        self.rl_training = rl_training

        self.use_norm = True

        self.reuse = reuse
        self.sess = sess
        self.cluster = cluster
        self.index = index
        self.device = device

        self.input_size = SIZE_1 + SIZE_2 + model_rnn_size*model_state_space

        self._create_graph()

        self.rl_saver = tf.train.Saver()
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset_old_network(self):
        self.policy_ppo.assign_policy_parameters()
        self.policy_ppo.reset_mean_returns()

        self.sess.run(self.results_sum.assign(0))
        self.sess.run(self.game_num.assign(0))

    def _create_graph(self):
        if self.reuse:
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse

        worker_device = "/job:worker/task:%d" % self.index + self.device
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=self.cluster)):
            self.results_sum = tf.get_variable(name="results_sum", shape=[], initializer=tf.zeros_initializer)
            self.game_num = tf.get_variable(name="game_num", shape=[], initializer=tf.zeros_initializer)

            self.global_steps = tf.get_variable(name="global_steps", shape=[], initializer=tf.zeros_initializer)
            self.win_rate = self.results_sum / self.game_num

            self.mean_win_rate = tf.summary.scalar('mean_win_rate_dis', self.results_sum / self.game_num)
            self.merged = tf.summary.merge([self.mean_win_rate])

            mini_scope = self.ntype
            with tf.variable_scope(mini_scope):
                ob_space = self.input_size
                act_space_array = ACTION_SPACE
                self.policy = Policy_net('policy', self.sess, ob_space, act_space_array)
                self.policy_old = Policy_net('old_policy', self.sess, ob_space, act_space_array)
                self.policy_ppo = PPOTrain('PPO', self.sess, self.policy, self.policy_old, lr=P.mini_lr, epoch_num=P.mini_epoch_num)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.policy_saver = tf.train.Saver(var_list=var_list)

    def Update_result(self, result_list):
        win = 0
        for i in result_list:
            if i > 0:
                win += 1
        self.sess.run(self.results_sum.assign_add(win))
        self.sess.run(self.game_num.assign_add(len(result_list)))

    def Update_summary(self, counter):
        print("Update summary........")

        policy_summary = self.policy_ppo.get_summary_dis()
        self.summary_writer.add_summary(policy_summary, counter)

        summary = self.sess.run(self.merged)
        self.summary_writer.add_summary(summary, counter)
        self.sess.run(self.global_steps.assign(counter))

        print("Update summary finished!")

        steps = int(self.sess.run(self.global_steps))
        win_game = int(self.sess.run(self.results_sum))
        all_game = int(self.sess.run(self.game_num))
        win_rate = win_game / float(all_game) if all_game != 0 else 0.

        return steps, win_rate

    def get_win_rate(self):
        return float(self.sess.run(self.win_rate))

    def Update_policy(self, buffer):
        self.policy_ppo.ppo_train_dis(buffer.observations, buffer.tech_actions,
                                      buffer.rewards, buffer.values, buffer.values_next, buffer.gaes, buffer.returns, verbose=True)

    def get_global_steps(self):
        return int(self.sess.run(self.global_steps))

    def save_policy(self):
        self.policy_saver.save(self.sess, self.policy_model_path_save)
        print("policy has been saved in", self.policy_model_path_save)

    def restore_policy(self):
        self.policy_saver.restore(self.sess, self.policy_model_path_load)
        print("Restore policy from", self.policy_model_path_load)


model_path_name = 'tf_models'
SIZE_1 = 64                # image latent size
SIZE_2 = 20                # non-image obs feature size
model_rnn_size = 512
model_num_mixture = 5
model_restart_factor = 10.

def default_hps():
  return HyperParams(num_steps=2000, # train model for 2000 steps.
                     max_seq_len=300, # train on sequences of 300
                     seq_width=SIZE_1,    # width of our data (64)
                     rnn_size=model_rnn_size,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=model_num_mixture,   # number of mixtures in MDN
                     restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)


hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=2, use_recurrent_dropout=0, is_training=0)


class SecondNetwork(object):
    
    def __init__(self, sess=None, rl_training=False, index=0,
                 reuse=False, cluster=None, device='/gpu:0',
                 load_model=True, net_path_name=model_path_name, ntype='assist-model'):
        self.index = index
        #reset_graph()     

        self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)
        self.rnn = DreamModel(hps_sample, gpu_mode=False, reuse=True)

        if load_model:
          self.vae.load_json(os.path.join(net_path_name, 'vae.json'))
          self.rnn.load_json(os.path.join(net_path_name, 'rnn.json'))

        self.outwidth = SIZE_1 + SIZE_2

    def rnn_init_state(self):
        return self.rnn.sess.run(self.rnn.initial_state)

    def rnn_next_state(self, feature, action, reward, prev_state):
        prev_feature = np.zeros((1, 1, self.outwidth))
        prev_feature[0][0] = feature

        prev_action = np.zeros((1, 1))
        prev_action[0] = action

        prev_reward = np.ones((1, 1))
        prev_reward[0] = reward

        feed = {self.rnn.input_z: prev_feature[:,:,:self.rnn.hps.seq_width],
                self.rnn.input_obs: prev_feature[:,:,self.rnn.hps.seq_width:],
                self.rnn.input_reward: prev_reward,
                self.rnn.input_action: prev_action,
                self.rnn.initial_state: prev_state
        }

        return self.rnn.sess.run(self.rnn.final_state, feed)


    def rnn_output(self, hidden_state, feature):
        return np.concatenate([feature, np.concatenate((hidden_state.c, hidden_state.h), axis=1)[0]])
