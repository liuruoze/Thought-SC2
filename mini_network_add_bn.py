import numpy as np
import tensorflow as tf

import param as P

from algo.ppo_add_bn import Policy_net, PPOTrain

# for mini game
_SIZE_MINI_INPUT = 20
_SIZE_MINI_ACTIONS = 10

class MiniNetwork(object):

    def __init__(self, sess=None, summary_writer=tf.summary.FileWriter("logs/"), rl_training=False,
                 reuse=False, cluster=None, index=0, device='/gpu:0',
                 ppo_load_path=None, ppo_save_path=None, 
                 ob_space_add=0, act_space_add=0, freeze_head=False, use_bn=False, use_sep_net=True,
                 restore_model=False, restore_from=None, restore_to=None):
        self.policy_model_path_load = ppo_load_path + "mini"
        self.policy_model_path_save = ppo_save_path + "mini"
        print("Hello !")
        self.rl_training = rl_training

        self.use_norm = True

        self.reuse = reuse
        self.sess = sess
        self.cluster = cluster
        self.index = index
        self.device = device

        self.ob_space_add = ob_space_add
        self.act_space_add = act_space_add
        self.freeze_head = freeze_head
        self.use_bn = use_bn
        self.use_sep_net = use_sep_net
        print("Hello !")
        self.restore_model = restore_model
        self.restore_from = restore_from
        self.restore_to = restore_to

        if self.ob_space_add == 0 and self.act_space_add == 0:
            self.use_add = False
            self.lr=P.mini_lr
            self.epoch_num = P.mini_epoch_num
        else:
            self.use_add = True
            self.lr=P.mini_lr_add
            self.epoch_num = P.mini_epoch_num
            
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
            print("Reuse !")
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse
        
        print("Hello !")
            
        worker_device = "/job:worker/task:%d" % self.index + self.device
        with tf.device(tf.train.replica_device_setter(worker_device=worker_device, cluster=self.cluster)):
            self.results_sum = tf.get_variable(name="results_sum", shape=[], initializer=tf.zeros_initializer)
            self.game_num = tf.get_variable(name="game_num", shape=[], initializer=tf.zeros_initializer)

            self.global_steps = tf.get_variable(name="global_steps", shape=[], initializer=tf.zeros_initializer)
            self.win_rate = self.results_sum / self.game_num

            self.mean_win_rate = tf.summary.scalar('mean_win_rate_dis', self.results_sum / self.game_num)
            self.merged = tf.summary.merge([self.mean_win_rate])

            mini_scope = "MiniPolicyNN"
            with tf.variable_scope(mini_scope):
                ob_space = _SIZE_MINI_INPUT
                act_space_array = _SIZE_MINI_ACTIONS
                self.policy = Policy_net('policy', self.sess, ob_space, self.ob_space_add, 
                    act_space_array, self.act_space_add, self.freeze_head, self.use_bn, self.use_sep_net)
                self.policy_old = Policy_net('old_policy', self.sess, ob_space, self.ob_space_add, 
                    act_space_array, self.act_space_add, self.freeze_head, self.use_bn, self.use_sep_net)
                self.policy_ppo = PPOTrain('PPO', self.sess, self.policy, self.policy_old, lr=self.lr, epoch_num=self.epoch_num)
            
            var_train_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var_all_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            if self.restore_model:
                print('restore_model')
                if self.restore_from == 'mini' and self.restore_to == 'mini':
                    print('restore_model: mini to mini')
                    self.old_policy_saver = tf.train.Saver(var_list=variables_to_restore)
                elif self.restore_from == 'mini' and self.restore_to == 'source':
                    print('restore_model: mini to source')    
                    if self.use_add:
                        print('restore_model: mini to source, use_add')
                        variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy.scope)
                        old_variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.policy_old.scope)
                        variables_to_restore += old_variables_to_restore
           
                        variables_to_restore = [v for v in variables_to_restore if len(v.name.split('/')) > 2 and 'DenseLayer3' not in v.name.split('/')]
                        variables_to_restore = [v for v in variables_to_restore if len(v.name.split('/')) > 2 and v.name.split('/')[-2] != 'add_output_layer']
                        variables_to_restore = [v for v in variables_to_restore if len(v.name.split('/')) > 2 and 
                            'AdaptiveWeight:0' not in v.name.split('/') and 'AttentionWeight' not in v.name.split('/')]
                        print('variables_to_restore:', variables_to_restore)
                        self.old_policy_saver = tf.train.Saver(var_list=variables_to_restore)
                    else:
                        self.old_policy_saver = tf.train.Saver(var_list=var_all_list)
                elif self.restore_from == 'source' and self.restore_to == 'source':
                    self.old_policy_saver = tf.train.Saver(var_list=var_all_list)                
                else:
                    self.old_policy_saver = tf.train.Saver(var_list=var_all_list)
            else:
                self.old_policy_saver = tf.train.Saver(var_list=var_all_list)

            self.new_policy_saver = tf.train.Saver(var_list=var_all_list)
            

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

        if counter % 1 == 0:
            gradient_summary = self.policy_ppo.get_summary_gradient()
            if gradient_summary:
                self.summary_writer.add_summary(gradient_summary, counter)
            else:
                print("gradient_summary is None")

        summary = self.sess.run(self.merged)
        self.summary_writer.add_summary(summary, counter)
        self.sess.run(self.global_steps.assign(counter))

        print("Update summary finished!")

        steps = int(self.sess.run(self.global_steps))
        win_game = int(self.sess.run(self.results_sum))
        all_game = int(self.sess.run(self.game_num))
        #print('all_game:', all_game)
        win_rate = win_game / float(all_game) if all_game != 0 else 0.

        return steps, win_rate

    def get_win_rate(self):
        return float(self.sess.run(self.win_rate))

    def Update_policy(self, buffer, verbose=False):
        self.policy_ppo.ppo_train_dis(buffer.observations, buffer.obs_add, buffer.tech_actions,
                                      buffer.rewards, buffer.values, buffer.values_next, buffer.gaes, buffer.returns, 
                                      buffer.return_values, self.index, self.summary_writer, verbose=verbose)

    def get_global_steps(self):
        return int(self.sess.run(self.global_steps))

    def save_policy(self):
        self.new_policy_saver.save(self.sess, self.policy_model_path_save)
        print("policy has been saved in", self.policy_model_path_save)

    def restore_policy(self):
        self.old_policy_saver.restore(self.sess, self.policy_model_path_load)
        print("Restore policy from", self.policy_model_path_load)
