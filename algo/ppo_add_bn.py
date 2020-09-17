import tensorflow as tf
import numpy as np
import copy
import lib.layer as layer
import param as P


# This part of the code was borrowed mainly from the PPO part of
# https://github.com/uidilr/gail_ppo_tf and was partially modified.

class Policy_net:

    def __init__(self, name: str, sess, ob_space, add_ob_space, act_space_array, add_act_space, freeze_head=False, use_bn=True,
        use_sep_net=True, activation=tf.nn.relu):
        """
        :param name: string
        """
        self.sess = sess
        self.add_weight = 0.05

        self.weight_type = P.weight_type

        self.freeze_head = freeze_head
        self.use_bn = use_bn
        self.activation = activation
        self.use_sep_net = use_sep_net

        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, ob_space], name='obs')

        if add_ob_space > 0:
            self.obs_add = tf.placeholder(dtype=tf.float32, shape=[None, add_ob_space], name='obs_add')
            self.use_add_obs = True
        else:
            self.obs_add = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='obs_add')
            self.use_add_obs = False

        self.act_space_array = act_space_array
        self.add_act_space = add_act_space

        if self.use_sep_net:
            self.network = self.sep_policy_value_net
        else:
            self.network = self.dual_policy_value_net


        self.train_probs, self.train_act, self.train_v_preds = self.network(name, freeze_head=self.freeze_head, norm=self.use_bn)
        self.test_probs, self.test_act, self.test_v_preds = self.network(name, freeze_head=self.freeze_head, norm=self.use_bn, 
            is_training=False, reuse=True)



    def sep_policy_value_net(self, name, hidden_units=64, 
            activation=tf.nn.relu, freeze_head=False, initial_type='original', norm=True, is_training=True, reuse=False):
        
        with tf.variable_scope(name, reuse=reuse):

            with tf.variable_scope('policy_net'):
                with tf.variable_scope('controller'):              
                    layer_1 = layer.dense_layer(self.obs, hidden_units, "DenseLayer1", norm=norm, is_training=is_training, 
                        func=activation, initial_type=initial_type, trainable=not freeze_head)
                    self.layer_2 = layer.dense_layer(layer_1, hidden_units, "DenseLayer2", norm=norm, is_training=is_training, 
                        func=activation, initial_type=initial_type, trainable=not freeze_head)

                    # weighted sum
                    if self.use_add_obs:
                        self.layer_3 = layer.dense_layer(self.obs_add, hidden_units, "DenseLayer3", norm=norm, is_training=is_training, 
                            func=activation, initial_type=initial_type)

                        if self.weight_type == 'AttentionWeight':
                            self.attention_weight = layer.dense_layer(self.layer_3, hidden_units, "AttentionWeight", initial_type='original', func=tf.nn.sigmoid)
                            #self.attention_weight = tf.nn.sigmoid(self.layer_3)
                            self.layer_4 = self.layer_2 * self.attention_weight

                        elif self.weight_type == 'AdaptiveWeight':
                            self.adaptive_weight = tf.get_variable(name="AdaptiveWeight", shape=[hidden_units], 
                                initializer=tf.zeros_initializer(), trainable=True)                    
                            self.layer_4 = (1 - self.adaptive_weight) * self.layer_2 + self.adaptive_weight * self.layer_3

                        elif self.weight_type == 'AddWeight':
                            self.layer_4 = (1. -  self.add_weight) * self.layer_2 + self.add_weight * self.layer_3

                        else:
                            self.layer_4 = self.layer_2 + self.layer_3
                    else:
                        self.layer_4 = self.layer_2

                    #self.train_probs = layer.dense_layer(layer_2, act_space_array, "output", func=tf.nn.softmax)
                    probs = layer.output_layer(self.layer_4, self.act_space_array, self.add_act_space, "output", 
                        is_training=is_training, initial_type=initial_type,
                        func=tf.nn.softmax)

                    act = tf.multinomial(tf.log(probs), num_samples=1)
                    act = tf.reshape(act, shape=[-1])

            with tf.variable_scope('value_net'):
                layer_1 = layer.dense_layer(self.obs, hidden_units, "DenseLayer1", norm=norm, is_training=is_training, 
                    func=activation, initial_type=initial_type, trainable=not freeze_head)
                layer_2 = layer.dense_layer(layer_1, hidden_units, "DenseLayer2", norm=norm, is_training=is_training, 
                    func=activation, initial_type=initial_type, trainable=not freeze_head)

                v_preds = layer.dense_layer(layer_2, 1, "DenseLayer4", initial_type=initial_type, is_training=is_training, func=None)
        
            self.scope = tf.get_variable_scope().name

        return probs, act, v_preds


    def dual_policy_value_net(self, name, hidden_units=64, 
        activation=tf.nn.relu, freeze_head=False, initial_type='original', norm=True, is_training=True, reuse=False):
        
        with tf.variable_scope(name, reuse=reuse):
            with tf.variable_scope('policy_net'):
                with tf.variable_scope('controller'):              
                    layer_1 = layer.dense_layer(self.obs, hidden_units, "DenseLayer1", norm=norm, is_training=is_training, 
                        func=activation, initial_type=initial_type, trainable=not freeze_head)
                    layer_2 = layer.dense_layer(layer_1, hidden_units, "DenseLayer2", norm=norm, is_training=is_training, 
                        func=activation, initial_type=initial_type, trainable=not freeze_head)
                    
                    # weighted sum
                    if self.use_add_obs:
                        self.layer_3 = layer.dense_layer(self.obs_add, hidden_units, "DenseLayer3", norm=norm, is_training=is_training, 
                            func=activation, initial_type=initial_type)

                        if self.weight_type == 'AttentionWeight':
                            self.attention_weight = layer.dense_layer(self.layer_3, hidden_units, "AttentionWeight", initial_type='original', func=tf.nn.sigmoid)
                            #self.attention_weight = tf.nn.sigmoid(self.layer_3)
                            self.layer_4 = self.layer_2 * self.attention_weight

                        elif self.weight_type == 'AdaptiveWeight':
                            self.adaptive_weight = tf.get_variable(name="AdaptiveWeight", shape=[hidden_units], 
                                initializer=tf.zeros_initializer(), trainable=True)                    
                            self.layer_4 = (1 - self.adaptive_weight) * self.layer_2 + self.adaptive_weight * self.layer_3

                        elif self.weight_type == 'AddWeight':
                            self.layer_4 = (1. -  self.add_weight) * self.layer_2 + self.add_weight * self.layer_3

                        else:
                            self.layer_4 = self.layer_2 + self.layer_3
                    else:
                        self.layer_4 = self.layer_2

                    #self.train_probs = layer.dense_layer(layer_2, act_space_array, "output", func=tf.nn.softmax)
                    probs = layer.output_layer(self.layer_4, self.act_space_array, self.add_act_space, "output", 
                        is_training=is_training, initial_type=initial_type,
                        func=tf.nn.softmax)

                    act = tf.multinomial(tf.log(probs), num_samples=1)
                    act = tf.reshape(act, shape=[-1])

                    v_preds = layer.dense_layer(layer_4, 1, "DenseLayer4", initial_type=initial_type, is_training=is_training, func=None)
        
            self.scope = tf.get_variable_scope().name

        return probs, act, v_preds
        

    def get_action(self, obs, obs_add, verbose=True):
        act_probs, act, v_preds \
            = self.sess.run([self.test_probs, self.test_act, self.test_v_preds], feed_dict={self.obs: obs.reshape([1, -1])
                ,self.obs_add: obs_add.reshape([1, -1])})

        if verbose:
            print("Tech:", 'act_probs:', act_probs, 'act:', act)
            print("Value:", v_preds)
        return act[0], np.asscalar(v_preds)

    def get_action_more(self, obs, obs_add, verbose=True):
        if not self.use_add_obs:
            return self.get_action(obs, obs_add, verbose=verbose)

        act_probs, act, v_preds, layer_2, layer_3, layer_4 \
            = self.sess.run([self.test_probs, self.test_act, self.test_v_preds, self.layer_2, self.layer_3, self.layer_4], 
                feed_dict={self.obs: obs.reshape([1, -1])
                , self.obs_add: obs_add.reshape([1, -1])})

        if verbose:
            print("Tech:", 'act_probs:', act_probs, 'act:', act)
            print("Value:", v_preds)
            print("layer_2:", layer_2)
            print("layer_3:", layer_3)
            print("layer_4:", layer_4) 
        return act[0], np.asscalar(v_preds)

    def get_action_probs(self, obs, obs_add, verbose=True):
        act_probs, act, v_preds \
            = self.sess.run([self.test_probs, self.test_act, self.test_v_preds], feed_dict={self.obs: obs.reshape([1, -1])
                ,self.obs_add: obs_add.reshape([1, -1])})

        if verbose:
            print("obs:", obs)
            print("Tech:", 'act_probs:', act_probs, 'act:', act)
            print("Value:", v_preds)
        return act_probs[0], np.asscalar(v_preds)

    def get_values(self, obs, obs_add):
        v_preds = self.sess.run(self.test_v_preds, feed_dict={self.obs: obs.reshape([1, -1])
                ,self.obs_add: obs_add.reshape([1, -1])})
        v_preds = np.asscalar(v_preds)
        return v_preds

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOTrain:

    def __init__(self, name, sess, Policy, Old_Policy, gamma=0.995, clip_value=0.2, c_1=0.01, c_2=1e-6, lr=1e-4, epoch_num=20):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        :param epoch_num: num for update
        """
        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.sess = sess

        # comman param for all ppo
        self.gamma = P.gamma
        self.lamda = P.lamda
        self.batch_size = P.batch_size
        self.clip_value = P.clip_value
        self.c_1 = P.c_1
        self.c_2 = P.c_2

        # for different network
        self.adam_lr = lr
        self.epoch_num = epoch_num

        self.adam_epsilon = 1e-5
        self.update_count = 0
        self.restore_model = P.restore_model

        self.gradients_values = None
        self.grad_summ_summary = None
        self.print_gradient_step = 10

        with tf.variable_scope(name):
            pi_trainable = self.Policy.get_trainable_variables()
            old_pi_trainable = self.Old_Policy.get_trainable_variables()

            pi_global = self.Policy.get_variables()
            old_pi_global = self.Old_Policy.get_variables()

            assign_global = True
            if assign_global:
                pi = pi_global
                old_pi = old_pi_global
            else:
                pi = pi_trainable
                old_pi = old_pi_trainable

            # assign_operations for policy parameter values to old policy parameters
            with tf.variable_scope('assign_op'):
                self.assign_ops = []
                #for v_old, v in zip(old_pi_trainable, pi_trainable):
                for v_old, v in zip(old_pi, pi):
                    self.assign_ops.append(tf.assign(v_old, v))

            # inputs for train_op
            with tf.variable_scope('train_inp'):
                self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
                self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
                self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
                self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
                self.returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')

                # define distribute variable
                self.returns_sum = tf.get_variable(name="returns_sum", shape=[], initializer=tf.zeros_initializer)
                self.proc_num = tf.get_variable(name="proc_num", shape=[], initializer=tf.zeros_initializer)

            act_probs = self.Policy.train_probs
            act_probs_old = self.Old_Policy.train_probs

            # probabilities of actions which agent took with policy
            act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
            act_probs = tf.reduce_sum(act_probs, axis=1)

            act_probs = act_probs

            # probabilities of actions which agent took with old policy
            act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
            act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

            act_probs_old = act_probs_old

            with tf.variable_scope('loss'):
                # construct computation graph for loss_clip
                # ratios = tf.divide(act_probs, act_probs_old)
                ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                                - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
                clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value,
                                                  clip_value_max=1 + self.clip_value)
                loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
                self.loss_clip = -tf.reduce_mean(loss_clip)

                # construct computation graph for loss of entropy bonus
                entropy = -tf.reduce_sum(self.Policy.train_probs *
                                              tf.log(tf.clip_by_value(self.Policy.train_probs, 1e-10, 1.0)), axis=1)
                entropy = entropy
                self.entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)

                # construct computation graph for loss of value function
                v_preds = self.Policy.train_v_preds
                if P.use_return_error:
                    loss_vf = tf.squared_difference(self.returns, v_preds)
                else:
                    loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
                self.loss_vf = tf.reduce_mean(loss_vf)

                # construct computation graph for loss
                self.total_loss = self.loss_clip + self.c_1 * self.loss_vf - self.c_2 * self.entropy
                self.sum_mean_returns = tf.summary.scalar('mean_return_dis', self.returns_sum / (self.proc_num + 0.0001))

            self.merged_dis = tf.summary.merge([self.sum_mean_returns])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.adam_lr, epsilon=self.adam_epsilon)
            self.gradients = optimizer.compute_gradients(self.total_loss, var_list=pi_trainable)

            # plot the gradients
            self.grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name.replace(':','_'), g[0]) for g in self.gradients
                if 'batch_norm' not in g[1].name.split('/')])

            self.train_op = optimizer.minimize(self.total_loss, var_list=pi_trainable)
            self.train_value_op = optimizer.minimize(self.loss_vf, var_list=pi_trainable)

    def train(self, obs, obs_add, actions, gaes, rewards, v_preds_next, returns):
        _, total_loss = self.sess.run([self.train_op, self.total_loss], feed_dict={self.Policy.obs: obs,
                                                                                   self.Policy.obs_add: obs_add, 
                                                                                   self.Old_Policy.obs: obs,
                                                                                   self.Old_Policy.obs_add: obs_add, 
                                                                                   self.actions: actions,
                                                                                   self.rewards: rewards,
                                                                                   self.v_preds_next: v_preds_next,
                                                                                   self.gaes: gaes,
                                                                                   self.returns: returns})
        return total_loss

    def train_value(self, obs, obs_add, gaes, rewards, v_preds_next, returns):
        _, value_loss = self.sess.run([self.train_value_op, self.loss_vf], feed_dict={self.Policy.obs: obs,
                                                                                      self.Policy.obs_add: obs_add, 
                                                                                      self.Old_Policy.obs: obs,
                                                                                      self.Old_Policy.obs_add: obs_add, 
                                                                                      self.rewards: rewards,
                                                                                      self.v_preds_next: v_preds_next,
                                                                                      self.gaes: gaes,
                                                                                      self.returns: returns})
        return value_loss



    def train_with_all(self, obs, obs_add, actions, gaes, rewards, v_preds_next, returns):
        _, total_loss, loss_clip, loss_vf, entropy, gradients, grad_summ_summary = self.sess.run([self.train_op, self.total_loss, 
                                                                self.loss_clip, self.loss_vf, self.entropy, self.gradients,
                                                                self.grad_summ_op], 
                                                                            feed_dict={self.Policy.obs: obs,
                                                                                      self.Policy.obs_add: obs_add, 
                                                                                      self.Old_Policy.obs: obs,
                                                                                      self.Old_Policy.obs_add: obs_add,
                                                                                   self.actions: actions,
                                                                                   self.rewards: rewards,
                                                                                   self.v_preds_next: v_preds_next,
                                                                                   self.gaes: gaes,
                                                                                   self.returns: returns})
        return total_loss, loss_clip, loss_vf, entropy, gradients, grad_summ_summary

    def get_summary_dis(self):
        return self.sess.run(self.merged_dis)

    def get_summary_gradient(self):
        #return self.sess.run(self.grad_summ_op)
        return None

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return self.sess.run(self.assign_ops)

    def reset_mean_returns(self):
        self.sess.run(self.returns_sum.assign(0))
        self.sess.run(self.proc_num.assign(0))

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * self.lamda * gaes[t + 1]
        return gaes

    def ppo_train_dis(self, observations, obs_add, actions, rewards, v_preds, v_preds_next,
                      gaes, returns, return_values, index, summary_writer, verbose=False):
        if verbose:
            print('PPO train now..........')

        # convert list to numpy array for feeding tf.placeholder
        observations = np.array(observations).astype(dtype=np.float32)
        obs_add = np.array(obs_add).astype(dtype=np.float32)
        actions = np.array(actions).astype(dtype=np.int32)

        gaes = np.array(gaes).astype(dtype=np.float32).reshape(-1)
        if P.use_adv_norm:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-10)
        
        return_values = np.array(return_values).astype(dtype=np.float32).reshape(-1)
        if P.use_return_norm:
            return_values = (return_values - return_values.mean()) / (return_values.std() + 1e-10)

        rewards = np.array(rewards).astype(dtype=np.float32).reshape(-1)
        v_preds_next = np.array(v_preds_next).astype(dtype=np.float32).reshape(-1)
        inp = [observations, actions, gaes, rewards, v_preds_next, obs_add, return_values]

        train_num = observations.shape[0]
        if train_num <= 0:
            return

        # self.assign_policy_parameters()
        # train
        # batch_size = max(observations.shape[0] // 10, self.batch_size)
        batch_size = min(train_num, self.batch_size)
        if verbose:
            print('batch_size is:', batch_size)

        max_steps = train_num // batch_size + 1
        if verbose:
            print('max_steps is:', max_steps)

        total_loss, loss_clip, loss_vf, entropy = None, None, None, None
        for epoch in range(self.epoch_num):
            for step in range(max_steps):
                # sample indices from [low, high)
                sample_indices = np.random.randint(low=0, high=train_num, size=batch_size)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data

                if self.restore_model and self.update_count < 3:
                    value_loss = self.train_value(obs=sampled_inp[0], obs_add=sampled_inp[5],
                                                  gaes=sampled_inp[2], returns=sampled_inp[6],
                                                  rewards=sampled_inp[3],
                                                  v_preds_next=sampled_inp[4])
                else:
                    total_loss, loss_clip, loss_vf, entropy, self.gradients_values, self.grad_summ_summary = self.train_with_all(
                                            obs=sampled_inp[0], obs_add=sampled_inp[5],
                                            actions=sampled_inp[1],
                                            gaes=sampled_inp[2], returns=sampled_inp[6],
                                            rewards=sampled_inp[3],
                                            v_preds_next=sampled_inp[4])
        if self.update_count % 1 == 0 and total_loss is not None:
            print("total_loss:", total_loss, 'loss_clip:', loss_clip, "loss_vf:", loss_vf, "entropy:", entropy)
        if self.update_count % self.print_gradient_step == 0 and self.gradients_values is not None:
            print('self.gradients_values:')

            if index == 0:
                summary_writer.add_summary(self.grad_summ_summary, self.update_count)
            #print('self.gradients_values:')
            for i, gv in enumerate(self.gradients_values):
                print(self.gradients[i][1].name, ":", str(gv))

        self.update_count += 1

        if verbose:
            print('np.mean(returns).shape:', np.mean(returns).shape)

        if len(np.mean(returns).shape) > 0:
            print("returns:", returns)

        self.sess.run(self.returns_sum.assign_add(np.mean(returns)))
        self.sess.run(self.proc_num.assign_add(1))

        print("ppo.returns_sum", self.sess.run(self.returns_sum))


        if verbose:
            print('PPO train end..........')
        return
