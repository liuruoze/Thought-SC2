import tensorflow as tf
import numpy as np
import lib.layer as layer
import lib.utils as U
import lib.config as C
import platform
from datetime import datetime
from sklearn import preprocessing


class DynamicNetwork(object):

    def __init__(self, name, sess, load_path=None, save_path=None, rl_training=False,
                 sl_training=True, reuse=False):
        self.system = platform.system()

        now = datetime.now()
        model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_dynamic/"
        if load_path is None:
            load_path = model_path
        if save_path is None:
            save_path = model_path

        self.tech_model_path_load = load_path + "probe"  # model_path  # "model/tech/probe"
        self.tech_model_path_save = save_path + "probe"

        self.sl_training = sl_training

        self.use_norm = False
        self.save_data = False
        self.predict_diff = True
        self.use_norm_input = True
        self.use_h_step_val = True
        self.use_norm_diff = False

        self.use_rule = True

        self.H = 5
        self.mean_and_std = [0, 0]
        self.norm_diff = None
        self.map_width = 64
        self.save_data_path = "data/simple/"

        self.reuse = reuse
        self.sess = sess
        with tf.variable_scope(name):
            self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")
            self._create_graph()
            self.scope = tf.get_variable_scope().name

        self.summary = tf.Summary()
        self.summary_op = tf.summary.merge_all()

        log_path = save_path.replace("model", "logs")
        self.summary_writer = tf.summary.create_file_writer(log_path, self.sess.graph)
        self._define_sl_saver()

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _define_sl_saver(self):
        self.tech_var_list_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        self.tech_saver = tf.train.Saver(var_list=self.tech_var_list_save)

    def _create_graph(self):

        if self.reuse:
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse

        with tf.name_scope("Input"):
            # tech net:
            self.last_state = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SIMPLE_INPUT], name="last_state")
            self.last_action = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_MAX_ACTIONS], name="last_action")

            #self.tech_input = tf.concat([self.last_state, self.last_action], axis=1, name="tech_input")
            self.now_state = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SIMPLE_INPUT], name="now_state")

            self.state_diff = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SIMPLE_INPUT], name="state_diff")
            self.ruled_diff = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SIMPLE_INPUT], name="predict_diff")

            if self.use_rule:
                self.tech_input = tf.concat([self.last_state, self.last_action, self.ruled_diff], axis=1, name="tech_input")
                self.tech_label = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tech_label")
            else:
                self.tech_input = tf.concat([self.last_state, self.last_action], axis=1, name="tech_input")
                self.tech_label = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SIMPLE_INPUT], name="tech_label")

            self.tech_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="tech_lr")

        with tf.name_scope("Network"):
            self.tech_predict, self.tech_net_scope = self._tech_net(self.tech_input)
            print("self.tech_net_scope:", self.tech_net_scope)

        if self.sl_training:
            with tf.name_scope("SL_loss"):
                self._define_sl_loss()

    def _define_sl_loss(self):
        # tech loss
        self.tech_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tech_net_scope)
        # print(self.tech_var_list_train)

        with tf.name_scope("Tech_loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.tech_predict, self.tech_label))
            #self.regularizer = tf.nn.l2_loss(self.tech_var_list_train)
            self.regularizer = tf.add_n([tf.nn.l2_loss(v) for v in self.tech_var_list_train
                                         if 'bias' not in v.name])
            self.beta = 0.01
            self.tech_loss = self.loss + self.beta * self.regularizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.tech_net_scope)

            with tf.control_dependencies(update_ops):
                self.tech_train_step = tf.train.AdamOptimizer(self.tech_lr_ph).minimize(
                    self.tech_loss, var_list=self.tech_var_list_train)

            with tf.name_scope("summary_tech"):
                self.tech_loss_sum = tf.summary.scalar('tech_loss', self.tech_loss)
                self.summary_tech_op = tf.summary.merge([self.tech_loss_sum])

    def _tech_net(self, data, trainable=True):
        with tf.variable_scope("Tech_net"):
            # if self.use_norm == True:
            #    data = layer.batch_norm(data, self.is_training, 'BN')
            d1 = layer.dense_layer(data, 256, "DenseLayer1", is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            d2 = layer.dense_layer(d1, 128, "DenseLayer2", is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            if self.use_rule:
                dout = layer.dense_layer(d2, 1, "DenseLayerOut", func=None, is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            else:
                dout = layer.dense_layer(d2, C._SIZE_SIMPLE_INPUT, "DenseLayerOut", func=None, is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            scope = tf.get_variable_scope().name
        return dout, scope

    def transform_state(self, state):
        scaler = preprocessing.StandardScaler()
        #print('mean_and_std', self.mean_and_std)
        scaler.mean_ = self.mean_and_std[0]
        scaler.scale_ = self.mean_and_std[1]
        return scaler.transform(state.reshape([1, -1])).reshape(-1)

    def inverse_transform_state(self, scaled_state):
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = self.mean_and_std[0]
        scaler.scale_ = self.mean_and_std[1]
        return scaler.inverse_transform(scaled_state.reshape([1, -1])).reshape(-1)

    def predict_tech(self, last_state, last_action):
        if self.use_rule:
            return self.predict_tech_by_rule(last_state, last_action)
        else:
            return self.predict_tech_by_learning(last_state, last_action)

    def predict_tech_by_rule(self, last_state, last_action, use_transform=True):
        if self.use_norm_input and use_transform:
            input_state = self.transform_state(last_state)
        else:
            input_state = last_state

        rule_state_diff = U.predict_state_diff_by_rule(last_state, last_action)

        last_action_one_hot = self.one_hot_label(np.asarray([last_action]), C._SIZE_MAX_ACTIONS)
        feed_dict = {self.last_state: input_state.reshape([1, -1]),
                     self.last_action: last_action_one_hot,
                     self.ruled_diff: rule_state_diff.reshape([1, -1]),
                     self.is_training: self.sl_training,
                     }

        mineral_diff_pred = self.tech_predict.eval(feed_dict, session=self.sess).reshape(-1)[0]

        next_state = last_state.reshape(-1) + rule_state_diff

        mineral_index = 1
        next_state[mineral_index] -= rule_state_diff[mineral_index]
        next_state[mineral_index] += mineral_diff_pred

        return next_state

    def predict_tech_by_learning(self, last_state, last_action):
        # TODO add transform_state
        if self.use_norm_input and True:
            input_state = self.transform_state(last_state)
        else:
            input_state = last_state

        last_action_one_hot = self.one_hot_label(np.asarray([last_action]), C._SIZE_MAX_ACTIONS)
        # print(last_action_one_hot)
        feed_dict = {self.last_state: input_state.reshape([1, -1]),
                     self.last_action: last_action_one_hot,
                     self.is_training: self.sl_training,
                     }
        tech_pred = self.tech_predict.eval(feed_dict, session=self.sess)
        if self.predict_diff:
            tech_pred += last_state
        return tech_pred.reshape(-1)

    def SL_train_tech_net(self, observations, tech_actions, next_observations, batch_size=1000, iter_num=100,
                          lr=1e-4, rate=0.6, use_val=True):
        print('SL_train_tech_net begin')
        sample_num = observations.shape[0]
        train_num = int(sample_num * rate)
        print('train_num:', train_num)
        overall_step = 0
        diff_observations = next_observations - observations
        predict_diff_observations = np.zeros([sample_num, C._SIZE_SIMPLE_INPUT])
        for idx in range(sample_num):
            predict_diff_observations[idx] = U.predict_state_diff_by_rule(observations[idx], tech_actions[idx])

        print('predict_diff_observations:', predict_diff_observations[0])
        print('predict_diff_observations:', predict_diff_observations[-1])

        if self.use_norm_input:
            scaler_obs = preprocessing.StandardScaler().fit(observations[0:train_num - 1, :])
            obs_mean, obs_std = scaler_obs.mean_, scaler_obs.scale_
            self.mean_and_std = np.array([obs_mean, obs_std])

            self.tech_saver.save(self.sess, self.tech_model_path_save)
            self.save_preprocessing(self.mean_and_std)
            observations = scaler_obs.transform(observations)

        for iter_index in range(iter_num):
            step = 0
            while step * batch_size <= train_num:
                begin_index, end_index = step * batch_size, min((step + 1) * batch_size, train_num - 1)
                random = np.random.random_sample(batch_size) * train_num
                random_indexs = np.array(random).astype(dtype=np.int32)

                batch_observations = observations[random_indexs, :]
                batch_tech_actions = self.one_hot_label(tech_actions[random_indexs], C._SIZE_MAX_ACTIONS)
                batch_next_observations = next_observations[random_indexs, :]
                batch_diff_observations = diff_observations[random_indexs, :]
                if self.use_rule:
                    batch_tech_label = batch_diff_observations[:, 1:2]
                else:
                    batch_tech_label = batch_diff_observations

                batch_predicted_diff = predict_diff_observations[random_indexs, :]
                feed_dict = {
                    self.last_state: batch_observations,
                    self.last_action: batch_tech_actions,
                    self.now_state: batch_next_observations,
                    self.state_diff: batch_diff_observations,
                    self.tech_label: batch_tech_label,
                    self.ruled_diff: batch_predicted_diff,
                    self.tech_lr_ph: lr,
                    self.is_training: self.sl_training,
                }
                _, loss, regularizer, tech_loss, predict, summary_str = self.sess.run(
                    [self.tech_train_step, self.loss, self.regularizer, self.tech_loss, self.tech_predict, self.summary_tech_op], feed_dict=feed_dict)
                step += 1

                if overall_step % 50 == 0:
                    print("loss:", loss, "regularizer:", regularizer)
                    print("tech: epoch: %d/%d, overall_step: %d, tech_loss: " % (iter_index + 1, iter_num, overall_step), tech_loss)
                    batch_int = np.array(batch_diff_observations).astype(dtype=np.int32)
                    print(batch_tech_actions[0])
                    print(batch_int[0][1])
                    predict_int = np.array(predict).astype(dtype=np.int32)
                    print(predict_int[0])

                    # begin validateion
                    if use_val:
                        val_step = 0
                        val_mean_loss = []
                        while val_step * batch_size + train_num <= sample_num:
                            v_begin_index, v_end_index = val_step * batch_size + train_num, min((val_step + 1) * batch_size + train_num, sample_num - 1)

                            val_batch_observations = observations[v_begin_index: v_end_index, :]
                            val_batch_tech_actions = self.one_hot_label(tech_actions[v_begin_index: v_end_index], C._SIZE_MAX_ACTIONS)
                            val_batch_next_observations = next_observations[v_begin_index: v_end_index, :]
                            val_batch_diff_observations = diff_observations[v_begin_index: v_end_index, :]
                            if self.use_rule:
                                val_batch_tech_label = val_batch_diff_observations[:, 1:2]
                            else:
                                val_batch_tech_label = val_batch_diff_observations

                            val_batch_predicted_diff = predict_diff_observations[v_begin_index: v_end_index, :]
                            feed_dict = {
                                self.last_state: val_batch_observations,
                                self.last_action: val_batch_tech_actions,
                                self.now_state: val_batch_next_observations,
                                self.state_diff: val_batch_diff_observations,
                                self.tech_label: val_batch_tech_label,
                                self.ruled_diff: val_batch_predicted_diff,
                                self.tech_lr_ph: lr,
                                self.is_training: False,
                            }

                            tech_loss, predict = self.sess.run([self.tech_loss, self.tech_predict], feed_dict=feed_dict)
                            val_step += 1
                            val_mean_loss.append(tech_loss)

                        val_loss = sum(val_mean_loss) / float(len(val_mean_loss))
                        print("tech: epoch: %d/%d, val_mean_loss: " % (iter_index + 1, iter_num), val_loss)
                        summary = tf.Summary(value=[
                            tf.Summary.Value(tag="val_loss", simple_value=val_loss),
                        ])
                        self.summary_writer.add_summary(summary, overall_step)

                    if overall_step % 500 == 0:
                        if self.use_h_step_val:  # if use H-step-prediction error
                            print("begin h_step_val:")
                            count = 0.
                            mean_error = 0.
                            test_num = 10000
                            begin, end = train_num, sample_num - self.H
                            val_data_num = end - begin
                            random_val = np.random.random_sample(test_num) * val_data_num + begin
                            random_val = np.array(random_val).astype(dtype=np.int32)
                            #print('random_val:', random_val)
                            for i in range(test_num):
                                v_begin_index, v_end_index = random_val[i], random_val[i] + self.H
                                val_batch_observations = observations[v_begin_index: v_end_index, :]
                                val_batch_tech_actions = tech_actions[v_begin_index: v_end_index]
                                val_batch_next_observations = next_observations[v_begin_index: v_end_index, :]
                                h_step_val_error = self.H_step_eval(val_batch_observations,
                                                                    val_batch_tech_actions, val_batch_next_observations, self.H)
                                if h_step_val_error != -1:
                                    mean_error += h_step_val_error
                                    count += 1
                            print('count:', count)
                            mean_error = mean_error / float(count)
                            print("tech: epoch: %d/%d, h_step_mean_error: " % (iter_index + 1, iter_num), mean_error)
                            summary = tf.Summary(value=[
                                tf.Summary.Value(tag="h_step_mean_error", simple_value=mean_error),
                            ])
                            self.summary_writer.add_summary(summary, overall_step)

                    self.summary_writer.add_summary(summary_str, overall_step)
                    self.tech_saver.save(self.sess, self.tech_model_path_save)
                overall_step += 1

        print('SL_train_tech_net end')

    def model_train_dis(self, observations, tech_actions, next_observations, verbose=False):
        observations = np.array(observations).astype(dtype=np.int32)
        tech_actions = np.array(tech_actions).astype(dtype=np.int32)
        next_observations = np.array(next_observations).astype(dtype=np.int32)
        if observations.shape[0] <= 0:
            return
        if self.save_data:
            print('observations:', observations.shape)
            print('tech_actions:', tech_actions.shape)
            print('next_observations:', next_observations.shape)
            tech_actions_mat = np.expand_dims(tech_actions, axis=-1)
            print('tech_actions_mat:', tech_actions_mat.shape)
            tech_record = np.concatenate([observations, tech_actions_mat, next_observations], axis=-1)
            print('tech_record:', tech_record.shape)
            cols = tech_record.shape[-1]
            print('cols:', cols)
            with open(self.save_data_path + "record.txt", 'ab') as f:
                np.savetxt(f, tech_record.astype(dtype=np.int32), fmt='%5.1i')

        observations = np.array(observations).astype(dtype=np.float32)
        tech_actions = np.array(tech_actions).astype(dtype=np.float32)
        next_observations = np.array(next_observations).astype(dtype=np.float32)
        self.SL_train_tech_net(observations, tech_actions, next_observations, batch_size=1000, iter_num=100, lr=1e-5, use_val=False)

    def H_step_eval(self, state_seq, action_seq, next_state_seq, H=5):
        state_now = self.inverse_transform_state(state_seq[0])
        true_state_now = state_now

        sum_error = 0.
        for i in range(H):
            predict_next = self.predict_tech(state_now, action_seq[i])
            state_next = next_state_seq[i]

            # check if all states are in one trj
            seconds_now = true_state_now[0]
            seconds_next = state_next[0]
            if seconds_next < seconds_now:
                return -1

            #print('state_now:', state_now)
            #print('action_now:', action_seq[i])
            #print('state_next:', state_next)
            #print('predict_next:', predict_next)
            # input()
            error = predict_next - state_next

            # make the two feature for progress to be precent, like 0.75
            #error[1] *= 0.1
            error[4] *= 0.01
            error[7] *= 0.01

            #print('error:', error)
            squer_error = sum(error ** 2)
            #print('squer_error:', squer_error)
            sum_error += squer_error
            # input()
            true_state_now = state_next
            state_now = predict_next

        mean_error = sum_error / float(H)
        return mean_error

    def restore_tech(self, model_path=""):
        if model_path == "":
            model_path = self.tech_model_path_load
        print('model_path', model_path)
        self.tech_saver.restore(self.sess, model_path)

    def restore_preprocessing(self, model_path=''):
        if model_path == '':
            path = self.tech_model_path_load + "_propossing.txt"
        else:
            path = model_path + "_propossing.txt"
        with open(path, 'r') as f:
            self.mean_and_std = np.loadtxt(f)

    def restore_sl_model(self, model_path):
        self.restore_tech(model_path)
        self.restore_preprocessing(model_path)

    def save_preprocessing(self, data):
        path = self.tech_model_path_save + "_propossing.txt"
        with open(path, 'w') as f:
            np.savetxt(f, data)

    def one_hot_label(self, action_type_array, action_max_num):
        rows = action_type_array.shape[0]
        cols = action_max_num
        data = np.zeros((rows, cols))

        for i in range(rows):
            data[i, int(action_type_array[i])] = 1

        return data
