import tensorflow as tf
import numpy as np
import lib.layer as layer
import lib.utils as U
import lib.config as C
import platform
from datetime import datetime
from sklearn import preprocessing


class MappingNetwork(object):

    def __init__(self, name, sess, load_path=None, save_path=None, rl_training=False,
                 sl_training=True, reuse=False):
        self.system = platform.system()

        now = datetime.now()
        model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_mapping/"
        if load_path is None:
            load_path = model_path
        if save_path is None:
            save_path = model_path

        self.func_model_path_load = load_path
        self.func_model_path_save = save_path
        self.sl_training = sl_training

        self.use_norm = False
        self.save_data = False
        self.predict_diff = True
        self.use_norm_input = False
        self.use_h_step_val = True
        self.use_norm_diff = False

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
        self.func_var_list_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope)
        self.func_saver = tf.train.Saver(var_list=self.func_var_list_save)

    def _create_graph(self):

        if self.reuse:
            tf.get_variable_scope().reuse_variables()
            assert tf.get_variable_scope().reuse

        with tf.name_scope("Input"):
            # func net:
            self.source_state = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_SOURCE_INPUT], name="source_state")
            self.mini_state = tf.placeholder(dtype=tf.float32, shape=[None, C._SIZE_MINI_INPUT], name="mini_state")

            self.func_label = tf.concat([self.mini_state], axis=1, name="func_label")
            self.func_lr_ph = tf.placeholder(dtype=tf.float32, shape=[], name="func_lr")

        with tf.name_scope("Network"):
            self.func_predict, self.net_scope = self._func_net(self.source_state)
            print("self.net_scope:", self.net_scope)

        if self.sl_training:
            with tf.name_scope("SL_loss"):
                self._define_sl_loss()

    def _define_sl_loss(self):
        # func loss
        self.func_var_list_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.net_scope)
        # print(self.func_var_list_train)

        with tf.name_scope("Tech_loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.func_predict, self.func_label))
            #self.regularizer = tf.nn.l2_loss(self.func_var_list_train)
            self.regularizer = tf.add_n([tf.nn.l2_loss(v) for v in self.func_var_list_train
                                         if 'bias' not in v.name])
            self.beta = 0.01
            self.func_loss = self.loss + self.beta * self.regularizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_scope)

            with tf.control_dependencies(update_ops):
                self.func_train_step = tf.train.AdamOptimizer(self.func_lr_ph).minimize(
                    self.func_loss, var_list=self.func_var_list_train)

            with tf.name_scope("summary_func"):
                self.func_loss_sum = tf.summary.scalar('func_loss', self.func_loss)
                self.summary_func_op = tf.summary.merge([self.func_loss_sum])

    def _func_net(self, data, trainable=True):
        with tf.variable_scope("Func_net"):
            # if self.use_norm == True:
            #    data = layer.batch_norm(data, self.is_training, 'BN')
            d1 = layer.dense_layer(data, 256, "DenseLayer1", is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            d2 = layer.dense_layer(d1, 128, "DenseLayer2", is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            dout = layer.dense_layer(d2, C._SIZE_MINI_INPUT, "DenseLayerOut", func=None, is_training=self.is_training, trainable=trainable, norm=self.use_norm)
            scope = tf.get_variable_scope().name
        return dout, scope

    def transform_state(self, state):
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = self.mean_and_std[0]
        scaler.scale_ = self.mean_and_std[1]
        return scaler.transform(state.reshape([1, -1])).reshape(-1)

    def inverse_transform_state(self, scaled_state):
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = self.mean_and_std[0]
        scaler.scale_ = self.mean_and_std[1]
        return scaler.inverse_transform(scaled_state.reshape([1, -1])).reshape(-1)

    def predict_func(self, source_state, use_transform=True):
        if self.use_norm_input and use_transform:
            input_state = self.transform_state(source_state)
        else:
            input_state = source_state

        #last_action_one_hot = self.one_hot_label(np.asarray([last_action]), C._SIZE_MAX_ACTIONS)
        feed_dict = {self.source_state: input_state.reshape([1, -1]),
                     self.is_training: self.sl_training,
                     }

        mini_state = self.func_predict.eval(feed_dict, session=self.sess).reshape(-1)

        return mini_state

    def SL_train_func_net(self, observations, next_observations, batch_size=1000, iter_num=100,
                          lr=1e-4, rate=0.7, use_val=True):
        print('SL_train_func_net begin')
        sample_num = observations.shape[0]
        train_num = int(sample_num * rate)
        print('train_num:', train_num)
        overall_step = 0

        if self.use_norm_input:
            scaler_obs = preprocessing.StandardScaler().fit(observations[0:train_num - 1, :])
            obs_mean, obs_std = scaler_obs.mean_, scaler_obs.scale_
            self.mean_and_std = np.array([obs_mean, obs_std])

            self.func_saver.save(self.sess, self.func_model_path_save)
            self.save_preprocessing(self.mean_and_std)
            observations = scaler_obs.transform(observations)

        for iter_index in range(iter_num):
            step = 0
            while step * batch_size <= train_num:
                begin_index, end_index = step * batch_size, min((step + 1) * batch_size, train_num - 1)
                random = np.random.random_sample(batch_size) * train_num
                random_indexs = np.array(random).astype(dtype=np.int32)

                batch_observations = observations[random_indexs, :]
                batch_next_observations = next_observations[random_indexs, :]
                feed_dict = {
                    self.source_state: batch_observations,
                    self.mini_state: batch_next_observations,
                    self.func_lr_ph: lr,
                    self.is_training: self.sl_training,
                }
                _, loss, regularizer, func_loss, predict, summary_str = self.sess.run(
                    [self.func_train_step, self.loss, self.regularizer,
                     self.func_loss, self.func_predict, self.summary_func_op], feed_dict=feed_dict)
                step += 1

                if overall_step % 50 == 0:
                    print("loss:", loss, "regularizer:", regularizer)
                    print("func: epoch: %d/%d, overall_step: %d, func_loss: " % (iter_index + 1, iter_num, overall_step), func_loss)
                    batch_int = np.array(batch_observations).astype(dtype=np.int32)
                    print(batch_int[0])
                    predict_int = np.array(predict).astype(dtype=np.int32)
                    print(predict_int[0])

                    # begin validateion
                    if use_val:
                        val_step = 0
                        val_mean_loss = []
                        while val_step * batch_size + train_num <= sample_num:
                            v_begin_index, v_end_index = val_step * batch_size + train_num, min((val_step + 1) * batch_size + train_num, sample_num - 1)

                            val_batch_observations = observations[v_begin_index: v_end_index, :]
                            val_batch_next_observations = next_observations[v_begin_index: v_end_index, :]
                            feed_dict = {
                                self.source_state: batch_observations,
                                self.mini_state: batch_next_observations,
                                self.func_lr_ph: lr,
                                self.is_training: False,
                            }
                            func_loss, predict = self.sess.run([self.func_loss, self.func_predict], feed_dict=feed_dict)
                            val_step += 1
                            val_mean_loss.append(func_loss)

                        val_loss = sum(val_mean_loss) / float(len(val_mean_loss))
                        print("func: epoch: %d/%d, val_mean_loss: " % (iter_index + 1, iter_num), val_loss)
                        summary = tf.Summary(value=[
                            tf.Summary.Value(tag="val_loss", simple_value=val_loss),
                        ])
                        self.summary_writer.add_summary(summary, overall_step)

                    self.summary_writer.add_summary(summary_str, overall_step)
                    self.func_saver.save(self.sess, self.func_model_path_save)
                overall_step += 1

        print('SL_train_func_net end')

    def model_train_dis(self, observations, next_observations, verbose=False):
        observations = np.array(observations).astype(dtype=np.int32)
        next_observations = np.array(next_observations).astype(dtype=np.int32)
        if observations.shape[0] <= 0:
            return
        if self.save_data:
            print('observations:', observations.shape)
            print('next_observations:', next_observations.shape)
            tech_record = np.concatenate([observations, next_observations], axis=-1)
            print('tech_record:', tech_record.shape)
            cols = tech_record.shape[-1]
            print('cols:', cols)
            with open(self.save_data_path + "record.txt", 'ab') as f:
                np.savetxt(f, tech_record.astype(dtype=np.int32), fmt='%5.1i')

        if self.sl_training:
            observations = np.array(observations).astype(dtype=np.float32)
            next_observations = np.array(next_observations).astype(dtype=np.float32)
            self.SL_train_func_net(observations, next_observations, batch_size=1000, iter_num=100, lr=1e-5, use_val=False)

    def restore_func(self, model_path=""):
        if model_path == "":
            model_path = self.func_model_path_load
        print('model_path', model_path)
        self.func_saver.restore(self.sess, model_path)

    def restore_preprocessing(self, model_path=''):
        if model_path == '':
            path = self.func_model_path_load + "_propossing.txt"
        else:
            path = model_path + "_propossing.txt"
        with open(path, 'r') as f:
            self.mean_and_std = np.loadtxt(f)

    def restore_sl_model(self, model_path):
        self.restore_func(model_path)
        self.restore_preprocessing()

    def save_preprocessing(self, data):
        path = self.func_model_path_save + "_propossing.txt"
        with open(path, 'w') as f:
            np.savetxt(f, data)

    def one_hot_label(self, action_type_array, action_max_num):
        rows = action_type_array.shape[0]
        cols = action_max_num
        data = np.zeros((rows, cols))

        for i in range(rows):
            data[i, int(action_type_array[i])] = 1

        return data
