import os
from prototpye.dynamic_network import DynamicNetwork
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

''' Normalization in TensorFlow '''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import scipy.ndimage as ndimage
import lib.config as C
from sklearn import preprocessing


def cut_file(record_path):
    f = open(record_path, "r")
    lines = f.readlines()
    last_line = lines[-1]
    print(last_line)
    f.close()

    f = open(record_path + ".change", "w")
    for line in lines:
        if line != last_line:
            f.write(line)
    f.close()


def train(record_path, is_restore=False, is_norm=False):
    # train model
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model_path = "./model/" + "20190218-182616" + "_dynamic/"
    dynamic_net = DynamicNetwork(name='train', sess=sess, load_path=model_path)
    dynamic_net.initialize()

    if is_restore:
        dynamic_net.restore_tech()

    model_data = np.loadtxt(record_path)

    sample_num = model_data.shape[0]

    print('sample_num:', sample_num)
    print('sample_data:', model_data[0])
    print('sample_data:', model_data[-1])

    observations = model_data[:, :C._SIZE_SIMPLE_INPUT]
    tech_actions = model_data[:, C._SIZE_SIMPLE_INPUT:C._SIZE_SIMPLE_INPUT + 1]
    next_observations = model_data[:, C._SIZE_SIMPLE_INPUT + 1:]

    tech_actions = np.squeeze(tech_actions, axis=1)
    dynamic_net.SL_train_tech_net(observations, tech_actions, next_observations, batch_size=5000, iter_num=10000, lr=1e-3)


if __name__ == "__main__":
    # simple_test()
    record_path = "../shared_data/record.txt"
    # cut_file(record_path)
    train(record_path + ".change", False)
