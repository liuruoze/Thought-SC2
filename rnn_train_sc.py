'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

from vae.vae import ConvVAE, reset_graph
from rnn.rnn import HyperParams, MDNRNN

os.environ["CUDA_VISIBLE_DEVICES"]="0"
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

F_MODE_I_N = 0    # combine non-image feature and image feature
F_MODE_N = 1     # only non-image feature obs
F_MODE_I = 2     # only image obs

ACTION_SPACE = 10
SIZE_1 = 32                # image latent size
SIZE_2 = 20                # non-image obs feature size

THE_MODE = F_MODE_I_N

DATA_DIR = "series"
model_save_path = "tf_rnn"

if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
initial_z_save_path = "tf_initial_z"
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def random_batch(mode=THE_MODE):
  indices = np.random.permutation(N_data)[0:batch_size]
  nf = data_non_image_feature[indices]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  s = logvar.shape
  img = mu + np.exp(logvar/2.0) * np.random.randn(*s)

  assert img.shape[:-1] == nf.shape[:-1]

  if mode == F_MODE_I_N:
    z = np.concatenate([img, nf], axis=-1)
  elif mode == F_MODE_N:
    z = nf
  elif mode == F_MODE_I:
    z = img
  else:
    z = np.concatenate([img, nf], axis=-1)

  return z, action

def default_hps():
  return HyperParams(num_steps=50000,
                     max_seq_len=299, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=SIZE_1+SIZE_2+ACTION_SPACE,    # width of our data (32 + 20 + 10 actions)
                     output_seq_width=SIZE_1+SIZE_2,    # width of our data is 32 + 20
                     rnn_size=256,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
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
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series_10000.npz"))


# load preprocessed data and change data type
data_non_image_feature = raw_data["obs"]
#print(type(data_non_image_feature[0]))
data_mu = raw_data["mu"]
#print(data_mu[:1000, 0, :])
data_logvar = raw_data["logvar"]
#print(data_logvar[0].shape)
#print(raw_data["action"][15, 24])
data_action =  get_one_hot(raw_data["action"], ACTION_SPACE)

max_seq_len = hps_model.max_seq_len

N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

# save 1000 initial mu and logvars:
#
'''
initial_non_image_feature = np.copy(data_non_image_feature[:1000, 0, :]*10000).astype(np.int).tolist()
initial_mu = np.copy(data_mu[:1000, 0, :]*10000).astype(np.int).tolist()
initial_logvar = np.copy(data_logvar[:1000, 0, :]*10000).astype(np.int).tolist()
with open(os.path.join("tf_initial_z", "initial_z.json"), 'wt') as outfile:
  json.dump([initial_non_image_feature, initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))
'''


reset_graph()
rnn = MDNRNN(hps_model)

# train loop:
hps = hps_model
start = time.time()
for local_step in range(hps.num_steps):

  step = rnn.sess.run(rnn.global_step)
  curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate

  raw_z, raw_a = random_batch()
  inputs = np.concatenate((raw_z[:, :-1, :], raw_a[:, :-1, :]), axis=2)
  outputs = raw_z[:, 1:, :] # teacher forcing (shift by one predictions)

  feed = {rnn.input_x: inputs, rnn.output_x: outputs, rnn.lr: curr_learning_rate}
  (train_cost, state, train_step, _) = rnn.sess.run([rnn.cost, rnn.final_state, rnn.global_step, rnn.train_op], feed)
  if (step%20==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    output_log = "step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, train_cost, time_taken)
    print(output_log)

# save the model (don't bother with tf checkpoints json all the way ...)
rnn.save_json(os.path.join(model_save_path, "rnn.json"))
