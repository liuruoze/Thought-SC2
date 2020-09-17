'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import gc
from rnn.rnn_dream import reset_graph, ConvVAE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

ITERATION_INDEX = "2"

DATA_DIR = "record_img_dream" + "_" + ITERATION_INDEX
SERIES_DIR = "series_dream" + "_" + ITERATION_INDEX
model_path_name = "tf_models" + "_" + ITERATION_INDEX

IN_CHANNELS = 12

# Hyperparameters for ConvVAE
z_size=64
file_size=3000
batch_size=300 # use 100 instead # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5    #0.5

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def append_dim(data, l, batch_size):
  l = data.shape[0]
  print('l:', l)
  print('shape', data.shape[1:])
  new_shape = (batch_size,) + data.shape[1:]
  print('new shape:', new_shape)

  new_data = np.zeros(new_shape)
  print('new_data.shape', new_data.shape)
  new_data[:l,] = data
  return new_data

def load_raw_data_list(filelist):
  obs_list = []
  img_list = []
  action_list = []
  reward_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i] 
    try:
      raw_data = np.load(os.path.join(DATA_DIR, filename))
    except:
      continue 
    #print(filename)

    if filename == '1594098046.npz' or len(filename.split('.')) > 2:
      print('Match!')
      
    else:
      print(filename)
      l = raw_data['obs'].shape[0]

      if l < batch_size:
        l = raw_data['obs'].shape[0]
        new_obs = append_dim(raw_data['obs'], l, batch_size)
        new_img = append_dim(raw_data['img'], l, batch_size)
        new_action = append_dim(raw_data['action'], l, batch_size)
        new_reward = append_dim(raw_data['reward'], l, batch_size)

        print('shape:', raw_data['obs'][:batch_size].shape)
        if random.random() < 0.5:
          obs_list.append(new_obs[:batch_size])
          img_list.append(new_img[:batch_size])
          action_list.append(new_action[:batch_size])
          reward_list.append(new_reward[:batch_size])
        else:
          obs_list.append(new_obs[-batch_size:])
          img_list.append(new_img[-batch_size:])
          action_list.append(new_action[-batch_size:])
          reward_list.append(new_reward[-batch_size:])

        #print('raw_data[action].shape:', raw_data['action'].shape)
        if ((i+1) % 1000 == 0):
          print("loading file", (i+1))
          print("collect carbige", gc.collect())
          gc.collect()
  return obs_list, img_list, action_list, reward_list

def encode_batch(batch_img):
  simple_obs = np.copy(batch_img).astype(np.float)
  simple_obs = simple_obs.reshape(-1, 64, 64, IN_CHANNELS)

  #print('simple_obs.shape:', simple_obs.shape)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))
  return mu, logvar, z

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size))
  #batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, IN_CHANNELS)
  return batch_img

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:file_size]

obs_dataset, img_list, action_dataset, reward_dataset = load_raw_data_list(filelist)
gc.collect()

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=True) # use GPU on batchsize of 1000 -> much faster

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
for i in range(len(img_list)):
  data_batch = img_list[i]
  mu, logvar, z = encode_batch(data_batch)
  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
obs_dataset = np.array(obs_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)
reward_dataset = np.array(reward_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, obs=obs_dataset, 
  mu=mu_dataset, logvar=logvar_dataset, reward=reward_dataset)
