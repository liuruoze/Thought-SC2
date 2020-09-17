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

DATA_DIR = "record_img_dream"
SERIES_DIR = "series_dream"
model_path_name = "tf_models"

IN_CHANNELS = 12

# Hyperparameters for ConvVAE
z_size=64
file_size=10000
batch_size=300 # use 100 instead # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5    #0.5

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist):
  obs_list = []
  img_list = []
  action_list = []
  reward_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    l = raw_data['obs'].shape[0]

    if l < batch_size:
      continue

    #print('shape1:', raw_data['obs'][:batch_size].shape)
    if random.random() < 0.5:
      obs_list.append(raw_data['obs'][:batch_size])
      img_list.append(raw_data['img'][:batch_size])
      action_list.append(raw_data['action'][:batch_size])
      reward_list.append(raw_data['reward'][:batch_size])
    else:
      obs_list.append(raw_data['obs'][-batch_size:])
      img_list.append(raw_data['img'][-batch_size:])
      action_list.append(raw_data['action'][-batch_size:])
      reward_list.append(raw_data['reward'][-batch_size:])

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
