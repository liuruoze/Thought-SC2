'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from rnn.rnn_dream import reset_graph, ConvVAE

# Hyperparameters for ConvVAE
z_size=64
batch_size=100
max_file_size=10000      #default is 10000
learning_rate=0.0001
kl_tolerance=0.5    #0.5

# Parameters for training
NUM_EPOCH = 3         #default is 10
DATA_DIR = "record_img_dream"

IN_CHANNELS = 12   # 3

model_save_path = "tf_models"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def count_length_of_filelist(filelist):
  # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
  N = len(filelist)
  total_length = 0
  load_size = min(max_file_size, N)
  for i in range(load_size):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))['img']
    l = len(raw_data)
    total_length += l
    if (i % 1000 == 0):
      print("loading file", i)
  return  total_length

def create_dataset(filelist, N=max_file_size, M=1000): # N is max_file_size episodes, M is number of timesteps
  M=300
  data = np.zeros((M*N, 64, 64, IN_CHANNELS), dtype=np.float16)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))['img']
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    #input("Press Enter to continue...")

    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print("loading file", i+1)
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:max_file_size]
#print("check total number of images:", count_length_of_filelist(filelist))
dataset = create_dataset(filelist)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=True,
              reuse=False,
              gpu_mode=True)

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)

    feed = {vae.x: obs,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json(model_save_path+"/vae.json")

# finished, final model:
vae.save_json(model_save_path+"/vae.json")
