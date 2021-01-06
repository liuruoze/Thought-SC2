# PPO param
gamma = 0.9995
lamda = 0.9995
clip_value = 0.2
c_1 = 0.01
c_2 = 1e-3  # entropy
lr = 1e-4
epoch_num = 10

# ppo batch size
batch_size = 256

# network
# update_num = 10

# run param
restore_model = False
restore_dynamic = False

# reward param
time_weight = 5
result_weight = 20
