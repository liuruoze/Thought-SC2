# PPO param
gamma = 0.9995   # 0.9995 the same important as lamda
lamda = 0.9995   # important, change to 0.99 will make training failed
clip_value = 0.2
c_1 = 0.5  # 0.01
c_2 = 1e-3  # entropy 1e-3
batch_size = 256

# mini_network
mini_epoch_num = 10
mini_lr = 1e-4

# addition_network
mini_lr_add = 1e-4

# map info
use_small_map = True
map_batch_size = 256

# add weighted sum type
weight_type = 'AddWeight' # 'AddWeight', 'AdaptiveWeight', 'AttentionWeight'

# PPO value error
use_return_error = True

# adv norm and return norm
use_adv_norm = True
use_return_norm = False

# not used src_network
src_epoch_num = 10
src_lr = 1e-4

# not used
restore_model = False