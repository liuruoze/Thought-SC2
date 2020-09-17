USED_DEVICES = "6,7"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES

from uct.numpy_impl import *
import tensorflow as tf
from prototype.dynamic_network import DynamicNetwork
from prototype.hier_network import HierNetwork


def test(is_restore_policy=True, is_restore_dynamic=True):
    # train model
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    hier_net = HierNetwork(sess, policy_path='./model/20181217-154646/probe')
    hier_net.initialize()
    if is_restore_policy:
        hier_net.restore_policy()

    policy_net = PolicyNetinMCTS(hier_net)
    dynamic_model_path = './model/20181223-174748_dynamic/probe'
    if is_restore_dynamic:
        hier_net.restore_dynamic(dynamic_model_path)
    dynamic_net = hier_net.dynamic_net

    num_reads = 100
    import time
    tick = time.time()
    print(UCT_search(GameState(dynamic_net), num_reads, policy_net))
    tock = time.time()
    print("Took %s sec to run %s times" % (tock - tick, num_reads))
    #import resource
    #print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    test()
