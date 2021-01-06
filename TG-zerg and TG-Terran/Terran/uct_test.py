from uct.numpy_impl import *
import tensorflow as tf
from dynamic_network2 import DynamicNetwork
from hier_network import HierNetwork


def test(is_restore=True):
    # train model
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # model_net = ModelNetwork('train', sess)
    # model_net.initialize()
    # if is_restore:
    #    model_net.restore_tech()

    C._LOAD_MODEL_PATH = './model/20181212-151843/'
    hier_net = HierNetwork(sess)
    hier_net.initialize()
    if is_restore:
        hier_net.restore_policy()
    policy_net = PolicyNetinMCTS(hier_net)

    dynamic_model_path = './model/20181218-183513_dynamic/probe'
    if 1:
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
