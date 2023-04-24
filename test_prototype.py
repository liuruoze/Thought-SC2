from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

USED_DEVICES = "4,5"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES

import sys
import threading
import time

import tensorflow as tf
from absl import app
from absl import flags
from pysc2 import maps
from pysc2.lib import stopwatch

from lib import config as C
import param as P

# from pysc2.env import sc2_env
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer
from prototype.dynamic_network import DynamicNetwork
from prototype.hier_network import HierNetwork
from prototype.multi_agent import MultiAgent

from datetime import datetime
import multiprocessing as mp
import numpy as np
from logging import warning as logging
from uct.numpy_impl import *

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("on_server", True, "Whether is running on server.")
flags.DEFINE_bool("debug_mode", False, "Whether is debuging")

flags.DEFINE_integer("num_for_update", 100, "Number of episodes for each train.")
flags.DEFINE_string("log_path", "./logs/", "Path for log.")
flags.DEFINE_string("device", USED_DEVICES, "Device for training.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")

flags.DEFINE_enum("agent_race", "P", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "T", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "1", sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 10000, "Total agent steps.")
flags.DEFINE_integer("step_mul", 4, "Game steps per agent step.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to replays_save a replay at the end.")
flags.DEFINE_string("replay_dir", "multi-agent/", "dir of replay to replays_save.")

flags.DEFINE_string("restore_model_path", "./model/20181217-154646/", "path for restore model")
flags.DEFINE_bool("restore_model", True, "Whether to restore old model")
flags.DEFINE_string("restore_dynamic_path", "./model/20181223-174748_dynamic/", "path for restore dynamic")
flags.DEFINE_bool("restore_dynamic", True, "Whether to restore old dynamic model")

flags.DEFINE_integer("parallel", 10, "How many processes to run in parallel.")
flags.DEFINE_integer("thread_num", 5, "How many thread to run in the process.")
flags.DEFINE_integer("port_num", 3370, "the start port to create distribute tf")
flags.DEFINE_integer("max_iters", 100, "the rl agent max run iters")

flags.DEFINE_bool("use_MCTS", False, "Whether to use MCTS to choose actions.")
flags.DEFINE_integer("num_reads", 50, "Iteration numbers of MCTS to excuate.")

flags.DEFINE_bool("use_Dyna", True, "Whether to use Dyna to add more data.")
flags.DEFINE_integer("Dyna_steps_fisrt", 3, "The first simulated steps for dyna.")
flags.DEFINE_integer("Dyna_decrese_counter", 1, "Every this counter(overall training steps) dyna_steps decrease 1.")

# nohup python main.py > result.out &
# kill -9 `ps -ef |grep liuruoze | grep Main_Thread | awk '{print $2}' `
# kill -9 `ps -ef |grep liuruoze | grep main.py | awk '{print $2}' `
# kill -9 `ps -ef |grep lrz | grep main.py | awk '{print $2}' `
# ps -ef | grep 'SC2_x64' | awk '{print $2}' | xargs kill -9
# ps -ef |grep pangzhj | grep 'main.py'
# ps -ef | grep liuruoze | grep -v sshd
# export -n http_proxy
# export -n https_proxy

# kill -9 `ps -ef |grep liuruoze | grep test_prototype.py | awk '{print $2}' `
# kill -9 `ps -ef |grep lrz | grep main.py | awk '{print $2}' `
# fuser -v /dev/nvidia*

FLAGS(sys.argv)


if not FLAGS.on_server or FLAGS.debug_mode:
    PARALLEL = 1
    THREAD_NUM = 1
    MAX_AGENT_STEPS = 2000
    DEVICE = ['/gpu:0']
    NUM_FOR_UPDATE = 1
    TRAIN_ITERS = 1
    NUM_READS = 50
    PORT_NUM = FLAGS.port_num
else:
    PARALLEL = FLAGS.parallel
    THREAD_NUM = FLAGS.thread_num
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    if USED_DEVICES == '-1':
        DEVICE = ['/cpu:0']
    else:
        DEVICE = ['/gpu:' + str(dev) for dev in range(len(FLAGS.device.split(',')))] 
    NUM_FOR_UPDATE = FLAGS.num_for_update
    TRAIN_ITERS = FLAGS.max_iters
    NUM_READS = FLAGS.num_reads
    PORT_NUM = FLAGS.port_num


LOG = FLAGS.log_path
if not os.path.exists(LOG):
    os.makedirs(LOG)

SERVER_DICT = {"worker": [], "ps": []}

# define some global variable
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
Counter = 0
Waiting_Counter = 0
Update_Counter = 0
Result_List = []


def run_thread(agent, game_num, Synchronizer, difficulty):
    global UPDATE_EVENT, ROLLING_EVENT, Counter, Waiting_Counter, Update_Counter, Result_List

    num = 0
    proc_name = mp.current_process().name

    C._FPS = 22.4 / FLAGS.step_mul  # 5.6
    step_mul = FLAGS.step_mul  # 4
    C.difficulty = difficulty
    with sc2_env.SC2Env(
            map_name=FLAGS.map,
            agent_race=FLAGS.agent_race,
            bot_race=FLAGS.bot_race,
            difficulty=difficulty,
            step_mul=step_mul,
            score_index=-1,
            game_steps_per_episode=MAX_AGENT_STEPS,
            screen_size_px=(FLAGS.screen_resolution, FLAGS.screen_resolution),
            minimap_size_px=(FLAGS.minimap_resolution, FLAGS.minimap_resolution),
            visualize=False) as env:
        # env = available_actions_printer.AvailableActionsPrinter(env)
        agent.set_env(env)

        while True:
            agent.play()

            if FLAGS.training:
                # check if the num of episodes is enough to update
                num += 1
                Counter += 1
                reward = agent.result['reward']
                Result_List.append(reward)
                logging("(diff: %d) %d epoch: %s get %d/%d episodes! return: %d!" %
                        (int(difficulty), Update_Counter, proc_name, len(Result_List), game_num * THREAD_NUM, reward))

                # time for update
                if num == game_num:
                    num = 0
                    ROLLING_EVENT.clear()
                    # worker stops rolling, wait for update
                    if agent.index != 0 and THREAD_NUM > 1:
                        Waiting_Counter += 1
                        if Waiting_Counter == THREAD_NUM - 1:  # wait for all the workers stop
                            UPDATE_EVENT.set()
                        ROLLING_EVENT.wait()

                    # update!
                    else:
                        if THREAD_NUM > 1:
                            UPDATE_EVENT.wait()

                        Synchronizer.wait()  # wait for other processes to update

                        agent.update_network(Result_List)
                        Result_List.clear()
                        agent.global_buffer.reset()

                        Synchronizer.wait()

                        Update_Counter += 1

                        # finish update
                        UPDATE_EVENT.clear()
                        Waiting_Counter = 0
                        ROLLING_EVENT.set()

            if FLAGS.save_replay:
                env.save_replay(FLAGS.replay_dir)

            agent.reset()


def Worker(index, update_game_num, Synchronizer, cluster, log_path, model_path, dynamic_path):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    worker = tf.train.Server(cluster, job_name="worker", task_index=index, config=config)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(target=worker.target, config=config)
    Net = HierNetwork(sess=sess, summary_writer=None, rl_training=FLAGS.training,
                      cluster=cluster, index=index, device=DEVICE[index % len(DEVICE)], ppo_save_path=model_path, 
                      ppo_load_path=FLAGS.restore_model_path, dynamic_load_path=FLAGS.restore_dynamic_path)
    policy_in_mcts = PolicyNetinMCTS(Net)
    dynamic_net = Net.dynamic_net
    dynamic_net.restore_sl_model(FLAGS.restore_dynamic_path + "probe")

    global_buffer = Buffer()
    agents = []
    for i in range(THREAD_NUM):
        agent = MultiAgent(index=i, global_buffer=global_buffer, net=Net,
                                       restore_model=FLAGS.restore_model, rl_training=FLAGS.training,
                                       restore_internal_model=FLAGS.restore_dynamic,
                                       use_mcts=FLAGS.use_MCTS, num_reads=NUM_READS,
                                       policy_in_mcts=policy_in_mcts, dynamic_net=dynamic_net,
                                       use_dyna=FLAGS.use_Dyna, dyna_steps_fisrt=FLAGS.Dyna_steps_fisrt,
                                       dyna_decrese_counter=FLAGS.Dyna_decrese_counter)
        agents.append(agent)

    print("Worker %d: waiting for cluster connection..." % index)
    sess.run(tf.report_uninitialized_variables())
    print("Worker %d: cluster ready!" % index)

    while len(sess.run(tf.report_uninitialized_variables())):
        print("Worker %d: waiting for variable initialization..." % index)
        time.sleep(1)
    print("Worker %d: variables initialized" % index)

    game_num = np.ceil(update_game_num // THREAD_NUM)

    UPDATE_EVENT.clear()
    ROLLING_EVENT.set()

    # Run threads
    threads = []
    for i in range(THREAD_NUM - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], game_num, Synchronizer, FLAGS.difficulty))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(3)

    run_thread(agents[-1], game_num, Synchronizer, FLAGS.difficulty)

    for t in threads:
        t.join()


def Parameter_Server(Synchronizer, cluster, log_path, model_path, dynamic_path, procs):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(target=server.target, config=config)
    summary_writer = tf.summary.create_file_writer(log_path)
    Net = HierNetwork(sess=sess, summary_writer=summary_writer, rl_training=FLAGS.training,
                      cluster=cluster, index=0, device=DEVICE[0 % len(DEVICE)],
                      ppo_load_path=FLAGS.restore_model_path, ppo_save_path=model_path, dynamic_load_path=FLAGS.restore_dynamic_path)
    policy_in_mcts = PolicyNetinMCTS(Net)
    dynamic_net = Net.dynamic_net
    dynamic_net.restore_sl_model(FLAGS.restore_dynamic_path + "probe")

    agent = MultiAgent(index=-1, net=Net, restore_model=FLAGS.restore_model, rl_training=FLAGS.training,
                                   restore_internal_model=FLAGS.restore_dynamic,     
                                   use_mcts=FLAGS.use_MCTS, num_reads=NUM_READS,
                                   policy_in_mcts=policy_in_mcts, dynamic_net=dynamic_net,
                                   use_dyna=FLAGS.use_Dyna, dyna_steps_fisrt=FLAGS.Dyna_steps_fisrt,
                                   dyna_decrese_counter=FLAGS.Dyna_decrese_counter)

    print("Parameter server: waiting for cluster connection...")
    sess.run(tf.report_uninitialized_variables())
    print("Parameter server: cluster ready!")

    print("Parameter server: initializing variables...")
    agent.init_network()
    print("Parameter server: variables initialized")

    update_counter = 0
    while True:
        agent.reset_old_network()

        # wait for update
        Synchronizer.wait()
        logging("Update Network!")
        # TODO count the time , compare cpu and gpu
        time.sleep(1)

        # update finish
        Synchronizer.wait()
        logging("Update Network finished!")

        agent.update_summary(update_counter)
        update_counter += 1
        agent.save_model()


def _main(unused_argv):
    """Run agents"""
    maps.get(FLAGS.map)  # Assert the map exists.

    # create distribute tf cluster
    start_port = FLAGS.port_num
    SERVER_DICT["ps"].append("localhost:%d" % start_port)
    for i in range(PARALLEL):
        SERVER_DICT["worker"].append("localhost:%d" % (start_port + 1 + i))

    Cluster = tf.train.ClusterSpec(SERVER_DICT)

    global LOG
    if FLAGS.training:
        now = datetime.now()
        LOG = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

        model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        dynamic_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_dynamic/"
        if not os.path.exists(dynamic_path):
            os.makedirs(dynamic_path)

    UPDATE_GAME_NUM = NUM_FOR_UPDATE
    per_update_num = np.ceil(UPDATE_GAME_NUM / PARALLEL)

    Synchronizer = mp.Barrier(PARALLEL + 1)
    # Run parallel process

    procs = []
    for index in range(PARALLEL):
        p = mp.Process(name="Worker_%d" % index, target=Worker,args=(index, per_update_num, Synchronizer, Cluster, 
            LOG, model_path, dynamic_path))
        procs.append(p)
        p.daemon = True
        p.start()
        time.sleep(1)

    Parameter_Server(Synchronizer, Cluster, LOG, model_path, dynamic_path, procs)

    for p in procs:
        p.join()

    if FLAGS.profile:
        print(stopwatch.sw)


if __name__ == "__main__":
    app.run(_main)
