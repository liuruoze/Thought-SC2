from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

USED_DEVICES = "0"
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

import lib.config as C
import param as P
import zerg_source_agent as source_agent
# from pysc2.env import sc2_env
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer

from source_network import SourceNetwork
from mini_network import MiniNetwork
from strategy.protoss_agent import Protoss
from strategy.terran_agent import Terran
from strategy.zerg_agent import Zerg
from strategy.agent import Dummy
from mapping_env import SimulatePlatform
import unit.protoss_unit as P
import unit.terran_unit as T
import unit.zerg_unit as Z

from datetime import datetime
import multiprocessing as mp
import numpy as np
from logging import warning as logging
from uct.numpy_impl import *

FLAGS = flags.FLAGS
flags.DEFINE_bool("training", True, "Whether to train agents.")
flags.DEFINE_bool("on_server", False, "Whether is running on server.")
flags.DEFINE_integer("num_for_update", 120, "Number of episodes for each train.")
flags.DEFINE_string("log_path", "./logs/", "Path for log.")
flags.DEFINE_string("device", USED_DEVICES, "Device for training.")

flags.DEFINE_string("map", "Simple64", "Name of a map to use.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
flags.DEFINE_integer("screen_resolution", 64, "Resolution for screen feature layers.")
flags.DEFINE_integer("minimap_resolution", 64, "Resolution for minimap feature layers.")

flags.DEFINE_enum("agent_race", "Z", sc2_env.races.keys(), "Agent's race.")
flags.DEFINE_enum("bot_race", "Z", sc2_env.races.keys(), "Bot's race.")
flags.DEFINE_enum("difficulty", "2", sc2_env.difficulties.keys(), "Bot's strength.")
flags.DEFINE_integer("max_agent_steps", 18000, "Total agent steps.")
flags.DEFINE_integer("step_mul", 4, "Game steps per agent step.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", False, "Whether to replays_save a replay at the end.")
flags.DEFINE_string("replay_dir", "multi-agent/", "dir of replay to replays_save.")

flags.DEFINE_string("restore_model_path", "./model/20190215-124147_mini/", "path for restore model")
flags.DEFINE_bool("restore_model", True, "Whether to restore old model")
flags.DEFINE_string("restore_mini_path", "./model/20190215-124147_mini/", "path for restore mini")

flags.DEFINE_integer("parallel", 12, "How many processes to run in parallel.")
flags.DEFINE_integer("thread_num", 5, "How many thread to run in the process.")
flags.DEFINE_integer("port_num", 4370, "the start port to create distribute tf")
flags.DEFINE_integer("max_iters", 300, "the rl agent max run iters")


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

if not FLAGS.on_server:
    PARALLEL = 1
    THREAD_NUM = 1
    MAX_AGENT_STEPS = 18000
    DEVICE = ['/gpu:0']
    NUM_FOR_UPDATE = 5
    TRAIN_ITERS = 5
    PORT_NUM = FLAGS.port_num
else:
    PARALLEL = FLAGS.parallel
    THREAD_NUM = FLAGS.thread_num
    MAX_AGENT_STEPS = FLAGS.max_agent_steps
    #DEVICE = ['/gpu:' + dev for dev in FLAGS.device.split(',')]
    DEVICE = ['/cpu:0']
    NUM_FOR_UPDATE = FLAGS.num_for_update
    TRAIN_ITERS = FLAGS.max_iters
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


def Init_brain_strategy_model(mini_net=None):
    strategy_agent = None
    if FLAGS.agent_race == 'P':
        strategy_agent = Protoss(net=mini_net, restore_model=True)
    elif FLAGS.agent_race == 'T':
        strategy_agent = Terran(net=mini_net, restore_model=True)
    elif FLAGS.agent_race == 'Z':
        strategy_agent = Zerg(net=mini_net, restore_model=True)

    blue_agent = None
    blue_agent = Dummy()
    if FLAGS.bot_race == 'T':
        blue_agent.get_power()

    mini_env = SimulatePlatform(red_agent=strategy_agent, blue_agent=blue_agent,
                                distance=5, max_steps=100)
    mini_env.init()
    strategy_agent.set_env(mini_env)
    strategy_agent.init_network()
    return strategy_agent


def Worker(index, update_game_num, Synchronizer, cluster, model_path):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    worker = tf.train.Server(cluster, job_name="worker", task_index=index, config=config)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(target=worker.target, config=config)

    #restore_mini_path = './model/20190103-151905_mini/'
    #restore_mini_path = './model/20190110-113720_mini/'
    mini_net = MiniNetwork(sess=sess, index=index, summary_writer=None, rl_training=False, cluster=cluster,
                           ppo_load_path=FLAGS.restore_mini_path, ppo_save_path=model_path)
    mini_net.initialize()
    Net = SourceNetwork(sess=sess, summary_writer=None, rl_training=FLAGS.training,
                        cluster=cluster, index=index, device=DEVICE[index % len(DEVICE)],
                        load_path=FLAGS.restore_model_path, save_path=model_path)

    global_buffer = Buffer()
    agents = []
    for i in range(THREAD_NUM):
        mini_agent = Init_brain_strategy_model(mini_net)
        agent = source_agent.SourceAgent(index=i, global_buffer=global_buffer, net=Net,
                                         restore_model=FLAGS.restore_model, rl_training=FLAGS.training,
                                         strategy_agent=mini_agent)
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


def Parameter_Server(Synchronizer, cluster, log_path, model_path, procs):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    sess = tf.Session(target=server.target, config=config)
    summary_writer = tf.summary.FileWriter(log_path)
    Net = SourceNetwork(sess=sess, summary_writer=summary_writer, rl_training=FLAGS.training,
                        cluster=cluster, index=0, device=DEVICE[0 % len(DEVICE)],
                        load_path=FLAGS.restore_model_path, save_path=model_path)

    agent = source_agent.SourceAgent(index=-1, net=Net, restore_model=FLAGS.restore_model, rl_training=FLAGS.training)

    print("Parameter server: waiting for cluster connection...")
    sess.run(tf.report_uninitialized_variables())
    print("Parameter server: cluster ready!")

    print("Parameter server: initializing variables...")
    agent.init_network()
    print("Parameter server: variables initialized")

    update_counter = 0
    while update_counter <= TRAIN_ITERS:
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

    for p in procs:
        print('Process terminate')
        p.terminate()


def _main(unused_argv):
    # create distribute tf cluster
    start_port = PORT_NUM
    SERVER_DICT["ps"].append("localhost:%d" % start_port)
    for i in range(PARALLEL):
        SERVER_DICT["worker"].append("localhost:%d" % (start_port + 1 + i))

    Cluster = tf.train.ClusterSpec(SERVER_DICT)

    now = datetime.now()
    model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_source/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_path = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "_source/"

    UPDATE_GAME_NUM = NUM_FOR_UPDATE
    per_update_num = np.ceil(UPDATE_GAME_NUM / PARALLEL)

    Synchronizer = mp.Barrier(PARALLEL + 1)
    # Run parallel process

    procs = []
    for index in range(PARALLEL):
        p = mp.Process(name="Worker_%d" % index, target=Worker, args=(index, per_update_num, Synchronizer, Cluster, model_path))
        procs.append(p)
        p.daemon = True
        p.start()
        time.sleep(1)

    Parameter_Server(Synchronizer, Cluster, log_path, model_path, procs)

    '''
    for p in procs:
        p.join()

    if FLAGS.profile:
        print(stopwatch.sw)
        '''


if __name__ == "__main__":
    app.run(_main)
