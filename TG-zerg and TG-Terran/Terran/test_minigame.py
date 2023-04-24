from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

USED_DEVICES = "4"
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = USED_DEVICES

import sys
import threading
import time

import tensorflow as tf
import multiprocessing as mp
import numpy as np

from logging import warning as logging
from datetime import datetime
from mini_network import MiniNetwork
from strategy.protoss_agent import Protoss
from strategy.terran_agent import Terran
from strategy.terran_agent import Terran, DummyTerran
# from strategy.agent import Dummy
from unit.units import Army
from lib.replay_buffer import Buffer
from mapping_env import SimulatePlatform
from absl import app
from absl import flags

import unit.protoss_unit as P
import unit.terran_unit as T

FLAGS = flags.FLAGS
flags.DEFINE_string("restore_model_path", "./model/20190221-134411_mini/", "path for restore model")
flags.DEFINE_bool("restore_model", False, "Whether to restore old model")
FLAGS(sys.argv)

# define some global variable
UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
Counter = 0
Waiting_Counter = 0
Update_Counter = 0
Result_List = []

NUM_FOR_UPDATE = 500
PARALLEL = 1
THREAD_NUM = 1
if True:
    PARALLEL = 10
    THREAD_NUM = 1
PORT_NUM = 36360
TRAIN_ITERS = 220
SERVER_DICT = {"worker": [], "ps": []}

DIFF = 1

def eval(agent, game_num, Synchronizer):
    blue_agent = DummyTerran()
    blue_agent.get_power()
    env = SimulatePlatform(red_agent=agent, blue_agent=blue_agent,
                           distance=5, max_steps=150)
    env.init()
    agent.set_env(env)
    val_results = []

    for _ in range(game_num):
        agent.play_with_rl()
        val_results.append(agent.result)
        agent.reset()

    win_rate = 0.0
    if len(val_results) > 0:
        win_rate = val_results.count(1) / float(len(val_results))

    return win_rate


def run_thread(agent, game_num, Synchronizer, difficulty):
    global UPDATE_EVENT, ROLLING_EVENT, Counter, Waiting_Counter, Update_Counter, Result_List

    num = 0
    proc_name = mp.current_process().name

    blue_agent = DummyTerran(diff=difficulty)
    blue_agent.get_power()

    env = SimulatePlatform(red_agent=agent, blue_agent=blue_agent,
                           distance=5, max_steps=100)
    env.init()
    agent.set_env(env)

    while True:
        # agent.play_with_rl(verbos=True)'
        env.simulate(False)

        if True:
            # check if the num of episodes is enough to update
            num += 1
            Counter += 1
            reward = agent.result
            Result_List.append(reward)
            logging("(diff: %d) %d epoch: %s get %d/%d episodes! return: %f!" %
                    (int(difficulty), Update_Counter, proc_name, len(Result_List), game_num * THREAD_NUM, reward))

            # time for update
            if num == game_num:
                num = 0
                ROLLING_EVENT.clear()
                # worker stops rolling, wait for update
                if agent.agent_id != 0 and THREAD_NUM > 1:
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

        win_rate = agent.net.get_win_rate()
        if win_rate > 0.90:
            difficulty += 1
            env.blue_agent.set_diff(difficulty)
            print('Increase difficulty to:', difficulty)

        env.reset()


def Worker(index, update_game_num, Synchronizer, cluster, model_path):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    worker = tf.train.Server(cluster, job_name="worker", task_index=index, config=config)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(target=worker.target, config=config)

    mini_net = MiniNetwork(sess, index=index, summary_writer=None, rl_training=True, cluster=cluster,
                           ppo_load_path=FLAGS.restore_model_path, ppo_save_path=model_path)
    global_buffer = Buffer()
    agents = []
    for i in range(THREAD_NUM):
        agent = Terran(agent_id=i, global_buffer=global_buffer, net=mini_net, restore_model=FLAGS.restore_model)
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
    difficulty = DIFF

    # Run threads
    threads = []
    for i in range(THREAD_NUM - 1):
        t = threading.Thread(target=run_thread, args=(agents[i], game_num, Synchronizer, difficulty))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(3)

    run_thread(agents[-1], game_num, Synchronizer, difficulty)

    for t in threads:
        t.join()


def Parameter_Server(Synchronizer, cluster, log_path, model_path, procs):
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(target=server.target, config=config)
    summary_writer = tf.summary.create_file_writer(log_path)
    mini_net = MiniNetwork(sess, index=0, summary_writer=summary_writer, rl_training=True, cluster=cluster,
                           ppo_load_path=FLAGS.restore_model_path, ppo_save_path=model_path)
    agent = Terran(agent_id=-1, global_buffer=Buffer(), net=mini_net, restore_model=FLAGS.restore_model)

    print("Parameter server: waiting for cluster connection...")
    sess.run(tf.report_uninitialized_variables())
    print("Parameter server: cluster ready!")

    print("Parameter server: initializing variables...")
    agent.init_network()
    print("Parameter server: variables initialized")

    update_counter = 0
    # max_win_rate = 0.
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

        # agent.update_summary(update_counter)
        # update_counter += 1
        # agent.save_model()
        steps, win_rate = agent.update_summary(update_counter)
        # logging("Steps: %d, win rate: %f" % (steps, win_rate))

        update_counter += 1

        agent.save_model()
            # max_win_rate = win_rate


        # if update_counter % 5 == 0:
        #     print('iter:', update_counter, ', begin to eval')
            #win_rate = eval(agent, NUM_FOR_UPDATE, Synchronizer)
            #print('win rate:', win_rate)
            # if win_rate >= 0.95:
            #    break

    for p in procs:
        print('Process terminate')
        p.terminate()


if __name__ == "__main__":
    # create distribute tf cluster
    start_port = PORT_NUM
    SERVER_DICT["ps"].append("localhost:%d" % start_port)
    for i in range(PARALLEL):
        SERVER_DICT["worker"].append("localhost:%d" % (start_port + 1 + i))

    Cluster = tf.train.ClusterSpec(SERVER_DICT)

    now = datetime.now()
    model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_mini/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    LOG = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "_mini/"

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

    Parameter_Server(Synchronizer, Cluster, LOG, model_path, procs)

    # for p in procs:
    #    print('Process join')
    #    p.join()
