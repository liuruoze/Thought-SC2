from enum import Enum

import numpy as np
import time
from lib.replay_buffer import Buffer

class MiniAgent():

    def __init__(self, agent_id=0, global_buffer=None, net=None, restore_model=False):
        self.agent_id = agent_id
        self.net = net
        self.global_buffer = global_buffer
        self.greedy_action = False
        self.local_buffer = Buffer()
        self.env = None
        self.restore_model = restore_model

        self.reset()

    def __str__(self):
        return None

    def set_env(self, env):
        self.env = env

    def reset(self):
        self.step = 0
        self.obs = None
        self.reward = 0
        self.done = False
        self.result = 0
        self.local_buffer.reset()

    def play(self, show_details=False):
        #self.reset()
        self.obs = self.env.reset()
        state_last = None

        while True:
            # get the action
            if self.greedy_action:
                action_prob, v_preds = self.net.policy.get_action_probs(self.obs, verbose=False)
                action = np.argmax(action_prob)
            else:
                action, v_preds = self.net.policy.get_action(self.obs, verbose=False)

            # use the action to push the env step
            self.obs, self.reward, self.done, info = self.env.step(action)

            # add info to buffer
            if state_last is not None:
                if show_details:
                    print('state_last:', state_last, ', action_last:', action_last, ', state_now:', self.obs)
                v_preds_next = self.net.policy.get_values(self.obs)
                v_preds_next = self.get_values(v_preds_next)
                self.local_buffer.append(state_last, action_last, self.obs, self.reward, v_preds, v_preds_next)
            
            state_last = self.obs
            action_last = action

            if self.done:
                self.result = self.reward
                print('play end, total return', self.result) if show_details else None
                if len(self.local_buffer.rewards) > 0:
                    self.global_buffer.add(self.local_buffer)
                print("add %d buffer!" % (len(self.local_buffer.rewards))) if 1 else None
                break



    def init_network(self):
        self.net.initialize()
        if self.restore_model:
            self.net.restore_policy()

    def update_network(self, result_list):
        self.net.Update_policy(self.global_buffer)
        self.net.Update_result(result_list)

    def reset_old_network(self):
        self.net.reset_old_network()

    def save_model(self):
        self.net.save_policy()

    def update_summary(self, counter):
        return self.net.Update_summary(counter)

    def get_values(self, values):
        # check if the game is end
        if self.done:
            return 0
        else:
            return values

    