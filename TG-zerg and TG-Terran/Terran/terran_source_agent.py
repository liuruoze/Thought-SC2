# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from logging import warning as logging

import contrib.utils as U
import param as P
from lib import config as C
from lib import transform_pos as T
from contrib import option as M
from lib import environment
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer
from strategy.terran_agent import TerranAction


class SourceAgent(base_agent.BaseAgent):
    """Agent for source game of starcraft."""

    def __init__(self, index=0, rl_training=False, restore_model=False, global_buffer=None, net=None,
                 strategy_agent=None):
        super(SourceAgent, self).__init__()
        self.net = net
        self.index = index
        self.global_buffer = global_buffer
        self.restore_model = restore_model

        # model in brain
        self.strategy_agent = strategy_agent
        self.strategy_act = None

        # count num
        self.step = 0

        self.strategy_wait_secs = 4
        self.strategy_flag = False
        self.policy_wait_secs = 2
        self.policy_flag = True

        self.env = None
        self.obs = None

        # buffer
        self.local_buffer = Buffer()
        self.mini_state = []
        self.mini_state_mapping = []

        self.num_players = 2
        self.on_select = None
        self._result = None
        self.is_end = False
        self.is_attack = False
        self._gases = None

        self.rl_training = rl_training

        self.reward_type = 0

    def reset(self):
        super(SourceAgent, self).reset()
        self.step = 0
        self.obs = None
        self._result = None
        self.is_end = False
        self.is_attack = False
        self._gases = None

        self.policy_flag = True

        self.local_buffer.reset()

        if self.strategy_agent is not None:
            self.strategy_agent.reset()

    def set_env(self, env):
        self.env = env

    def init_network(self):
        self.net.initialize()
        if self.restore_model:
            self.net.restore_policy()

    def reset_old_network(self):
        self.net.reset_old_network()

    def save_model(self):
        self.net.save_policy()

    def update_network(self, result_list):
        self.net.Update_policy(self.global_buffer)
        self.net.Update_result(result_list)

    def update_summary(self, counter):
        return self.net.Update_summary(counter)

    def get_policy_input(self, obs):
        high_input, tech_cost, pop_num = U.get_input(obs)
        policy_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return policy_input


    def mini_step(self, action):

        if action == TerranAction.Build_SCV.value:
            M.mineral_worker(self)

        elif action == TerranAction.Build_Refinery.value:
            if self._gases is not None:
                # U.find_gas_pos(self.obs, 1)
                gas_1 = self._gases[0]
                gas_2 = self._gases[1]

                if gas_1 is not None and not U.is_Refinery_on_gas(self.obs, gas_1):
                    gas_1_pos = T.world_to_screen_pos(self.env.game_info, gas_1.pos, self.obs)
                    M.build_by_idle_worker(self, C._BUILD_REFINERY_S, gas_1_pos)

                elif gas_2 is not None and not U.is_Refinery_on_gas(self.obs, gas_2):
                    gas_2_pos = T.world_to_screen_pos(self.env.game_info, gas_2.pos, self.obs)
                    M.build_by_idle_worker(self, C._BUILD_REFINERY_S, gas_2_pos)

        elif action == TerranAction.Gather_gas.value:
            M.gather_resource(self, 'gas')

        elif action == TerranAction.Gather_mineral.value:
            M.gather_resource(self, 'mineral')

        elif action == TerranAction.Build_Marine.value:
            M.train_army(self, C._TRAIN_MARINE)

        elif action == TerranAction.Build_Reaper.value:
            M.train_army(self, C._TRAIN_REAPER)

        elif action == TerranAction.Build_SupplyDepot.value:
            valid_index = U.get_valid_mask_screen(self.obs, size=3)
            pos = U.get_pos(valid_index)
            M.build_by_idle_worker(self, C._BUILD_SUPPLYDEPOT_S, pos)

        elif action == TerranAction.Build_Barracks.value:
            valid_index = U.get_valid_mask_screen(self.obs, size=3)
            pos = U.get_pos(valid_index)
            M.build_by_idle_worker(self, C._BUILD_BARRACKS_S, pos)

        elif action == TerranAction.Attack.value:
            self.is_attack = True

        elif action == TerranAction.Defend.value:
            M.retreat_step(self)

        else:
            self.safe_action(C._NO_OP, 0, [])

        if self.is_attack:
            M.attack_step(self)
        # if any queen exists, try to inject lavra to hatchery.
        # M.inject_larva(self) TODO

    def get_the_input(self):
        high_input, tech_cost, pop_num = U.get_input(self.obs)
        controller_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return controller_input

    def combine_state_and_mini_action(self, state, strategy_act):
        act = np.zeros((1, 1))
        act[0, 0] = strategy_act
        action_array = self.one_hot_label(act, C._SIZE_MINI_ACTIONS)[0]
        combined_state = np.concatenate([state, action_array], axis=0)
        return combined_state

    def mapping_source_to_mini(self, source_state):
        mini_state = self.net.mapping.predict_func(source_state, use_transform=False)
        return mini_state

    def mapping_source_to_mini_by_rule(self, source_state):
        simple_input = np.zeros([17])
        simple_input[0] = 0  # self.time_seconds
        simple_input[1] = source_state[27]  # self.mineral_worker_nums
        simple_input[2] = source_state[29] + source_state[31]  # self.gas_worker_nums
        simple_input[3] = source_state[2]  # self.mineral
        simple_input[4] = source_state[3]  # self.gas
        simple_input[5] = source_state[6]  # self.food_cap
        simple_input[6] = source_state[7]  # self.food_used
        simple_input[7] = source_state[10]  # self.army_nums

        simple_input[8] = source_state[11]  # self.Refinery_num
        simple_input[9] = source_state[12]  # self.SupplyDepot_num
        simple_input[10] = source_state[13]  # self.Barracks_num

        simple_input[11] = source_state[15]  # self.Marine_num
        simple_input[12] = source_state[16]  # self.Reaper_num

        simple_input[13] = source_state[4] + source_state[2]
        simple_input[14] = source_state[5] + source_state[3]

        simple_input[15] = source_state[4]
        simple_input[16] = source_state[5]

        return simple_input

    def play(self, verbose=False):
        is_attack = False
        state_last = None

        self.safe_action(C._NO_OP, 0, [])
        self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
        self._gases = U.find_initial_gases(self.obs)

        while True:
            # self.mini_step(3)
            self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
            #self.safe_action(C._NO_OP, 0, [])
            if self.policy_flag and (not self.is_end):

                state_now = self.mapping_source_to_mini_by_rule(self.get_the_input())

                # action_prob, v_preds = self.net.policy.get_action_probs(state_now, verbose=False)
                # action = np.argmax(action_prob)
                # print(action)

                action, v_preds = self.net.policy.get_action(state_now, verbose=False)
                # print(action)

                self.mini_step(action)
                if state_last is not None:
                    # print(state_now)
                    # print(TerranAction(int(action)).name)
                    # time.sleep(0.5)
                    v_preds_next = self.net.policy.get_values(state_now)
                    v_preds_next = self.get_values(v_preds_next)
                    reward = 0
                    self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)


                state_last = state_now
                action_last = action
                self.policy_flag = False

            if self.is_end:
                if self.rl_training:
                    self.local_buffer.rewards[-1] += 1 * self.result['reward']  # self.result['win']
                    # print(self.local_buffer.rewards)
                    self.global_buffer.add(self.local_buffer)
                    # print("add %d buffer!" % (len(self.local_buffer.rewards)))
                break

    # def play(self, verbose=False):
    #     is_attack = False
    #     while True:
    #         #self.safe_action(C._NO_OP, 0, [])
    #         self.safe_action(C._MOVE_CAMERA, 0, [C.base_camera_pos])
    #         if self.policy_flag and (not self.is_end):
    #             mini_state_mapping = self.mapping_source_to_mini_by_rule(self.get_the_input())
    #             #print('state:', mini_state_mapping)
    #             mini_act = self.strategy_agent.get_action_by_policy(mini_state_mapping)[0]
    #             print("Action: ", ZergAction(int(mini_act)).name)
    #             self.mini_step(mini_act)

    #             if mini_act == ZergAction.Attack.value:
    #                 is_attack = True
    #             if is_attack:
    #                 self.mini_step(ZergAction.Attack.value)

    #             self.policy_flag = False

    #         if self.is_end:
    #             break

    def set_flag(self):
        if self.step % C.time_wait(self.strategy_wait_secs) == 1:
            self.strategy_flag = True

        if self.step % C.time_wait(self.policy_wait_secs) == 1:
            self.policy_flag = True

    def safe_action(self, action, unit_type, args):
        if M.check_params(self, action, unit_type, args, 1):
            self.obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.step += 1
            self.update_result()
            self.set_flag()

    def select(self, action, unit_type, args):
        # safe select
        if M.check_params(self, action, unit_type, args, 0):
            self.obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.on_select = unit_type
            self.update_result()
            self.step += 1
            self.set_flag()

    @property
    def result(self):
        return self._result

    def update_result(self):
        if self.obs is None:
            return
        if self.obs.last() or self.env.state == environment.StepType.LAST:
            self.is_end = True
            outcome = 0
            o = self.obs.raw_observation
            player_id = o.observation.player_common.player_id
            for r in o.player_result:
                if r.player_id == player_id:
                    outcome = sc2_env._possible_results.get(r.result, 0)
            frames = o.observation.game_loop
            result = {}
            result['outcome'] = outcome
            result['reward'] = self.obs.reward
            result['frames'] = frames

            result['win'] = 0
            if result['reward'] == 1:
                result['win'] = 1

            self._result = result
            print('play end, total return', self.obs.reward)
            self.step = 0

    def one_hot_label(self, action_type_array, action_max_num):
        rows = action_type_array.shape[0]
        cols = action_max_num
        data = np.zeros((rows, cols))

        for i in range(rows):
            data[i, int(action_type_array[i])] = 1

        return data

    def get_values(self, values):
        # check if the game is end
        if self.is_end and self.result['reward'] != 0:
            return 0
        else:
            return values
