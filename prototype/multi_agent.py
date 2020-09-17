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

import numpy as np
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions
from logging import warning as logging

import lib.utils as U
import param as P
from lib import config as C
from lib import option as M
from lib import environment
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer
from uct.numpy_impl import *
from .dynamic_network import DynamicNetwork
from .hier_network import HierNetwork


class MultiAgent(base_agent.BaseAgent):
    """My first agent for starcraft."""

    def __init__(self, index=0, rl_training=False, restore_model=False, restore_internal_model=False, global_buffer=None, net=None,
                 use_mcts=False, num_reads=0, policy_in_mcts=None, dynamic_net=None, use_dyna=False,
                 dyna_steps_fisrt=0, dyna_decrese_counter=0):
        super(MultiAgent, self).__init__()
        self.net = net
        self.index = index
        self.global_buffer = global_buffer
        self.restore_model = restore_model

        self.restore_dynamic = restore_internal_model

        # count num
        self.step = 0

        self.policy_wait_secs = 2
        self.policy_flag = True

        self.env = None
        self.obs = None

        # buffer
        self.local_buffer = Buffer()

        self.num_players = 2
        self.on_select = None
        self._result = None
        self.is_end = False

        self.rl_training = rl_training

        self.reward_type = 0

        # mcts about
        self.use_mcts = use_mcts
        self.num_reads = num_reads
        self.policy_in_mcts = policy_in_mcts
        self.dynamic_net = dynamic_net

        # dyna about
        self.use_dyna = use_dyna
        self.dyna_steps_fisrt = dyna_steps_fisrt
        self.dyna_decrese_counter = dyna_decrese_counter
        self.dyna_steps = dyna_steps_fisrt

    def reset(self):
        super(MultiAgent, self).reset()
        self.step = 0
        self.obs = None
        self._result = None
        self.is_end = False

        self.policy_flag = True

        self.local_buffer.reset()

    def set_env(self, env):
        self.env = env

    def init_network(self):
        self.net.initialize()
        if self.restore_model:
            self.net.restore_policy()
        if self.restore_dynamic:
            # print('self.net.restore_dynamic()')
            self.net.restore_dynamic("")
            # self.dynamic_net.restore_sl_model("")

    def reset_old_network(self):
        self.net.reset_old_network()

    def save_model(self):
        self.net.save_policy()

    def update_network(self, result_list):
        self.net.Update_policy(self.global_buffer)
        # self.net.Update_internal_model(self.global_buffer)
        self.net.Update_result(result_list)
        # self.update_policy_in_mcts()

    def update_policy_in_mcts(self):
        values = self.global_buffer.values
        values_array = np.array(values).astype(dtype=np.float32).reshape(-1)
        print('values_array:', values_array)
        min_v = np.min(values_array)
        print('min_v:', min_v)
        max_v = np.max(values_array)
        print('max_v:', max_v)
        self.policy_in_mcts.update_min_max_v(min_v, max_v)

        mean_v = np.mean(values_array)
        print('mean_v:', mean_v)
        std_v = np.std(values_array)
        print('std_v:', std_v)
        self.policy_in_mcts.update_mean_std_v(mean_v, std_v)

    def update_summary(self, counter):
        self.net.Update_summary(counter)
        #self.global_update_count = counter
        # every some global_update_count dyna_step-1
        # if self.use_dyna:
        #    self.dyna_steps = 5 - self.global_update_count // 20
        #logging("global_update_count: %d, dyna_steps: %d" % (self.global_update_count, self.dyna_steps))

    def get_policy_input(self, obs):
        high_input, tech_cost, pop_num = U.get_input(obs)
        policy_input = np.concatenate([high_input, tech_cost, pop_num], axis=0)
        return policy_input

    def tech_step(self, tech_action):
        if tech_action == 0:  # nothing
            self.safe_action(C._NO_OP, 0, [])
        elif tech_action == 1:  # worker
            M.mineral_worker(self)
        elif tech_action == 2:  # pylon
            no_unit_index = U.get_unit_mask_screen(self.obs, size=2)
            pos = U.get_pos(no_unit_index)
            M.build_by_idle_worker(self, C._BUILD_PYLON_S, pos)

    def get_simple_state(self, obs):
        simple_state = U.get_simple_state(obs)
        return simple_state

    def set_dyna_steps(self):
        global_steps = self.net.get_global_steps()
        # every some global_update_count dyna_step-1
        self.dyna_steps = max(self.dyna_steps_fisrt - global_steps // self.dyna_decrese_counter, 0)
        logging("global_update_count: %d, dyna_steps: %d" % (global_steps, self.dyna_steps))

    def play(self, verbose=False):
        M.set_source(self)

        if self.use_dyna:
            self.set_dyna_steps()

        tech_act, v_preds = np.zeros(2)
        last_obs, state_last = None, None
        action_last, state_now = None, None
        step = 0

        while True:
            self.safe_action(C._NO_OP, 0, [])

            # only one second do one thing
            if self.policy_flag:
                now_obs = self.obs
                state_now = self.get_simple_state(now_obs)

                # (s_last, action) -> s_now,
                if last_obs:
                    #rule_state_diff = self.predict_state_diff_by_rule(state_last, action_last)
                    #print('state_last:', state_last, ', action_last:', action_last)
                    #print('rule_state_diff:', rule_state_diff, 'state_diff:', state_now - state_last)
                    if verbose:
                        print('state_last:', state_last, ', action_last:', action_last, ', state_now:', state_now)
                    # add data to buffer
                    reward = self.get_mineral_reward(last_obs, now_obs)
                    if self.reward_type == 0:
                        reward = 0
                    if verbose:
                        print("reward: ", reward)
                    v_preds_next = self.net.policy.get_values(state_now)
                    v_preds_next = self.get_values(v_preds_next)
                    self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)

                # predict action
                tech_act, v_preds = self.net.policy.get_action(state_now, verbose=False)

                # print mcts choose action
                if self.use_mcts:
                    game_state = GameState(dynamic_net=self.dynamic_net, state=state_now)
                    mcts_act = UCT_search(game_state=game_state, num_reads=self.num_reads, policy_in_mcts=self.policy_in_mcts)
                    if 1:
                        #print('state_now:', state_now)
                        print('mcts_act: ', mcts_act)
                        print('\n')
                    tech_act = mcts_act[0]

                # use dyna to add predicted trace
                if self.use_dyna:
                    self.simulated(state_now, tech_act, v_preds, self.dyna_steps)

                    # do action
                self.tech_step(tech_act)
                # finish
                step += 1
                last_obs = now_obs
                state_last = state_now
                action_last = tech_act
                self.policy_flag = False

            if self.is_end:
                if self.rl_training:
                    if self.reward_type == 0:
                        final_mineral = now_obs.raw_observation.observation.player_common.minerals
                        self.local_buffer.rewards[-1] += final_mineral
                        print('final_mineral:', final_mineral)
                        if verbose:
                            print('final_reward:', self.local_buffer.rewards[-1])
                    self.global_buffer.add(self.local_buffer)
                break

    def simulated(self, state_now, action_now, v_preds_now, dyna_steps, append_to_buffer=True):
        game_state = GameState(dynamic_net=self.dynamic_net, state=state_now)
        sim_buffer = Buffer()
        for _ in range(dyna_steps):
            # simulate next state
            next_game_state = game_state.play(action_now, verbose=False)

            state_last = state_now
            action_last = action_now
            state_now = next_game_state.obs()

            v_preds_last = v_preds_now
            v_preds_now = self.net.policy.get_values(state_now)
            v_preds_now = self.get_values(v_preds_now)

            reward = state_now[1] - state_last[1]

            if append_to_buffer:
                sim_buffer.append(state_last, action_last, state_now, reward, v_preds_last, v_preds_now)

            action_now, v_preds_now = self.net.policy.get_action(state_now, verbose=False)
            game_state = next_game_state

        #print('sim_buffer:', sim_buffer)
        self.global_buffer.add(sim_buffer, add_return=False)

    def get_mineral_reward(self, old_obs, now_obs):
        state_last = self.get_simple_state(old_obs)
        state_now = self.get_simple_state(now_obs)

        mineral_reward = state_now[1] - state_last[1]
        return mineral_reward

    def set_flag(self):
        if self.step % C.time_wait(self.policy_wait_secs) == 1:
            self.policy_flag = True

    def safe_action(self, action, unit_type, args):
        if M.check_params(self, action, unit_type, args, 1):
            obs = self.env.step([sc2_actions.FunctionCall(action, args)])[0]
            self.obs = obs
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

        # else:
        # print('Unavailable_actions id:', action, ' and type:', unit_type, ' and args:', args)

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
            self._result = result
            # print('play end, total return', self.obs.reward)

    def get_values(self, values):
        # check if the game is end
        if self.is_end and self.result['reward'] != 0:
            return 0
        else:
            return values
