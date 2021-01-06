from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

from lib import utils as U
from lib import config as C
from lib import macro_action as M
from lib import environment
from lib import my_sc2_env as sc2_env
from lib.replay_buffer import Buffer
from mini_agent import ProtossAction

import torchcraft as tc
import torchcraft.Constants as tcc
from torchcraft.Constants import unittypes as T
import unit.protoss_unit as P
import random


class MiniSourceAgent(base_agent.BaseAgent):
    """Agent for source game of starcraft."""

    def __init__(self, index=0, rl_training=False, restore_model=False, global_buffer=None, net=None, strategy_agent=None, greedy_action=False):
        super(MiniSourceAgent, self).__init__()
        self.net = net
        self.index = index
        self.global_buffer = global_buffer
        self.restore_model = restore_model

        # model in brain
        self.strategy_agent = strategy_agent
        self.strategy_act = None

        # count num
        self.step = 0

        self.strategy_wait_secs = 3
        self.strategy_flag = False
        self.policy_wait_secs = 2
        self.policy_flag = True

        self.env = None
        self.obs = None

        # buffer
        self.local_buffer = Buffer()

        self.num_players = 2
        self.on_select = None
        self._result = None
        self._gases = None
        self.is_end = False

        self.greedy_action = greedy_action
        self.rl_training = rl_training

        self.reset_tc()

    def reset(self):
        super(MiniSourceAgent, self).reset()
        self.step = 0
        self.obs = None
        self._result = None
        self._gases = None
        self.is_end = False

        self.strategy_flag = False
        self.policy_flag = True

        self.local_buffer.reset()

        if self.strategy_agent is not None:
            self.strategy_agent.reset()

        self.reset_tc()

    def reset_tc(self):
        self.num_pylon = 0
        self.num_gateway = 0
        self.num_cyber = 0

        self.max_pylon = 20
        self.max_gateway = 5
        self.max_cyber = 3

        self.enemy_pos = None
        self.retreat_pos = None
        self.rally_pos = [455, 165]

        self.base = None
        self.resourceUnits = []
        self.vespeneUnits = []

        self.initial_scout = False
        self.initial = False

    def set_env(self, env):
        self.env = env

    def set_obs(self, state):
        self.obs = state

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

    def mini_step(self, action):
        if action == ProtossAction.Build_probe.value:
            M.train_unit(self, T.Protoss_Nexus, T.Protoss_Probe)

        elif action == ProtossAction.Build_zealot.value:
            M.train_unit(self, T.Protoss_Gateway, T.Protoss_Zealot)

        elif action == ProtossAction.Build_Stalker.value:
            M.train_unit(self, T.Protoss_Gateway, T.Protoss_Dragoon)

        elif action == ProtossAction.Build_pylon.value:
            if not self.initial_scout:
                M.scout_manager(self)
                self.initial_scout = True
            
            pos = self.placer_manager(self.base, T.Protoss_Pylon)
            M.build_by_worker(self, T.Protoss_Probe, T.Protoss_Pylon, pos)

        elif action == ProtossAction.Build_gateway.value:
            pos = self.placer_manager(self.base, T.Protoss_Gateway)
            M.build_by_worker(self, T.Protoss_Probe, T.Protoss_Gateway, pos)

        elif action == ProtossAction.Build_Assimilator.value:
            pos = self.placer_manager(self.base, T.Protoss_Assimilator)
            M.build_by_worker(self, T.Protoss_Probe, T.Protoss_Assimilator, pos)

        elif action == ProtossAction.Build_CyberneticsCore.value:
            pos = self.placer_manager(self.base, T.Protoss_Cybernetics_Core)
            M.build_by_worker(self, T.Protoss_Probe, T.Protoss_Cybernetics_Core, pos)

        elif action == ProtossAction.Attack.value:
            M.attack_step(self, [T.Protoss_Zealot, T.Protoss_Dragoon], self.enemy_pos)
            

        elif action == ProtossAction.Retreat.value:
            M.retreat_step(self, [T.Protoss_Zealot, T.Protoss_Dragoon], self.retreat_pos)
            #pass

        elif action == ProtossAction.Do_nothing.value:
            M.no_op(self)

    def calculate_features(self):
        state = self.obs
        myunits = state.units[state.player_id]

        self.mineral_worker_nums = 0
        self.gas_worker_nums = 0

        self.spent_mineral = 0
        self.spent_gas = 0

        self.probe_num = 0
        self.zealot_num = 0
        self.Stalker_num = 0
        self.army_nums = 0

        self.gateway_num = 0
        self.pylon_num = 0
        self.Assimilator_num = 0
        self.CyberneticsCore_num = 0

        for unit in myunits:
            if unit.type == T.Protoss_Probe:
                self.spent_mineral += P.Probe().mineral_price
                if unit.completed:
                    self.probe_num += 1
                    if unit.gathering_minerals:
                        self.mineral_worker_nums += 1
                    if unit.gathering_gas:
                        self.gas_worker_nums += 1
            if unit.type == T.Protoss_Zealot:
                self.spent_mineral += P.Zealot().mineral_price
                if unit.completed:
                    if unit.visible:
                        self.zealot_num += 1
                        self.army_nums += 1
                    else:
                        print('find invisible Zealot')
            if unit.type == T.Protoss_Dragoon:
                self.spent_mineral += P.Stalker().mineral_price
                self.spent_gas += P.Stalker().gas_price
                if unit.completed:
                    if unit.visible:
                        self.Stalker_num += 1
                        self.army_nums += 1
                    else:
                        print('find invisible Dragoon')
            if unit.type == T.Protoss_Pylon:
                self.spent_mineral += P.Pylon().mineral_price
                if unit.completed:
                    self.pylon_num += 1
            if unit.type == T.Protoss_Gateway:
                self.spent_mineral += P.Gateway().mineral_price
                if unit.completed:
                    self.gateway_num += 1
            if unit.type == T.Protoss_Assimilator:
                self.spent_mineral += P.Assimilator().mineral_price
                if unit.completed:
                    self.Assimilator_num += 1
            if unit.type == T.Protoss_Cybernetics_Core:
                self.spent_mineral += P.CyberneticsCore().mineral_price
                if unit.completed:
                    self.CyberneticsCore_num += 1

        self.mineral = state.frame.resources[state.player_id].ore
        self.gas = state.frame.resources[state.player_id].gas
        self.food_used = state.frame.resources[state.player_id].used_psi
        self.food_cup = state.frame.resources[state.player_id].total_psi

    def mapping_source_to_mini_by_rule(self, state):
        simple_input = np.zeros([20])
        simple_input[0] = 0  # self.time_seconds
        simple_input[1] = self.mineral_worker_nums  # self.mineral_worker_nums
        simple_input[2] = self.gas_worker_nums  # self.gas_worker_nums
        simple_input[3] = self.mineral  # self.mineral
        simple_input[4] = self.gas  # self.gas
        simple_input[5] = self.food_cup  # self.food_cup
        simple_input[6] = self.food_used  # self.food_used
        simple_input[7] = self.army_nums  # self.army_nums

        simple_input[8] = self.gateway_num  # self.gateway_num
        simple_input[9] = self.pylon_num  # self.pylon_num
        simple_input[10] = self.Assimilator_num  # self.Assimilator_num
        simple_input[11] = self.CyberneticsCore_num  # self.CyberneticsCore_num

        simple_input[12] = self.zealot_num  # self.zealot_num
        simple_input[13] = self.Stalker_num  # self.Stalker_num
        simple_input[14] = self.probe_num  # self.probe_num

        simple_input[15] = self.mineral + self.spent_mineral  # self.collected_mineral
        simple_input[16] = self.spent_mineral  # self.spent_mineral
        simple_input[17] = self.gas + self.spent_gas  # self.collected_gas
        simple_input[18] = self.spent_gas  # self.spent_gas
        simple_input[19] = 1  # self.Nexus_num

        return simple_input

    def play(self, verbose=False):
        self.play_train_mini(verbose=verbose)

    def play_train_mini(self, verbose=False):
        is_attack = False
        state_last = None

        while not self.obs.game_ended:
            #print('self.step:', self.step)
            #print('self.frame_from_bwapi:', self.obs.frame_from_bwapi)

            if self.obs.game_ended:
                break

            if self.step >= C.time_wait_sc1(900):
                self.env.send([[tcc.restart]])
                self.obs = self.env.recv()
                self.update_result(time_out=True)
                continue

            if not self.initial:
                self.initial_manager()
                self.initial = True

            self.safe_action([])

            if self.policy_flag and (not self.is_end):
                self.calculate_features()
                state_now = self.mapping_source_to_mini_by_rule(self.obs)
                if self.greedy_action:
                    action_prob, v_preds = self.net.policy.get_action_probs(state_now, verbose=False)
                    action = np.argmax(action_prob)
                else:
                    action, v_preds = self.net.policy.get_action(state_now, verbose=False)

                #print(ProtossAction(action).name)
                self.mini_step(action)
                if self.is_end:
                    #self.env.send([[tcc.restart]])
                    #self.obs = self.env.recv()
                    break

                if state_last is not None:
                    if 0:
                        print('state_last:', state_last, ', action_last:', action_last, ', state_now:', state_now)
                    v_preds_next = self.net.policy.get_values(state_now)
                    v_preds_next = self.get_values(v_preds_next)
                    reward = 0
                    self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)

                # continuous attack, consistent with mind-game
                if action == ProtossAction.Attack.value:
                    is_attack = True
                
                if is_attack:
                    self.mini_step(ProtossAction.Attack.value)
                    if self.is_end:
                        break

                state_last = state_now
                action_last = action
                self.policy_flag = False

            if self.strategy_flag and (not self.is_end):
                #print('self.strategy_flag:', self.strategy_flag)
                M.worker_manager(self)
                self.strategy_flag = False


        if self.rl_training:
            self.local_buffer.rewards[-1] += 1 * self.result['reward']  # self.result['win']
            #print(self.local_buffer.rewards)
            self.global_buffer.add(self.local_buffer)
            #print("add %d buffer!" % (len(self.local_buffer.rewards)))


    def set_flag(self):
        if self.step % C.time_wait_sc1(self.strategy_wait_secs) == 1:
            self.strategy_flag = True

        if self.step % C.time_wait_sc1(self.policy_wait_secs) == 1:
            self.policy_flag = True

    def safe_action(self, actions):
        if len(actions) > 0:
            pass
            #print("Sending actions: " + str(actions))
        self.env.send(actions)
        self.obs = self.env.recv()
        self.step += 1
        self.update_result()
        self.set_flag()

    @property
    def result(self):
        return self._result

    def judge_if_we_win_or_draw(self):
        enemy_untis = None
        for i in self.obs.units:
            if i != self.obs.player_id and i != self.obs.neutral_id:
                enemy_untis = self.obs.units[i]

        if enemy_untis is None:
            return True
        else:
            if len(enemy_untis) <= 4:
                army = M.selectArmy(self, [T.Protoss_Zealot, T.Protoss_Dragoon])
                if army is not None:
                    if len(army) >= 6:
                        return True
                    else:
                        print('len(army):', len(army))
        print('len(enemy_untis):', len(enemy_untis))
        return False

    def update_result(self, time_out=False):
        if self.obs is None:
            return
        if self.obs.waiting_for_restart:
            print("WAITING FOR RESTART...")
        if self.obs.game_ended:
            self.is_end = True
            frames = self.obs.frame_from_bwapi
            outcome = self.obs.game_won
            reward = self.obs.game_won

            if time_out:
                if self.judge_if_we_win_or_draw():
                    reward = 1
                    outcome = 1
                else:
                    reward = 0
                    outcome = 0
            elif not self.obs.game_won:
                reward = -1
                outcome = -1

            result = {}
            result['outcome'] = outcome
            result['reward'] = reward
            result['frames'] = frames

            self._result = result
            #print('play end, total result', self._result)
            self.step = 0

    def get_values(self, values):
        # check if the game is end
        if self.is_end and self.result['reward'] != 0:
            return 0
        else:
            return values

    def placer_manager(self, base, unit_type):
        pylon_size = 8
        gateway_size = 16
        cyber_size = 16

        if unit_type == T.Protoss_Pylon:
            initial_polyon_x = base.x - 16
            initial_polyon_y = base.y + 8
            #print(initial_polyon_x, initial_polyon_y)
            colum_index = 0
            row_index = self.num_pylon
            if self.num_pylon > 0.5 * self.max_pylon:
                row_index = self.num_pylon - 0.5 * self.max_pylon
                colum_index = 1

            target_x = initial_polyon_x - int(colum_index * 8)
            target_y = initial_polyon_y + int(row_index * pylon_size)
            self.num_pylon = (self.num_pylon + 1) % self.max_pylon
            return [target_x, target_y]
        elif unit_type == T.Protoss_Gateway:
            initial_gateway_x = base.x - 8
            initial_gateway_y = base.y + 10
            #print(initial_gateway_x, initial_gateway_y)
            target_x = initial_gateway_x 
            target_y = initial_gateway_y + self.num_gateway * gateway_size
            self.num_gateway = (self.num_gateway + 1) % self.max_gateway
            return [target_x, target_y]
        elif unit_type == T.Protoss_Cybernetics_Core:
            initial_cyber_x = base.x - 36
            initial_cyber_y = base.y + 10
            #print(initial_cyber_x, initial_cyber_y)
            target_x = initial_cyber_x 
            target_y = initial_cyber_y + self.num_cyber * cyber_size
            self.num_cyber = (self.num_cyber + 1) % self.max_cyber
            return [target_x, target_y]
        elif unit_type == T.Protoss_Assimilator:
            if len(self.vespeneUnits) > 0:
                vespene = self.vespeneUnits[0]
                target_x = vespene.x - 8
                target_y = vespene.y - 4
                #print(target_x, target_y)
                return [target_x, target_y]
        return [-1, -1]

    def initial_manager(self):
        self.obs = self.env.recv()
        state = self.obs

        frame_no = state.frame_from_bwapi
        #print('begin frame_no:', frame_no)

        myunits = state.units[state.player_id]

        # initial base
        for unit in myunits:
            if unit.type == T.Protoss_Nexus:
                self.base = unit
                break

        # initial mineral and gas
        neutralUnits = state.units[state.neutral_id]
        for u in neutralUnits:
            if u.type == T.Resource_Mineral_Field or u.type == T.Resource_Mineral_Field_Type_2 \
                    or u.type == T.Resource_Mineral_Field_Type_3:
                if u.visible:
                    self.resourceUnits.append(u)
            if u.type == T.Resource_Vespene_Geyser:
                if u.visible:
                    self.vespeneUnits.append(u)

        #print('resourceUnits:', len(self.resourceUnits))
        #print('vespeneUnits:', len(self.vespeneUnits))

        # initial workers
        actions = []
        for unit in myunits:
            if unit.type == T.Protoss_Probe and unit.completed:
                if unit.idle:
                    target = M.get_closest(unit.x, unit.y, self.resourceUnits)
                    actions.append([
                        tcc.command_unit, unit.id,
                        tcc.unitcommandtypes.Right_Click_Unit,
                        target.id,
                    ])

        self.safe_action(actions)
