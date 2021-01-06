from enum import Enum

import numpy as np
import math
import lib.config as C
import lib.utils as U
import time

from strategy.agent import Agent as A
from unit.units import Army

import unit.protoss_unit as P


class ProtossAction(Enum):
    Do_nothing = 0
    Build_worker = 1
    Build_zealot = 2
    Build_pylon = 3
    Build_gateway = 4
    Attack = 5
    Move = 6
    Defend = 7
    Build_sub_base = 8
    Build_cannon = 9
    All = 10


class Protoss(A):

    def __init__(self, agent_id=0, global_buffer=None, net=None, restore_model=False):
        A.__init__(self, agent_id=agent_id, global_buffer=global_buffer,
                   net=net, restore_model=restore_model)
        self.gateway_num = 0
        self.pylon_num = 0
        self.zealot_num = 0
        self.collected_mineral = 0
        self.MAX_ACTIONS = ProtossAction.All.value

    def __str__(self):
        return str(self.time_seconds) + ', ' + str(self.mineral) + \
            ', ' + str(self.mineral_worker_nums) + ', ' + str(self.zealot_num) + ', ' + str(self.food_cap)

    def reset(self):
        super().reset()
        self.gateway_num = 0
        self.pylon_num = 0
        self.zealot_num = 0
        self.collected_mineral = 0

    def obs(self):
        simple_input = np.zeros([11])
        simple_input[0] = 0  # self.time_seconds
        simple_input[1] = self.mineral_worker_nums
        simple_input[2] = self.gas_worker_nums
        simple_input[3] = self.mineral
        simple_input[4] = self.gas
        simple_input[5] = self.food_cap
        simple_input[6] = self.food_used
        simple_input[7] = self.army_nums
        simple_input[8] = self.gateway_num
        simple_input[9] = self.pylon_num
        simple_input[10] = self.zealot_num
        return simple_input

    def set_obs(self, state):
        self.mineral_worker_nums = state[1]
        self.gas_worker_nums = state[2]
        self.mineral = state[3]
        self.gas = state[4]
        self.food_cap = state[5]
        self.food_used = state[6]
        self.army_nums = state[7]
        self.gateway_num = state[8]
        self.pylon_num = state[9]
        self.zealot_num = state[10]

    def get_next_state(self, action):
        self.env.step(self.player_id, action)
        return self.obs()

    @property
    def result(self):
        return self._result

    def play_with_mpc(self, verbose=False):
        max_steps = 100
        state_now = self.obs()
        if verbose:
            print('initial state:', state_now)
            print('initial env:', self.env)
        state_last, action_last = None, None
        for i in range(max_steps):
            if self.is_end or i == max_steps - 1:
                if verbose:
                    print(self.local_buffer.rewards)
                if self.env.win_index == self.player_id:
                    pass
                self._result = sum(self.local_buffer.rewards)
                # self.global_buffer.add(self.local_buffer)
                break

            if state_last is not None:
                reward = self.get_mineral_reward(state_last, state_now)
                if True:
                    print('reward:', reward)
                self.local_buffer.append(state_last, action_last, state_now, reward, 0, 0)

            action, v_preds = self.mpc.get_action(state_now, agent_clone=self, verbose=verbose)
            state_last = state_now
            state_now = self.get_next_state(action)
            if verbose:
                print('state now:', state_now.astype(dtype=np.int32))
                time.sleep(1)
            action_last = action

    def play_with_rl(self, verbose=False):
        max_steps = 125
        state_now = self.obs()
        if verbose:
            print('initial state:', state_now)
            print('initial env:', self.env)

        state_last, action_last = None, None
        for i in range(max_steps):
            if self.is_end or i == max_steps - 1:
                if self.env.win_index == self.player_id:
                    ratio = (i + 1) / float(max_steps)
                    the_reward = 1. - ratio / 1.5
                    self.local_buffer.rewards[-1] += the_reward
                    self._result = the_reward

                if verbose:
                    print(self.local_buffer.rewards)

                #self._result = sum(self.local_buffer.rewards)
                self.global_buffer.add(self.local_buffer)
                break

            if state_last is not None:
                reward = 0  # = self.get_pop_reward(state_last, state_now)
                if 0:
                    print('reward:', reward)
                v_preds_next = self.net.policy.get_values(state_now)
                v_preds_next = self.get_values(v_preds_next)
                self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)

            action, v_preds = self.net.policy.get_action(state_now, verbose=False)
            state_last = state_now
            state_now = self.get_next_state(action)
            if verbose:
                print('state now:', state_now.astype(dtype=np.int32))
                print('action:', action)
                time.sleep(1)
            action_last = action

    def get_pop_reward(self, state_last, state_now):
        pop_reward = state_now[6] - state_last[6]
        return pop_reward

    def get_mineral_reward(self, state_last, state_now):
        mineral_reward = state_now[3] - state_last[3]
        return mineral_reward

    def get_values(self, values):
        # check if the game is end
        if self.is_end and self.result != 0:
            return 0
        else:
            return values

    def get_action_by_policy(self, obs):
        act, v_preds = self.net.policy.get_action(obs, verbose=True)
        return act, v_preds

    '''def get_policy_action(self, obs):
        random = np.random.randint(self.MAX_ACTIONS)
        action = random
        return action'''

    def fullfill_technology(self, unit):
        if type(unit) == P.Zealot:
            if self.gateway_num > 0:
                return True

        return False

    def fullfill_creature_condition(self, unit):
        if self.mineral >= unit.mineral_price and self.gas >= unit.gas_price:
            if self.food_cap >= self.food_used + unit.food_used and self.fullfill_technology(unit):
                return True
        else:
            return False

    def win(self):
        if self.zealot_num >= 8:
            return True
        else:
            return False

    def get_build_num(self, unit):
        max_n = self.gateway_num
        n = 1
        #print('max_n:', max_n)
        for i in range(max_n):
            if unit.mineral_price * i < self.mineral and unit.food_used * i + self.food_used < self.food_cap:
                continue
            else:
                n = i - 1
                break
        #print('n:', n)
        return n

    def step(self, action):
        if action == ProtossAction.Build_worker.value:
            if self.mineral >= 50 and self.food_used < self.food_cap:
                self.mineral_worker_nums += 1
                self.food_used += 1
                self.mineral -= 50
        elif action == ProtossAction.Build_zealot.value:
            Zealot = P.Zealot()
            if self.fullfill_creature_condition(Zealot):
                n = self.get_build_num(Zealot)
                self.army_nums += n
                self.zealot_num += n
                self.food_used += Zealot.food_used * n
                self.mineral -= Zealot.mineral_price * n
                self.add_unit(Zealot, n)
        elif action == ProtossAction.Build_pylon.value:
            if self.mineral >= 100:
                self.building_nums += 1
                self.food_cap += 8
                self.pylon_num += 1
                self.mineral -= 100
        elif action == ProtossAction.Build_gateway.value:
            if self.mineral >= 150 and self.pylon_num >= 1:
                self.gateway_num += 1
                self.building_nums += 1
                self.mineral -= 150
        elif action == ProtossAction.Attack.value:
            if self.military_num() > 0:
                #print('order:', self.env.army[self.player_id].order)
                self.env.army[self.player_id].order = Army.Order.ATTACK
                #print('order:', self.env.army[self.player_id].order)

        elif action == ProtossAction.Defend.value:
            if self.military_num() > 0:
                self.env.army[self.player_id].order = Army.Order.DEFEND
        elif action == ProtossAction.Build_sub_base.value:
            pass
        elif action == ProtossAction.Build_cannon.value:
            pass

        # update mineral
        self.collected_mineral += min(self.mineral_worker_nums, 16) * 3
        if self.collected_mineral <= 10000:
            self.mineral += min(self.mineral_worker_nums, 16) * 3

        self.time_seconds += 5

        # update population
        if self.military_num() == 0:
            #print('order:', self.env.army[self.player_id].order)
            self.env.army[self.player_id].order = Army.Order.NOTHING
            #print('order:', self.env.army[self.player_id].order)
        else:
            self.army_nums = self.military_num()
            self.zealot_num = self.military_num()
            self.food_used = self.military_num() * 2 + self.mineral_worker_nums + self.gas_worker_nums
