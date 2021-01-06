from enum import Enum, unique

import math
import numpy as np
import lib.config as C
import lib.utils as U
import time

from strategy.agent import Agent as A
from unit.units import Army
import unit.zerg_unit as Z

@unique
class ZergAction(Enum):
    Do_nothing = 0
    Attack = 1
    Defend = 2
    Build_drone = 3
    Build_overlord = 4
    Build_queen = 5
    Build_zergling = 6
    Build_roach = 7
    Build_extractor = 8
    Build_spawningpool = 9
    Build_roachwarren = 10
    Gather_gas = 11
    Gather_mineral = 12


class Zerg(A):

    def __init__(self, agent_id=0, global_buffer=None, net=None, restore_model=False):
        super().__init__(agent_id=agent_id, global_buffer=global_buffer, net=net, restore_model=restore_model)
        self.init_features()
        self.init_rl_param()
        
    def __str__(self):
        return str(self.time_seconds) + ', ' + str(self.mineral) + \
            ', ' + str(self.mineral_worker_nums) + ', ' + str(self.zergling_num) + ', ' + str(self.food_cap)

    def init_features(self):
        self.mineral_reserves = 10800
        self.gas_reserves = 4500
        self.hatchery_num = 1
        self.larva_num = 3
        self.overlord_num = 1
        self.extractor_num = 0
        self.spawningpool_num = 0
        self.roachwarren_num = 0
        self.queen_num = 0
        self.zergling_num = 0
        self.roach_num = 0
        self.collected_minerals = 0
        self.spent_minerals = 0
        self.collected_gas = 0
        self.spent_gas = 0

        self.workers_list = {Z.Drone: 12, Z.Overlord: 1}
        self.building_list = {Z.Hatchery: 1}
        self.units_in_production = []
        self.buildings_in_production = []

    def init_rl_param(self):
        self.policy = None
        if self.global_buffer is not None:
            self.policy = self.net.policy
        else:
            self.policy = self.net.policy_old

        self.state_last = None
        self.state_now = self.obs()

        self.action_last = None
        self.action = None
        self.v_preds = None

    def reset(self, pos):
        super().reset(pos)
        self.init_features()
        self.init_rl_param()

    def obs(self):
        simple_input = np.zeros((21))
        simple_input[0] = 0
        simple_input[1] = self.mineral_worker_nums
        simple_input[2] = self.gas_worker_nums
        simple_input[3] = self.mineral
        simple_input[4] = self.gas
        simple_input[5] = self.food_cap
        simple_input[6] = self.food_used
        simple_input[7] = self.army_nums
        simple_input[8] = self.larva_num
        simple_input[9] = self.overlord_num
        simple_input[10] = self.spawningpool_num
        simple_input[11] = self.roachwarren_num
        simple_input[12] = self.zergling_num
        simple_input[13] = self.roach_num
        simple_input[14] = self.extractor_num
        simple_input[15] = self.queen_num
        simple_input[16] = self.building_nums
        simple_input[17] = self.collected_minerals
        simple_input[18] = self.spent_minerals
        simple_input[19] = self.collected_gas
        simple_input[20] = self.spent_gas
        
        return simple_input

    def get_next_state(self, action):
        self.step(action)
        return self.obs()
    
    @property
    def result(self):
        return self._result    

    def step_auto(self, verbose=False):
        if self.env.is_end:
            final_reward = 0
            if self.env.win_index == self.player_id:
                final_reward = 1
            elif self.env.win_index == -1:
                final_reward = 0
            else: final_reward = -1
        
            self.local_buffer.rewards[-1] += final_reward
            self._result = final_reward

            if self.global_buffer is not None:
                self.global_buffer.add(self.local_buffer)
            
            return

        if self.state_last is not None:
            v_preds_next = self.policy.get_values(self.state_now)
            v_preds_next = self.get_values(v_preds_next)
            reward = 0
            self.local_buffer.append(self.state_last, self.action_last, self.state_now, reward, self.v_preds, v_preds_next)

        self.action, self.v_preds = self.policy.get_action(self.state_now, verbose=False)
        self.state_last = self.state_now
        self.state_now = self.get_next_state(self.action)

        # if True:
        #     print('state now:', self.state_now.astype(dtype=np.int32))
        #     print('action:', ZergAction(int(self.action)).name)
        #     time.sleep(1)
        self.action_last = self.action


    def play_with_rl(self, verbose=False):
        max_steps = 250
        state_now = self.obs()

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
                
                self.global_buffer.add(self.local_buffer)
                break
            
            if state_last is not None:
                reward = 0
                v_preds_next = self.net.policy.get_values(state_now)
                v_preds_next = self.get_values(v_preds_next)
                self.local_buffer.append(state_last, action_last, state_now, reward, v_preds, v_preds_next)
            
            action, v_preds = self.net.policy.get_action(state_now, verbose=False)
            state_last = state_now
            state_now = self.get_next_state(action)
            # if verbose:
            #     print("State now: ", state_now.astype(dtype=np.int32))
            #     print("Action: ", ZergAction(int(action)).name)
            #     time.sleep(1)
            action_last = action

    def get_pop_reward(self, state_last, state_now):
        pop_reward = state_now[6] - state_last[6]
        return pop_reward / 200

    def get_resource_reward(self, state_last, state_now):
        mineral_reward = (state_now[3] - state_last[3]) / 10800
        gas_reward = (state_now[4] -state_last[4]) / 4500
        return mineral_reward + gas_reward

    def get_values(self, values):
        if self.is_end and self.result != 0:
            return 0
        else:
            return values
    
    def get_action_by_policy(self, obs):
        act, v_preds = self.net.policy.get_action(obs, verbose=False)
        return act, v_preds

    def fullfill_technology(self, unit):
        if type(unit) == Z.Extractor:
            if self.hatchery_num > 0 and self.extractor_num < self.hatchery_num * 2:
                return True

        elif type(unit) == Z.SpawningPool:
            if self.hatchery_num > 0:
                return True

        elif type(unit) == Z.RoachWarren:
            if self.spawningpool_num > 0:
                return True

        # elif type(unit) == Z.EvolutionChamber:
        #     if self.hatchery_num > 0:
        #         return True

        # elif type(unit) == Z.SpineCrawler:
        #     if self.evolutionchamber_num > 0:
        #         return True

        elif type(unit) == Z.Drone:
            if self.hatchery_num > 0:
                return True

        elif type(unit) == Z.Overlord:
            if self.hatchery_num > 0:
                return True

        elif type(unit) == Z.Queen:
            if self.hatchery_num > 0:
                return True

        elif type(unit) == Z.Zergling:
            if self.spawningpool_num > 0:
                return True

        elif type(unit) == Z.Roach:
            if self.roachwarren_num > 0:
                return True
        
        return False

    def fullfill_creature_condition(self, unit):
        if self.mineral >= unit.mineral_price and self.gas >= unit.gas_price:
            if type(unit) == Z.Overlord and self.larva_num > 0:
                return True
            elif self.food_cap >= self.food_used + unit.food_used and self.fullfill_technology(unit):
                if type(unit) == Z.Queen:
                    return True
                elif self.larva_num > 0:
                    return True

        return False

    def fullfill_building_condition(self, building):
        if self.mineral >= building.mineral_price and self.gas >= building.gas_price:
            if self.mineral_worker_nums > 0 and self.fullfill_technology(building):
                return True
        
        return False
    
    def get_build_num(self, unit):
        for i in range(self.larva_num, -1, -1):
            if unit.mineral_price * i <= self.mineral and unit.gas_price * i <= self.gas:
                return i

    def update(self):
        self.update_resources()

        self.update_unit_progress()
        self.update_units()
        self.update_building_progress()
        self.update_buildings()

        self.update_supply()
        self.update_larva()
        self.update_army()

        self.time_seconds += self.time_per_step


    def update_army(self):
        if self.military_num() == 0:
            self.env.army[self.player_id].order = Army.Order.NOTHING
            self.env.army[self.player_id].pos = self.pos

    def update_supply(self):
        self.food_used = 0
        self.food_used = sum(map(lambda x: x[0]().food_used * x[1], list(self.workers_list.items()) + list(self.army_list.items())))
        self.food_used += sum(map(lambda x: x.food_used, self.units_in_production))

        self.food_cap = min(self.hatchery_num * 6 + self.overlord_num * 8, 200)

    def update_resources(self):
        mineral_income, gas_income = self.get_income()

        self.mineral += mineral_income
        self.gas += gas_income
        self.collected_minerals += mineral_income
        self.collected_gas += gas_income

    def get_income(self):
        mineral_income = self.mineral_worker_nums * 5 if self.mineral_worker_nums <= 16 else 16 * 5
        if self.mineral_reserves - mineral_income > 0:
            self.mineral_reserves -= mineral_income
        else:
            mineral_income = self.mineral_reserves
            self.mineral_reserves = 0

        max_gas_workers = self.extractor_num * 3
        gas_income = self.gas_worker_nums * 4 if self.gas_worker_nums < max_gas_workers else max_gas_workers * 4
        if self.gas_reserves - gas_income >= 0:
            self.gas_reserves -= gas_income
        else:
            gas_income = self.gas_reserves
            self.gas_reserves = 0

        return mineral_income, gas_income

    def update_units(self):
        self.army_nums = sum(self.army_list.values())

        self.drone_num = self.workers_list.get(Z.Drone, 0)
        self.overlord_num = self.workers_list.get(Z.Overlord, 0)
        self.queen_num = self.workers_list.get(Z.Queen, 0)
        self.zergling_num = self.army_list.get(Z.Zergling, 0)
        self.roach_num = self.army_list.get(Z.Roach, 0)

    def update_buildings(self):
        # self.building_nums = sum(self.building_list.values())

        self.hatchery_num = self.building_list.get(Z.Hatchery, 0)
        self.extractor_num = self.building_list.get(Z.Extractor, 0)
        self.spawningpool_num = self.building_list.get(Z.SpawningPool, 0)
        self.roachwarren_num = self.building_list.get(Z.RoachWarren, 0)
        self.evolutionchamber_num = self.building_list.get(Z.EvolutionChamber, 0)
        self.spinecrawler_num  = self.building_list.get(Z.SpineCrawler, 0)

    def update_larva(self):
        if (self.time_seconds // self.time_per_step) % 2 == 0:
            self.larva_num = min(self.larva_num + 1, 3)
            if self.queen_num != 0:
                self.larva_num += 1

    def build_units(self, unit_type):
        unit = unit_type()
        
        # print(self.fullfill_creature_condition(unit))
        if not self.fullfill_creature_condition(unit):
            return

        self.units_in_production.append(unit)

        if unit_type != Z.Queen:
            self.larva_num -= 1

        self.mineral -= unit.mineral_price
        self.gas -= unit.gas_price
        self.spent_minerals += unit.mineral_price
        self.spent_gas += unit.gas_price

    def update_unit_progress(self):
        for unit in self.units_in_production:
            unit.progress += self.time_per_step
            unit_type = type(unit)
            if unit.progress >= unit.build_time:
                self.units_in_production.remove(unit)
                if unit_type == Z.Drone:
                    self.mineral_worker_nums += 1
                    self.add_unit(unit, 1, u_type='worker')
                elif unit_type == Z.Queen or unit_type == Z.Overlord:
                    self.add_unit(unit, 1, u_type='worker')
                elif unit_type == Z.Zergling:
                    self.add_unit(unit, 2, u_type='army')
                else:
                    self.add_unit(unit, 1, u_type='army')

    def build_structures(self, building_type):
        building = building_type()
        if not self.fullfill_building_condition(building):
            return

        self.building_nums += 1
        self.buildings_in_production.append(building)
        self.mineral -= building.mineral_price
        self.gas -= building.gas_price
        self.mineral_worker_nums -= 1
        self.workers_list[Z.Drone] -= 1
        self.spent_minerals += building.mineral_price
        self.spent_gas += building.gas_price

    def update_building_progress(self):
        for building in self.buildings_in_production:
            building.progress += self.time_per_step
            if building.progress >= building.build_time:
                self.buildings_in_production.remove(building)
                self.add_building(building)

    def step(self, action):
        if action == ZergAction.Build_drone.value:
            self.build_units(Z.Drone)

        elif action == ZergAction.Build_extractor.value:
           self.build_structures(Z.Extractor)
        
        elif action == ZergAction.Gather_gas.value:
            if self.mineral_worker_nums > 0 and self.extractor_num > 0:
                self.mineral_worker_nums -= 1
                self.gas_worker_nums += 1
        
        elif action == ZergAction.Gather_mineral.value:
            if self.gas_worker_nums > 0:
                self.mineral_worker_nums += 1
                self.gas_worker_nums -= 1

        elif action == ZergAction.Build_queen.value:
            self.build_units(Z.Queen)

        elif action == ZergAction.Build_zergling.value:
            self.build_units(Z.Zergling)

        elif action == ZergAction.Build_roach.value:
            self.build_units(Z.Roach)

        elif action == ZergAction.Build_overlord.value:
            self.build_units(Z.Overlord)
        
        elif action == ZergAction.Build_spawningpool.value:
            self.build_structures(Z.SpawningPool)

        elif action == ZergAction.Build_roachwarren.value:
            self.build_structures(Z.RoachWarren)
        
        # elif action == ZergAction.Build_evolutionchamber.value:
        #     self.build_structures(Z.EvolutionChamber)

        # elif action == ZergAction.Build_spinecrawler.value:
        #     self.build_structures(Z.SpineCrawler)

        elif action == ZergAction.Attack.value:
            if self.military_num() > 0:
                self.env.army[self.player_id].order = Army.Order.ATTACK
        
        elif action == ZergAction.Defend.value:
            if self.military_num() > 0:
                self.env.army[self.player_id].order = Army.Order.DEFEND
        
        self.update()


class DummyZerg(A):

    def __init__(self, diff=5):
        super().__init__(self)
        self.army_nums = 4
        self.building_nums = 3
        self.difficulty = diff

    def step_auto(self, verbose=False):
        if self.env.all_steps % 5 == 1:
            self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 65 and self.difficulty >= 3:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 55 and self.difficulty >= 5:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 45 and self.difficulty >= 7:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 35 and self.difficulty >= 9:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 25 and self.difficulty >= 11:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 20 and self.difficulty >= 13:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 15 and self.difficulty >= 15:
                self.add_unit(Z.Zergling(), 1)
            if self.env.all_steps >= 10 and self.difficulty >= 17:
                self.add_unit(Z.Zergling(), 1)


        if self.military_num() > 50:
            self.env.army[self.player_id].order = Army.Order.ATTACK
        else:
            self.env.army[self.player_id].order = Army.Order.DEFEND

    def set_diff(self, diff):
        self.difficulty = diff

    def reset(self, pos):
        super().reset(pos)
        self.army_list = {}
        self.building_list = {}
        self.get_power()
        self.army_nums = 4
        self.building_nums = 3

    def get_power(self):
        self.add_unit(Z.Zergling(), 4, u_type='army')
        # if self.difficulty >= 9:
        #     self.add_unit(Z.Zergling(), min(self.difficulty - 8, 4), u_type='army')
        # self.add_unit(Z.Roach(), min(self.difficulty - 1, 3), u_type='army')        
        # self.add_unit(Z.Hydralisk(), min(self.difficulty - 1, 10), u_type='army')
        self.add_building(Z.Hatchery(), 1)
        self.add_building(Z.SpawningPool(), 1)
        self.add_building(Z.RoachWarren(), 1)