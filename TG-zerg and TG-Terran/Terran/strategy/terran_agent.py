from enum import Enum, unique

import math
import numpy as np
import lib.config as C
import lib.utils as U
import time

from strategy.agent import Agent as A
from unit.units import Army
import unit.terran_unit as T
import unit.zerg_unit as Z

@unique
class TerranAction(Enum):
    Do_nothing = 0
    Attack = 1
    Defend = 2

    Build_SupplyDepot = 3
    Build_Refinery = 4
    Build_Barracks = 5

    Build_SCV = 6
    Build_Marine = 7
    Build_Reaper = 8

    Gather_gas = 9
    Gather_mineral = 10


class Terran(A):

    def __init__(self, agent_id=0, global_buffer=None, net=None, restore_model=False):
        super().__init__(agent_id=agent_id, global_buffer=global_buffer, net=net, restore_model=restore_model)
        self.init_features()
        self.init_rl_param()

    def __str__(self):
        return str(self.time_seconds) + ', ' + str(self.mineral) + \
               ', ' + str(self.mineral_worker_nums) + str(self.food_cap)

    def init_features(self):
        self.mineral_reserves = 10800
        self.gas_reserves = 4500

        self.CommandCenter_num = 1
        self.Refinery_num = 0
        self.SupplyDepot_num = 0
        self.Barracks_num = 0

        self.busy_barracks_num = 0

        self.SCV_num = 12
        self.Marine_num = 0
        self.Reaper_num = 0

        self.workers_list = {T.SCV: 12}
        self.building_list = {T.CommandCenter: 1}
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
        simple_input = np.zeros(17)
        simple_input[0] = 0
        simple_input[1] = self.mineral_worker_nums
        simple_input[2] = self.gas_worker_nums
        simple_input[3] = self.mineral
        simple_input[4] = self.gas
        simple_input[5] = self.food_cap
        simple_input[6] = self.food_used
        simple_input[7] = self.army_nums

        simple_input[8] = self.Refinery_num
        simple_input[9] = self.SupplyDepot_num
        simple_input[10] = self.Barracks_num

        simple_input[11] = self.Marine_num
        simple_input[12] = self.Reaper_num

        simple_input[13] = self.collected_mineral
        simple_input[14] = self.collected_gas

        simple_input[15] = self.collected_mineral - self.mineral
        simple_input[16] = self.collected_gas - self.gas

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
            else:
                final_reward = -1

            self.local_buffer.rewards[-1] += final_reward
            self._result = final_reward

            if self.global_buffer is not None:
                self.global_buffer.add(self.local_buffer)

            return

        if self.state_last is not None:
            v_preds_next = self.policy.get_values(self.state_now)
            v_preds_next = self.get_values(v_preds_next)
            reward = 0
            self.local_buffer.append(self.state_last, self.action_last, self.state_now, reward, self.v_preds,
                                     v_preds_next)

        self.action, self.v_preds = self.policy.get_action(self.state_now, verbose=False)
        self.state_last = self.state_now
        self.state_now = self.get_next_state(self.action)

        # if True:
        #     #print('player_id:', self.player_id)
        #     print('state now:', self.state_now.astype(dtype=np.int32))
        #     print('action:', TerranAction(int(self.action)).name)
        #     time.sleep(1)
        self.action_last = self.action

    def get_pop_reward(self, state_last, state_now):
        pop_reward = state_now[6] - state_last[6]
        return pop_reward / 200

    def get_resource_reward(self, state_last, state_now):
        mineral_reward = (state_now[3] - state_last[3]) / 10800
        gas_reward = (state_now[4] - state_last[4]) / 4500
        return mineral_reward + gas_reward

    def get_values(self, values):
        if self.is_end and self.result != 0:
            return 0
        else:
            return values

    def get_action_by_policy(self, obs):
        act, v_preds = self.net.policy.get_action(obs, verbose=False)
        return act, v_preds

    def fullfill_creature_technology(self, unit):
        unit_type = type(unit)
        if unit_type == T.Marine or unit_type == T.Reaper:
            if self.Barracks_num - self.busy_barracks_num > 0:
                return True
        elif unit_type == T.SCV:
            if T.SCV not in map(lambda x: type(x), self.units_in_production):
                return True
        return False

    def fullfill_building_technology(self, building):
        unit = building
        if type(unit) == T.Refinery:
            if self.CommandCenter_num > 0 and self.Refinery_num < self.CommandCenter_num * 2:
                return True
        elif type(unit) == T.SupplyDepot:
            return True
        elif type(unit) == T.Barracks:
            if self.CommandCenter_num > 0 and self.SupplyDepot_num > 0:
                return True
        return False

    def fullfill_creature_condition(self, unit):
        if self.mineral >= unit.mineral_price and self.gas >= unit.gas_price:
            if self.food_cap >= self.food_used + unit.food_used and self.fullfill_creature_technology(unit):
                return True
        return False

    def fullfill_building_condition(self, building):
        if self.mineral >= building.mineral_price and self.gas >= building.gas_price and self.mineral_worker_nums > 0 \
                and self.fullfill_building_technology(building):
            return True
        return False

    def build_from_barracks(self, unit):
        barracks = self.get_empty_barracks()
        if barracks is not None:
            barracks.queue.enqueue(unit)

    def build_from_commandcenter(self, unit):
        commandcenter = self.get_empty_commandcenter()
        if commandcenter is not None:
            commandcenter.queue.enqueue(unit)

    def update(self):
        self.update_resources()

        self.update_unit_progress()
        self.update_units()
        self.update_building_progress()
        self.update_buildings()

        self.update_supply()
        # self.update_larva()
        self.update_army()

        self.time_seconds += self.time_per_step

    def update_army(self):
        if self.military_num() == 0:
            self.env.army[self.player_id].order = Army.Order.NOTHING
            self.env.army[self.player_id].pos = self.pos

    def update_supply(self):
        self.food_used = 0
        self.food_used = sum(map(lambda x: x[0]().food_used * x[1], list(self.workers_list.items()) + list(self.army_list.items())))
        # self.food_used += sum(map(lambda x: x.food_used, self.units_in_production))

        self.food_cap = min(self.CommandCenter_num * 15 + self.SupplyDepot_num * 8, 200)

    def update_resources(self):
        mineral_income, gas_income = self.get_income()

        self.mineral += mineral_income
        self.collected_mineral += mineral_income
        self.gas += gas_income
        self.collected_gas += gas_income

    def get_income(self):
        mineral_income = self.mineral_worker_nums * 6 if self.mineral_worker_nums <= 16 else 16 * 6  # 5 -> 6 -> 7
        if self.mineral_reserves - mineral_income > 0:
            self.mineral_reserves -= mineral_income
        else:
            mineral_income = self.mineral_reserves
            self.mineral_reserves = 0

        max_gas_workers = self.Refinery_num * 3
        gas_income = self.gas_worker_nums * 4 if self.gas_worker_nums < max_gas_workers else max_gas_workers * 4
        if self.gas_reserves - gas_income >= 0:
            self.gas_reserves -= gas_income
        else:
            gas_income = self.gas_reserves
            self.gas_reserves = 0

        return mineral_income, gas_income

    def update_units(self):
        self.army_nums = sum(self.army_list.values())

        self.SCV_num = self.workers_list.get(T.SCV, 0)
        self.Marine_num = self.army_list.get(T.Marine, 0)
        self.Reaper_num = self.army_list.get(T.Reaper, 0)

    def update_buildings(self):
        self.building_nums = sum(self.building_list.values())

        self.CommandCenter_num = self.building_list.get(T.CommandCenter, 0)
        self.Refinery_num = self.building_list.get(T.Refinery, 0)
        self.SupplyDepot_num = self.building_list.get(T.SupplyDepot, 0)
        self.Barracks_num = self.building_list.get(T.Barracks, 0)

    def build_units(self, unit_type):
        unit = unit_type()
        # num = self.get_build_num(unit)
        num = 1
        if num == 0:
            return
        if not self.fullfill_creature_condition(unit):
            return

        self.units_in_production.append(unit)

        self.mineral -= unit.mineral_price * num
        self.gas -= unit.gas_price * num

        # self.build_from_barracks(unit)


    def update_unit_progress(self):
        for unit in self.units_in_production:
            unit.progress += self.time_per_step
            unit_type = type(unit)
            if unit.progress >= unit.build_time:
                self.units_in_production.remove(unit)
                if unit_type == T.SCV:
                    self.mineral_worker_nums += 1
                    self.add_unit(unit, 1, u_type='worker')
                elif unit_type == T.Marine:
                    self.add_unit(unit, 1, u_type='army')
                elif unit_type == T.Reaper:
                    self.add_unit(unit, 1, u_type='army')
        self.busy_barracks_num = len(list(filter(lambda x: type(x) == T.Marine or type(x) == T.Reaper, self.units_in_production)))


    def build_structures(self, building_type):
        building = building_type()
        if self.fullfill_building_condition(building):
            self.buildings_in_production.append(building)
            self.mineral -= building.mineral_price
            self.gas -= building.gas_price
            self.mineral_worker_nums -= 1

    def update_building_progress(self):
        for building in self.buildings_in_production:
            building.progress += self.time_per_step
            if building.progress >= building.build_time:
                self.buildings_in_production.remove(building)
                self.add_building(building)
                self.mineral_worker_nums += 1

    def get_empty_barracks(self):
        min_produce_len = 5
        min_building = None

        for b in self.production_building_list:
            if type(b) == type(T.Barracks()):
                produce_len = b.queue.size()
                if produce_len < min_produce_len:
                    min_produce_len = produce_len
                    min_building = b

        return min_building

    def get_empty_commandcenter(self):
        min_produce_len = 5
        min_building = None

        for b in self.production_building_list:
            if type(b) == type(T.CommandCenter()):
                produce_len = b.queue.size()
                if produce_len < min_produce_len:
                    min_produce_len = produce_len
                    min_building = b

        return min_building

    def step(self, action):
        if action == TerranAction.Build_SCV.value:
            self.build_units(T.SCV)

        elif action == TerranAction.Build_Refinery.value:
            self.build_structures(T.Refinery)

        elif action == TerranAction.Gather_gas.value:
            if self.mineral_worker_nums > 0 and self.Refinery_num > 0:
                self.mineral_worker_nums -= 1
                self.gas_worker_nums += 1

        elif action == TerranAction.Gather_mineral.value:
            if self.gas_worker_nums > 0:
                self.mineral_worker_nums += 1
                self.gas_worker_nums -= 1

        elif action == TerranAction.Build_Marine.value:
            self.build_units(T.Marine)

        elif action == TerranAction.Build_Reaper.value:
            self.build_units(T.Reaper)

        elif action == TerranAction.Build_Barracks.value:
            self.build_structures(T.Barracks)

        elif action == TerranAction.Build_SupplyDepot.value:
            self.build_structures(T.SupplyDepot)

        elif action == TerranAction.Attack.value:
            if self.military_num() > 0:
                self.env.army[self.player_id].order = Army.Order.ATTACK

        elif action == TerranAction.Defend.value:
            if self.military_num() > 0:
                self.env.army[self.player_id].order = Army.Order.DEFEND

        self.update()


class DummyTerran(A):

    def __init__(self, diff=5):
        super().__init__(self)
        self.army_nums = 4
        self.building_nums = 3
        self.difficulty = diff

    def step_auto(self, verbose=False):
        if self.env.all_steps % 8 == 1:
            if self.env.all_steps >= 48:
                self.add_unit(T.Marine(), 1)
            if self.env.all_steps >= 56 and self.difficulty >= 3:
                self.add_unit(T.Marine(), 2)
            if self.env.all_steps >= 48 and self.difficulty >= 5:
                self.add_unit(T.Marine(), 2)
            if self.env.all_steps >= 40 and self.difficulty >= 7:
                self.add_unit(T.Marine(), 2)
            if self.env.all_steps >= 32 and self.difficulty >= 9:
                self.add_unit(T.Marine(), 1)
            if self.env.all_steps >= 32 and self.difficulty >= 11:
                self.add_unit(T.Marine(), 1)
            if self.env.all_steps >= 32 and self.difficulty >= 13:
                self.add_unit(T.Marine(), 1)
            if self.env.all_steps >= 32 and self.difficulty >= 15:
                self.add_unit(T.Marine(), 1)
            if self.env.all_steps >= 32 and self.difficulty >= 17:
                self.add_unit(T.Marine(), 1)

        if self.military_num() > 40:
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
        self.army_nums = 6
        self.building_nums = 3

    def get_power(self):
        self.add_unit(T.Marine(), 4, u_type='army')
        self.add_building(T.CommandCenter(), 1)
        self.add_building(T.Barracks(), 1)
        self.add_building(T.SupplyDepot(), 1)
