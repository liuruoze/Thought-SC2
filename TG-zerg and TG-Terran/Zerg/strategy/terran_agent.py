from enum import Enum

import numpy as np
import math
import lib.config as C
import lib.utils as U
from strategy.agent import Agent as A
import unit.terran_unit as T


class TerranAction(Enum):
    Do_nothing = 0
    Build_scv = 1
    Build_marine = 2
    Build_supplydepot = 3
    Build_barracks = 4
    Attack = 5
    Build_sub_base = 6
    Build_bunker = 7
    All = 8


class Terran(A):

    def __init__(self):
        A.__init__(self)
        self.barracks_num = 0
        self.supplydepot_num = 0

        self.marine_num = 0
        self.MAX_ACTIONS = TerranAction.All.value

    def state(self):
        return str(self.time_seconds) + ', ' + str(self.mineral) + \
            ', ' + str(self.mineral_worker_nums) + ', ' + str(self.marine_num) + ', ' + str(self.food_cap)

    def step(self, state):
        self.action = self.get_policy_action(state)
        self.play(self.action)
        print('state', self.state())
        print('action', self.action)

    def get_policy_action(self, state):
        random = np.random.randint(self.MAX_ACTIONS)
        action = random
        return action

    def fullfill_technology(self, unit):
        return True

    def fullfill_creature_condition(self, unit):
        if self.mineral >= unit.mineral_price and self.gas >= unit.gas_price:
            if self.food_cap >= self.food_used + unit.food_used and self.fullfill_technology(unit):
                return True
        else:
            return False

    def win(self):
        if self.marine_num >= 30:
            return True
        else:
            return False

    def play(self, action):
        if action == TerranAction.Build_scv.value:
            if self.mineral >= 50 and self.food_used < self.food_cap:
                self.mineral_worker_nums += 1
                self.food_used += 1
                self.mineral -= 50
        elif action == TerranAction.Build_marine.value:
            marine = T.Marine()
            if self.fullfill_creature_condition(marine):
                self.army_nums += 1
                self.marine_num += 1
                self.food_used += marine.food_used
                self.mineral -= marine.mineral_price
        elif action == TerranAction.Build_supplydepot.value:
            if self.mineral >= 100:
                self.building_nums += 1
                self.food_cap += 8
                self.supplydepot_num += 1
                self.mineral -= 100
        elif action == TerranAction.Build_barracks.value:
            if self.mineral >= 150:
                self.barracks_num += 1
                self.building_nums += 1
                self.mineral -= 150
        elif action == TerranAction.Attack.value:
            pass
            '''if self.army_nums >= 2 * self.enemy_army_nums:
                self.enemy_building_nums -= 1
                self.army_nums -= 0.5 * self.enemy_army_nums
            elif self.army_nums < 0.5 * self.enemy_army_nums:
                self.army_nums = 0
            else:
                self.army_nums *= 0.5'''
        elif action == TerranAction.Build_sub_base.value:
            pass
        elif action == TerranAction.Build_bunker.value:
            pass

        self.mineral += self.mineral_worker_nums * 3
        self.time_seconds += 5
