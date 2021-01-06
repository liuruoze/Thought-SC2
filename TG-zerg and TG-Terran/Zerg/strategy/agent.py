from enum import Enum

import numpy as np
import math
import lib.config as C
import lib.utils as U
from unit.units import Army
from lib.replay_buffer import Buffer
import unit.terran_unit as T
import unit.zerg_unit as Z
import unit.protoss_unit as P


class StrategyforSC2(Enum):
    RUSH = 1
    ECONOMY = 2
    DEFENDER = 3


class Agent:

    def __init__(self, agent_id=0, global_buffer=None, net=None, restore_model=False):
        self.env = None
        self.mpc = None
        self.net = net

        self.agent_id = agent_id
        self.player_id = 0

        self.global_buffer = global_buffer
        self.restore_model = restore_model

        self.local_buffer = Buffer()
        self.restart_game()

    def reset(self, pos):
        self.set_pos(pos)
        self.set_army()
        
        self.local_buffer.reset()
        self.restart_game()

    def restart_game(self):
        self.is_end = False
        self._result = 0

        self.time_seconds = 0
        self.mineral_worker_nums = 12
        self.gas_worker_nums = 0
        self.mineral = 50
        self.gas = 0
        self.food_cap = 14
        self.food_used = 12
        self.army_nums = 0
        self.enemy_army_nums = 0
        self.building_nums = 1
        self.enemy_building_nums = 1
        self.defender_nums = 0
        self.enemy_defender_nums = 0
        self.strategy = StrategyforSC2.RUSH
        self.enemy_strategy = StrategyforSC2.ECONOMY
        self.workers_list = {}
        self.army_list = {}
        self.building_list = {}
        self.remain_buildings_hp = 0

        self.time_per_step = 9

    def obs():
        return None

    def init(self, env, player_id, pos):
        self.set_env(env)
        self.set_id(player_id)
        self.set_pos(pos)
        self.set_army()

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

    def set_buffer(self, global_buffer):
        self.global_buffer = global_buffer

    def set_env(self, env):
        self.env = env

    def set_net(self, net):
        self.net = net

    def set_mpc(self, mpc):
        self.mpc = mpc

    def set_id(self, player_id):
        self.player_id = player_id

    def set_pos(self, pos):
        self.pos = pos

    def set_army(self):
        army = Army(self.player_id)
        army.pos = self.pos
        self.env.army[self.player_id] = army

    def add_unit(self, unit, num=1, u_type='army'):
        unit_type = type(unit)
        if u_type == 'worker':
            if unit_type in self.workers_list:
                self.workers_list[unit_type] += num
            else:
                self.workers_list[unit_type] = num
        elif u_type == 'army':
            if unit_type in self.army_list:
                self.army_list[unit_type] += num
            else:
                self.army_list[unit_type] = num

    def add_building(self, building, num=1):
        building_type = type(building)
        hp = building.hp
        self.remain_buildings_hp += hp * num
        if building_type in self.building_list.keys():
            self.building_list[building_type] += num
        else:
            self.building_list[building_type] = num

    def building_hp(self):
        return self.remain_buildings_hp

    def under_attack(self, attack_hp):
        self.remain_buildings_hp -= attack_hp

    def military_force(self):
        return self.army_list

    def military_num(self):
        return sum(self.army_list.values())

    def reset_military(self, remain_hp):
        all_hp = 0
        remain_creatures_list = {}
        for key, value in self.army_list.items():
            unit_type = key
            number = value
            unit = unit_type()
            count = 0
            for _ in range(number):
                all_hp += unit.hp
                if all_hp <= remain_hp:
                    count += 1
                else:
                    break
            remain_creatures_list[unit_type] = count
            if all_hp >= remain_hp:
                break
        # print(remain_hp)
        # print(self.army_list)
        # print(remain_creatures_list)
        self.army_list = remain_creatures_list


class Dummy(Agent):

    def __init__(self):
        Agent.__init__(self)
        self.army_nums = 5
        self.building_nums = 10

    def step(self, action):
        pass

    def get_power(self):
        self.add_unit(Z.Zergling(), 20, u_type='army')
        # self.add_unit(Z.Roach(), 4)
        self.add_building(Z.Hatchery(), 2)
        self.add_building(Z.SpawningPool(), 1)
        self.add_building(Z.RoachWarren(), 1)
