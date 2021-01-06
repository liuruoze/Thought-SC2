from enum import Enum

import time
import tensorflow as tf
import numpy as np
import math
import lib.config as C
import lib.utils as U

from datetime import datetime
from mini_network import MiniNetwork
from strategy.protoss_agent import Protoss
from strategy.terran_agent import Terran
from strategy.agent import Dummy
from unit.units import Army
import unit.protoss_unit as P
import unit.terran_unit as T
import unit.zerg_unit as Z


class RaceforSC2(Enum):
    PROTOSS = 1
    ZERG = 2
    TERRAN = 3


class State():

    def __init__(self):
        self.time_seconds = 0
        self.mineral_worker_nums = 12
        self.gas_worker_nums = 0
        self.mineral = 50
        self.gas = 0
        self.food_cup = 14
        self.food_used = 12
        self.ranger_nums = 0
        self.melee_nums = 0
        self.enemy_ranger_nums = 0
        self.enemy_melee_nums = 0
        self.building_nums = 1
        self.enemy_building_nums = 1
        self.defender_nums = 0
        self.enemy_defender_nums = 0

    def encoder(self):
        feature = np.array([self.time_seconds, self.mineral_worker_nums, self.gas_worker_nums, self.mineral, ...])
        return feature

    def transition(self, action):
        pass

    def logistics(self, action):
        pass

    def combat(self, action):
        pass


class StrategyState():

    def __init__(self):
        pass


class BattleField():

    def __init__(self, red_agent=None, blue_agent=None, field=None, max_steps=100):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.field = field
        self.max_steps = max_steps

    def get_damage(self, army):
        attack = 0
        damage = 0
        for key, value in army.items():
            unit_type = key
            unit = unit_type()
            attack += unit.attack * value
            damage += unit.dps * (1 + unit.range / 15) * value

        if attack != 0:
            attack /= sum(army.values())

        return attack, damage

    def get_equivalent_hp(self, army, attack):
        all_hp = 0
        for key, value in army.items():
            unit_type = key
            unit = unit_type()
            all_hp += unit.getEquivalentHP(attack) * value
        return all_hp

    def battle(self, verbose=False):

        red_army = self.red_agent.military_force()
        blue_army = self.blue_agent.military_force()

        if verbose:
            print('red_army:', red_army)
            print('blue_army:', blue_army)

        #red_all_hp, red_damage, red_armor = self.all_power(red_army)
        #blue_all_hp, blue_damage, blue_armor = self.all_power(blue_army)

        red_attack, red_damage = self.get_damage(red_army)
        blue_attack, blue_damage = self.get_damage(blue_army)

        red_all_hp = self.get_equivalent_hp(red_army, blue_attack)
        blue_all_hp = self.get_equivalent_hp(blue_army, red_attack)

        #red_remain_hp = min(red_all_hp - blue_damage + red_armor, red_all_hp)
        red_remain_hp = red_all_hp - blue_damage
        if verbose:
            print(red_remain_hp)

        #blue_remain_hp = min(blue_all_hp - red_damage + blue_armor, blue_all_hp)
        blue_remain_hp = blue_all_hp - red_damage
        if verbose:
            print(blue_remain_hp)

        # print(self.field)
        # print(self.red_agent.pos)
        # print(self.blue_agent.pos)

        if blue_remain_hp < 0 and self.field == self.blue_agent.pos:

            self.blue_agent.under_attack(-blue_remain_hp)

        if red_remain_hp < 0 and self.field == self.red_agent.pos:

            self.red_agent.under_attack(-red_remain_hp)

        if verbose:
            print(self.red_agent.building_hp())
            print(self.blue_agent.building_hp())
            

        self.red_agent.reset_military(red_remain_hp)
        self.blue_agent.reset_military(blue_remain_hp)

        red_army = self.red_agent.military_force()
        blue_army = self.blue_agent.military_force()

        if verbose:
            print('red_army:', red_army)
            print('blue_army:', blue_army)

class ObStateforSC2():

    def __init__(self):
        self.time_seconds = 0
        self.red_player_economy = 50
        self.blue_player_economy = 50
        self.red_player_food = 12
        self.blue_player_food = 12
        self.red_player_buildings = 1
        self.blue_player_buildings = 1
        self.red_player_race = RaceforSC2.PROTOSS
        self.blue_player_race = RaceforSC2.TERRAN
        self.red_army_pos = 0
        self.blue_army_pos = 0
        self.red_state = None
        self.blue_state = None


class SimulatePlatform():

    def __init__(self, red_agent=None, blue_agent=None, distance=5, max_steps=1):
        self.red_agent = red_agent
        self.blue_agent = blue_agent

        self.distance = distance
        self.red_pos = 0
        self.blue_pos = distance - 1
        self.max_steps = max_steps

        self.army = [Army(0), Army(1)]
        self.win_index = -1
        self.all_steps = 0
        self.is_end = False

    def reset(self):
        self.army = [Army(0), Army(1)]
        self.win_index = -1
        self.all_steps = 0
        self.is_end = False

        self.red_agent.reset(self.red_pos)
        self.blue_agent.reset(self.blue_pos)

    def init(self):
        self.red_agent.init(self, player_id=0, pos=self.red_pos)
        self.blue_agent.init(self, player_id=1, pos=self.blue_pos)

    def simulate(self, verbose=False):
        # print(self.army[0].pos)
        # print(self.army[1].pos)
        # print(self.red_agent.pos)
        # print(self.blue_agent.pos)

        for i in range(self.max_steps):
            self.red_agent.step_auto(False)
            self.blue_agent.step_auto(False)

            if verbose:
                print('Red army:', self.red_agent.military_force())
                print('Red order:', self.army[self.red_agent.player_id])
                print('Blue army:', self.blue_agent.military_force())
                print('Blue order:', self.army[self.blue_agent.player_id])
                time.sleep(0.3)

            if self.is_end == True:
                if verbose:
                    print('step:', i, ' game ends.')
                    print('winner is:', self.win_index)
                break

            self.battle_execute()
            self.all_steps += 1

            if self.blue_agent.building_hp() < 0:
                self.win_index = self.red_agent.player_id
                self.is_end = True
            elif self.red_agent.building_hp() < 0:
                self.win_index = self.blue_agent.player_id
                self.is_end = True
            elif i == self.max_steps - 2:
                self.is_end = True

    def battle_execute(self):
        army_0 = self.army[0]
        army_1 = self.army[1]
        combat_max_steps = 1
        # print('army_0.pos', army_0.pos)
        # print('army_1.pos', army_1.pos)
        if army_0.pos == army_1.pos:
            bf = BattleField(red_agent=self.red_agent, blue_agent=self.blue_agent, field=army_0.pos)
            for i in range(combat_max_steps):
                bf.battle(False)
        else:
            if army_0.order == Army.Order.ATTACK:
                army_0.pos = min(army_0.pos + 1, self.distance - 1)
            elif army_0.order == Army.Order.DEFEND:
                army_0.pos = max(army_0.pos - 1, 0)

            if army_1.order == Army.Order.ATTACK:
                army_1.pos = max(army_1.pos - 1, 0)
            elif army_1.order == Army.Order.DEFEND:
                army_1.pos = min(army_1.pos + 1, self.distance - 1)



def test():
    red_agent = Protoss()

    blue_agent = Dummy()
    blue_agent.add_unit(T.Marine(), 0)
    blue_agent.add_building(T.Commandcenter(), 1)
    blue_agent.add_building(T.Supplydepot(), 3)
    blue_agent.add_building(T.Barracks(), 1)

    env = SimulatePlatform(red_agent=red_agent, blue_agent=blue_agent,
                           distance=5, max_steps=100)
    env.init()
    env.simulate()


def net_test():
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
    )
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    now = datetime.now()
    model_path = "./model/" + now.strftime("%Y%m%d-%H%M%S") + "_mini/"
    mini_net = MiniNetwork(sess, ppo_load_path=model_path, ppo_save_path=model_path)
    mini_net.initialize()

    red_agent = Protoss()
    red_agent.set_net(mini_net)

    blue_agent = Dummy()
    blue_agent.add_unit(T.Marine(), 5)
    blue_agent.add_building(T.Commandcenter(), 1)
    blue_agent.add_building(T.Supplydepot(), 3)
    blue_agent.add_building(T.Barracks(), 1)

    env = SimulatePlatform(red_agent=red_agent, blue_agent=blue_agent,
                           distance=5, max_steps=100)
    env.init()
    # env.simulate()
    red_agent.play_with_rl()


def battle_test():
    red_agent = Protoss()
    blue_agent = Terran()
    red_agent.add_unit(P.Zealot(), 12)
    blue_agent.add_unit(T.Marine(), 1)
    blue_agent.add_building(T.Commandcenter(), 1)
    blue_agent.add_building(T.Supplydepot(), 5)
    blue_agent.add_building(T.Barracks(), 2)
    bf = BattleField(red_agent=red_agent, blue_agent=blue_agent)
    for i in range(10):
        bf.battle()


if __name__ == "__main__":
    net_test()
