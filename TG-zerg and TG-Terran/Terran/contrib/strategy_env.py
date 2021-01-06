import numpy as np

import lib.config as C
import lib.utils as U

from unit.zerg_unit import *
from unit.terran_unit import *
from unit.protoss_unit import *

RACES = ['P', 'T', 'Z', 'Dummy']

class StrategyEnv(object):
    
    def __init__(self, agent_race='Z', bot_race='Dummy'):
        self.red_player = Player(race=agent_race)
        self.blue_player = Player(race=bot_race)

        self.steps_per_episode = 250

    def reset(self):
        pass
    
    def restart(self):
        pass

class Player(object):

    def __init__(self, race='Z'):
        self.MAX_SUPPLY = 200

        self.race = race
        self.minerals = 0
        self.gas = 0

        self.bases = []
        self.building_list = {}
        self.unit_list = {}

        self.workers = 0
        self.supply_units = 0
        self.supply = 0
        self.supply_used = 0

    def reset(self):
        self.minerals = 0
        self.gas = 0
        self.bases = [Base().setup()]
        self.workers = bases[0].mineral_workers
        # self.supply =
         
    def step(self, actions):
        self.workers = sum([base.mineral_workers for base in bases])

        


    def get_supply(self):
        self.supply = sum([base.supply for base in bases])
        self.supply = self.supply_unit_num * 8
        self.supply = self.supply if self.supply <= self.MAX_SUPPLY else self.MAX_SUPPLY
    
    def get_supply_used(self):
        self.supply_used = self.workers
        for unit, num in self.unit_list:
            self.supply_used += unit().food_used * num

    def get_income(self):
        income = [base.income() for base in bases]
        mineral_income = sum(map(lambda x: x[0], income))
        gas_income = sum(map(lambda x: x[1], income))
        return mineral_income, gas_income
    
    def add_unit(self, unit, num):
        unit_type = type(unit)
        if unit_type in self.unit_list.keys():
            self.unit_list[unit_type] += num
        else:
            self.unit_list[unit_type] = num

    def add_building(self, building):
        building_type = type(building)
        if building_type in self.building_list.keys():
            self.building_list[building_type] += 1
        else:
            self.building_list[building_type] = 1


class Base(object):
    
    def __init__(self, race='Z'):
        self.MAX_MINERAL_WORKERS = 16
        self.MAX_GAS_WORKERS_PER_GEYSER = 3
        self.MINERAL_RESERVES = 10800
        self.GAS_RESERVES = 4500
        self.mineral_workers = 0
        self.gas_workers = 0
        self.gas_collectors = 0

        self.level = 1

        assert race in RACES
        if race == 'Z':
            self.base_instance = Hatchery()
        elif race == 'T':
            self.base_instance = CommandCenter()
        elif race == 'P':
            self.base_instance = Nexus()

        self.hp = self.base_instance.hp
        self.supply = self.base_instance.food_supply
    
    def setup(self, is_subbase=False):
        if is_subbase:
            self.mineral_workers = 0
        else:
            self.mineral_workers = 12
        
    def get_income(self):
        mineral_income = self.mineral_workers * 5 if self.mineral_workers < self.MAX_MINERAL_WORKERS else 16 * 5
        if self.MINERAL_RESERVES - mineral_income >= 0:
            self.MINERAL_RESERVES -= mineral_income
        else:
            mineral_income = self.MINERAL_RESERVES
            self.MINERAL_RESERVES = 0        

        max_gas_workers = self.gas_collectors * self.MAX_GAS_WORKERS_PER_GEYSER
        gas_income = self.gas_workers * 4 if self.gas_workers < max_gas_workers else max_gas_workers * 4
        # have some problems on gathering gas from different geysers
        if self.GAS_RESERVES - gas_income >= 0:
            self.GAS_RESERVES -= gas_income
        else:
            gas_income = self.GAS_RESERVES
            self.GAS_RESERVES = 0

        return mineral_income, gas_income

class Army(object):
    
    def __init__(self):
        self.hp


if __name__ == '__main__':
    pass