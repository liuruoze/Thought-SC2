import lib.config as C
import lib.utils as U
import lib.transform_pos as T
import numpy as np
import torchcraft.Constants as tcc
import random

"""For the sake of simplicity, we give the macro-action directly and not provide the way on how to get them.
Also for simplicity, the information required for macro operations is extracted directly from the original interface."""


def get_closest(x, y, units):
    dist = float('inf')
    u = None
    for unit in units:
        d = (unit.x - x)**2 + (unit.y - y)**2
        if d < dist:
            dist = d
            u = unit
    return u


def scout_manager(agent):
    myunits = agent.obs.units[agent.obs.player_id]
    positions = agent.obs.start_locations[2:4]
    for p in positions:
        pass
        #print(p.x, p.y)
    #print(agent.obs.start_locations)
    #print(positions)

    base = agent.base

    detecter = None
    actions = []
    for unit in myunits:
        # note: unit.completd is important, otherwise it will chose the unit which are still building
        if unit.type == tcc.unittypes.Protoss_Probe and unit.completed and unit.gathering_minerals:
            detecter = unit

    closest = get_closest(base.x, base.y, positions)

    #print('Begin to scout!')
    if detecter is not None:
        actions.append([
            tcc.command_unit, detecter.id,
            tcc.unitcommandtypes.Right_Click_Position,
            -1, base.x, base.y,
        ])

        for pos in positions:
            if pos == closest:
                agent.retreat_pos = pos
                continue
            actions.append([
                tcc.command_unit, detecter.id,
                tcc.unitcommandtypes.Right_Click_Position,
                -1, pos.x, pos.y, 1,
            ])
            agent.enemy_pos = pos

        actions.append([
            tcc.command_unit, detecter.id,
            tcc.unitcommandtypes.Right_Click_Position,
            -1, base.x, base.y, 1,
        ])
        
    agent.safe_action(actions)
    #print('End scout!')

def worker_manager(agent):
    myunits = agent.obs.units[agent.obs.player_id]
    actions = []

    # collect resources
    num_gather_mineral = 0
    best_gather_mineral = 12

    num_gather_gas = 0
    best_gather_gas = 3

    # idle worker to collect mineral
    for unit in myunits:
        
        if unit.type == tcc.unittypes.Protoss_Gateway and unit.completed:
            actions.append([
                tcc.command_unit,
                unit.id,
                tcc.unitcommandtypes.Set_Rally_Position,
                0, agent.rally_pos[0], agent.rally_pos[1],

            ])
        if unit.type == tcc.unittypes.Protoss_Probe and unit.completed:
            if unit.gathering_minerals:
                num_gather_mineral += 1
            if unit.gathering_gas:
                num_gather_gas += 1
            if unit.idle:
                target = get_closest(unit.x, unit.y, agent.resourceUnits)
                actions.append([
                    tcc.command_unit,
                    unit.id,
                    tcc.unitcommandtypes.Right_Click_Unit,
                    target.id,
                ])

    # redunt worker go to collect gas
    if num_gather_mineral > best_gather_mineral and num_gather_gas < best_gather_gas:
        transit_gas_worker = None
        for unit in myunits:
            if unit.type == tcc.unittypes.Protoss_Probe and unit.completed:
                if unit.gathering_minerals:
                    transit_gas_worker = unit

        assimilator = None
        for unit in myunits:
            # note: unit.completd is important, otherwise it will chose the unit which are still building
            if unit.type == tcc.unittypes.Protoss_Assimilator and unit.completed:
                assimilator = unit

        if transit_gas_worker is not None and assimilator is not None:
            actions.append([
                tcc.command_unit,
                transit_gas_worker.id,
                tcc.unitcommandtypes.Right_Click_Unit,
                assimilator.id,
            ])

    agent.safe_action(actions)


def train_unit(agent, camp_type, unit_type):
    # select a free building and train
    camp = selectCamp(agent, camp_type)

    actions = []
    if camp is not None:
        actions = [[
            tcc.command_unit, camp.id,
            tcc.unitcommandtypes.Train,
            0, 0, 0,
            unit_type,
        ]]
    agent.safe_action(actions)


def build_by_worker(agent, builder_type, building_type, build_pos):
    builder = selectBuilder(agent, builder_type)

    actions = []
    if builder is not None:
        actions = [[
            tcc.command_unit, builder.id,
            tcc.unitcommandtypes.Build,
            -1, build_pos[0], build_pos[1],
            building_type,
        ]]

    agent.safe_action(actions)


def attack_step(agent, army_types, attack_pos):
    army = selectArmy(agent, army_types)

    actions = []
    if army is not None and attack_pos is not None:
        for solider in army:
            actions.append([
                tcc.command_unit, solider.id,
                tcc.unitcommandtypes.Attack_Move,
                -1, attack_pos.x, 
                attack_pos.y,
                1,
            ])
    agent.safe_action(actions)


def retreat_step(agent, army_types, retreat_pos):
    army = selectArmy(agent, army_types)

    actions = []
    if army is not None and retreat_pos is not None:
        for solider in army:
            actions.append([
                tcc.command_unit, solider.id,
                tcc.unitcommandtypes.Move,
                -1, retreat_pos.x, retreat_pos.y, 1,
            ])
    agent.safe_action(actions)


def no_op(agent):
    actions = []
    agent.safe_action(actions)


def selectArmy(agent, army_types):
    military = []
    myunits = agent.obs.units[agent.obs.player_id]
    for unit in myunits:
        for j in army_types:
            if unit.type == j and unit.completed:
                military.append(unit)

    num = len(military)
    if num > 0:
        return military
    return None


def selectBuilder(agent, builder_type):
    # random select a probe
    workers = []
    myunits = agent.obs.units[agent.obs.player_id]
    for unit in myunits:
        # note: unit.completd is important, otherwise it will chose the unit which are still building
        if unit.type == builder_type and unit.completed and unit.gathering_minerals and not unit.carrying_minerals:
            workers.append(unit)

    num = len(workers)
    if num > 0:
        rand = np.random.choice(num, size=1)
        #print('rand:', rand)
        builder = workers[rand[0]]
        return builder
    return None


def selectCamp(agent, camp_type):
    # random select a camp
    camps = []
    myunits = agent.obs.units[agent.obs.player_id]

    for unit in myunits:
        # note: if not unit.training, means that the building is free, and we choose that
        if unit.type == camp_type and unit.completed and not unit.training:
            camps.append(unit)

    num = len(camps)
    if num > 0:
        rand = np.random.choice(num, size=1)
        camp = camps[rand[0]]
        return camp
    return None
