import argparse
import torchcraft as tc
import torchcraft.Constants as tcc
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage


from mini_agent import ProtossAction

parser = argparse.ArgumentParser(
    description='Plays simple micro battles with an attack closest heuristic')
parser.add_argument('-t', '--hostname', type=str,
                    help='Hostname where SC is running')
parser.add_argument('-p', '--port', default=11111,
                    help="Port to use")
parser.add_argument('-d', '--debug', default=0, type=int, help="Debug level")

args = parser.parse_args()

DEBUG = args.debug


def dprint(msg, level):
    if DEBUG > level:
        print(msg)


def get_dist(x, y, unit):
    d = (unit.x - x)**2 + (unit.y - y)**2
    return d


def get_state_features(state):
    features = []

    return features


skip_frames = 48
check_frames = 51
total_battles = 0
enemy_x = 228
enemy_y = 24
subbase_x = 236
subbase_y = 430

while total_battles < 100:
    print("CTRL-C to stop")
    nloop = 1

    cl = tc.Client()
    cl.connect(args.hostname, args.port)
    state = cl.init(micro_battles=False)
    for pid, player in state.player_info.items():
        print("player {} named {} is {}".format(player.id, player.name,
                                                tc.Constants.races._dict[player.race]))

    # Initial setup
    cl.send([
        [tcc.set_speed, 0],
        [tcc.set_gui, 1],
        [tcc.set_cmd_optim, 1],
        [tcc.set_frameskip, 0],
    ])

    for pos in state.start_locations:
        print(pos.x, pos.y)

    macro_action_list = [0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 1]

    base = None
    initial_base = False

    resourceUnits = []
    vespeneUnits = []
    initial_resourceUnits = False

    initial_detect = False
    initial_workers = False
    initial_all = False
    initial_gas_collecter = False

    num_pylon = 0
    num_gateway = 0
    num_cyber = 0

    pylon_size = 8
    gateway_size = 16
    cyber_size = 16

    max_pylon = 12
    max_gateway = 6
    max_cyber = 3

    while not state.game_ended:
        nloop += 1
        state = cl.recv()
        actions = []
        frame_no = state.frame_from_bwapi
        # print(frame_no)
        myunits = state.units[state.player_id]
        if frame_no == 0:
            if not initial_all:
                # initial base
                if not initial_base:
                    for unit in myunits:
                        if unit.type == tcc.unittypes.Protoss_Nexus:
                            base = unit
                    if base is not None:
                        initial_base = True

                # initial resourceUnits
                if not initial_resourceUnits:
                    neutralUnits = state.units[state.neutral_id]
                    for u in neutralUnits:
                        if u.type == tcc.unittypes.Resource_Mineral_Field or u.type == tcc.unittypes.Resource_Mineral_Field_Type_2 \
                                or u.type == tcc.unittypes.Resource_Mineral_Field_Type_3:
                            if u.visible:
                                resourceUnits.append(u)
                        if u.type == tcc.unittypes.Resource_Vespene_Geyser:
                            if u.visible:
                                vespeneUnits.append(u)
                    if len(resourceUnits) > 0:
                        initial_resourceUnits = True
                    print('resourceUnits:', len(resourceUnits))
                    print('vespeneUnits:', len(vespeneUnits))

                if not initial_workers:
                    for unit in myunits:
                        if unit.type == tcc.unittypes.Protoss_Probe and unit.idle:
                            if unit.idle:
                                target = get_closest(unit.x, unit.y, resourceUnits)
                                actions.append([
                                    tcc.command_unit,
                                    unit.id,
                                    tcc.unitcommandtypes.Right_Click_Unit,
                                    target.id,
                                ])
                    initial_workers = True

        elif frame_no > 0 and frame_no % skip_frames == 0:

            macro_action = random.randint(0, 9)

            if macro_action == 0:
                if state.frame.resources[state.player_id].ore >= 75:
                    print('ProtossAction.Build_probe')
                    actions = [[
                        tcc.command_unit,
                        base.id,
                        tcc.unitcommandtypes.Train,
                        0,
                        0,
                        0,
                        tcc.unittypes.Protoss_Probe,
                    ]]

            elif macro_action == 1:  # Build_pylon
                builder = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Probe and unit.completed and unit.gathering_minerals:
                        builder = unit

                if builder is not None:
                    if state.frame.resources[state.player_id].ore >= 150:
                        if base is not None:
                            if not initial_detect:
                                actions = begin_to_detect(state.start_locations, base, myunits)
                                initial_detect = True
                            else:
                                print('ProtossAction.Build_pylon')
                                building_size = 8
                                initial_polyon_x = base.x + pylon_size * int(max_pylon * 0.6)
                                initial_polyon_y = base.y - 6 - pylon_size - 3
                                print(initial_polyon_x, initial_polyon_y)
                                target_x = initial_polyon_x - num_pylon * pylon_size
                                target_y = initial_polyon_y
                                print(target_x, target_y)

                                if target_x != -1 and target_y != -1:
                                    actions = [[
                                        tcc.command_unit,
                                        builder.id,
                                        tcc.unitcommandtypes.Build,
                                        -1,
                                        target_x,
                                        target_y,
                                        tcc.unittypes.Protoss_Pylon,
                                    ]]
                                    num_pylon = (num_pylon + 1) % max_pylon
            elif macro_action == 2:  # Build_Gateway
                builder = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Probe and unit.completed and unit.gathering_minerals:
                        builder = unit

                pylon = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Pylon and unit.completed:
                        pylon = unit

                if pylon is not None and builder is not None:
                    if state.frame.resources[state.player_id].ore >= 200:
                        print('ProtossAction.Build_Gateway')
                        base_size = 6
                        initial_gateway_x = base.x + gateway_size * int(max_gateway * 0.6)
                        initial_gateway_y = base.y - 6 - pylon_size - gateway_size - 3
                        print(initial_gateway_x, initial_gateway_y)
                        target_x = initial_gateway_x - num_gateway * gateway_size
                        target_y = initial_gateway_y
                        print(target_x, target_y)

                        if target_x != -1 and target_y != -1:
                            actions = [[
                                tcc.command_unit,
                                builder.id,
                                tcc.unitcommandtypes.Build,
                                -1,
                                target_x,
                                target_y,
                                tcc.unittypes.Protoss_Gateway,
                            ]]
                            num_gateway = (num_gateway + 1) % max_gateway

            elif macro_action == 3:  # train zealot
                pass

            elif macro_action == 4:  # Protoss_Cybernetics_Core:
                builder = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Probe and unit.completed and unit.gathering_minerals:
                        builder = unit

                pylon = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Pylon and unit.completed:
                        pylon = unit

                if pylon is not None and builder is not None:
                    if state.frame.resources[state.player_id].ore >= 250:
                        print('ProtossAction.Protoss_Cybernetics_Core')
                        base_size = 6
                        initial_cyber_x = base.x + cyber_size * max_cyber
                        initial_cyber_y = base.y - 6
                        print(initial_cyber_x, initial_cyber_y)
                        target_x = initial_cyber_x - num_cyber * cyber_size
                        target_y = initial_cyber_y
                        print(target_x, target_y)

                        if target_x != -1 and target_y != -1:
                            actions = [[
                                tcc.command_unit,
                                builder.id,
                                tcc.unitcommandtypes.Build,
                                -1,
                                target_x,
                                target_y,
                                tcc.unittypes.Protoss_Cybernetics_Core,
                            ]]
                            num_cyber = (num_cyber + 1) % max_cyber

            elif macro_action == 5:  # Protoss_Assimilator:
                builder = None
                for unit in myunits:
                    # note: unit.completd is important, otherwise it will chose the unit which are still building
                    if unit.type == tcc.unittypes.Protoss_Probe and unit.completed and unit.gathering_minerals:
                        builder = unit

                if builder is not None:
                    if state.frame.resources[state.player_id].ore >= 125:
                        print('ProtossAction.Protoss_Assimilator')
                        if len(vespeneUnits) > 0:
                            vespene = vespeneUnits[0]
                            #print('base.x:', base.x)
                            #print('base.y:', base.y)
                            #print('vespene.x:', vespene.x)
                            #print('vespene.y:', vespene.y)
                            if vespene is not None:
                                actions = [[
                                    tcc.command_unit,
                                    builder.id,
                                    tcc.unitcommandtypes.Build,
                                    -1,
                                    vespene.x - 8,
                                    base.y - 6,
                                    tcc.unittypes.Protoss_Assimilator,
                                ]]

            elif macro_action == 6:  # train dragoon
                pass

            elif macro_action == 7:  # attack
                military = []
                for unit in myunits:
                    if unit.type == tcc.unittypes.Protoss_Zealot or unit.type == tcc.unittypes.Protoss_Dragoon:
                        if unit.completed:
                            military.append(unit)

                if len(military) > 0:
                    print('ProtossAction.Attack')
                    for solider in military:
                        actions.append([
                            tcc.command_unit,
                            solider.id,
                            tcc.unitcommandtypes.Attack_Move,
                            -1,
                            enemy_x,
                            enemy_y,
                        ])
            elif macro_action == 8:  # retreat
                military = []
                for unit in myunits:
                    if unit.type == tcc.unittypes.Protoss_Zealot or unit.type == tcc.unittypes.Protoss_Dragoon:
                        if unit.completed:
                            military.append(unit)

                if len(military) > 0:
                    print('ProtossAction.Retreat')
                    for solider in military:
                        actions.append([
                            tcc.command_unit,
                            solider.id,
                            tcc.unitcommandtypes.Move,
                            -1,
                            subbase_x,
                            subbase_y,
                        ])
            else:  # do nothing
                pass

        elif frame_no > 0 and frame_no % check_frames == 0:
            actions = worker_manager(myunits)
        else:
            pass

        if len(actions) > 0:
            dprint("Sending actions: " + str(actions), -1)
        cl.send(actions)
    cl.close()
