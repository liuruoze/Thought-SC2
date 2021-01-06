import os
import csv
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm

from absl import app
from absl import flags

from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

REPLAY_PATH = "/home/data3/liuruoze/3.16.1-Pack_2/replays/"
COPY_PATH = "/home/data3/ghf/filtered_replays/"
SAVE_PATH = "./result.csv"

RACE = ['Terran', 'Zerg', 'Protoss', 'Random']
RESULT = ['Victory', 'Defeat', 'Tie']

def check_info(replay_info):
    map_name = replay_info.map_name
    player1_race = replay_info.player_info[0].player_info.race_actual
    player2_race = replay_info.player_info[1].player_info.race_actual

    if map_name == "Catalyst LE" and player1_race == 3 and player2_race == 3:
        return True

    if map_name == "Abyssal Reef LE" and player1_race == 2 and player2_race == 2:
        return True

    return False

def store_info(replay_info):
    map_name = replay_info.map_name
    player1_race = RACE[replay_info.player_info[0].player_info.race_requested - 1]
    player2_race = RACE[replay_info.player_info[1].player_info.race_requested - 1]
    game_duration_loops = replay_info.game_duration_loops
    game_duration_seconds = replay_info.game_duration_seconds
    game_version = replay_info.game_version
    game_result = RESULT[replay_info.player_info[0].player_result.result - 1]
    return [map_name,
            game_version,
            game_result,
            player1_race,
            player2_race,
            game_duration_loops,
            game_duration_seconds]

def main(argv):
    run_config = run_configs.get()
    replay_files = os.listdir(REPLAY_PATH)

    result = []
    map_set = set()
    
    with run_config.start() as controller:
        for replay_file in tqdm(replay_files):
            try:
                replay_path = REPLAY_PATH + replay_file
                replay_data = run_config.replay_data(replay_path)
                replay_info = controller.replay_info(replay_data)
            except Exception:
                continue
            map_set.add(replay_info.map_name)
            if check_info(replay_info):
                result.append([replay_file] + store_info(replay_info))
                shutil.copy(replay_path, COPY_PATH)

    df = pd.DataFrame(result, columns=['Replay File', 'Map Name', 'Game Version', 'Game Result', 'Player1 Race', 'Player2 Race', 'Game Loops', 'Game Duration'])
    df.to_csv(path_or_buf=SAVE_PATH)

    print(map_set)

if __name__ == '__main__':
    app.run(main)