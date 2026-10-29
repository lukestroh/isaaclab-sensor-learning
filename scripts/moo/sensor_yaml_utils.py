#!/usr/bin/env python3
import glob
import os
import pprint as pp

sensor_cfg_dir = "/home/luke/IsaacLab/source/isaaclab-sensor-learning/source/isaaclab_sensor_learning/isaaclab_sensor_learning/config/sensors"


def get_all_sensors(sensor_cfg_dir: str) -> list[str]:
    all_sensors = glob.glob(os.path.join(sensor_cfg_dir, "**/*.yaml"))
    return all_sensors


def get_sensor_map_from_sensors(sensors: list[str]) -> dict[int, str]:
    sensor_map = {}
    for i in range(len(sensors)):
        sensor_map[i] = sensors[i]
    return sensor_map


def generate_sensor_map(sensor_cfg_dir: str) -> dict[int, str]:
    all_sensors = get_all_sensors(sensor_cfg_dir=sensor_cfg_dir)
    return get_sensor_map_from_sensors(sensors=all_sensors)


def main():
    sensor_map = generate_sensor_map(sensor_cfg_dir=sensor_cfg_dir)
    pp.pprint(sensor_map)
    return


if __name__ == "__main__":
    main()
