# ScenarioNet

We have added the conversion source code for using the Unimm framework, which enables training trajectory prediction by jointly processing sensor data (e.g., camera images, LiDAR point clouds).

- Unimm: [Unimm]

## Installation

Please refer to the following website to download the Argoverse2 sensor dataset.

- Argoverse 2: [Argoverse 2 Dataset Download Guide](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data)

Please refer to the following repository for installation instructions.

- ScenarioNet repository: [ScenarioNet GitHub](https://github.com/metadriverse/scenarionet)

The dataset converted using our code can be used with Unitraj.

- UniTraj repository: [UniTraj GitHub](https://github.com/vita-epfl/UniTraj)

## Getting Started

To convert the Argoverse 2 sensor dataset into the Unimm(Unitraj) format, use the following command:
```bash
python convert_argoverse2_sensor_dataset.py -d /path/to/output --raw_data_path /path/to/av2_sensor
```

Replace /path/to/output and /path/to/av2_sensor with the actual paths where you want to save the converted data and where the Argoverse 2 sensor dataset is located.
