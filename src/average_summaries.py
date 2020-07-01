import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins.hparams import plugin_data_pb2
from collections import defaultdict


FOLDER_NAME = 'aggregates'


def decode_hparam_bytes(hparam_bytes):
    hparams_data = plugin_data_pb2.HParamsPluginData.FromString(hparam_bytes)
    return hparams_data.session_start_info.hparams


def extract_steps_data_hparams(path):
    print(path)
    serialized_examples = tf.data.TFRecordDataset(str(path))
    data = defaultdict(list)
    steps = defaultdict(list)
    hparams = None
    for serialized_example in serialized_examples: # for each steps / tag
        event = Event.FromString(serialized_example.numpy())
        for value in event.summary.value: # I don't understand this iteration
            if value.tag != "_hparams_/session_start_info":
                steps[value.tag].append(event.step)
                data[value.tag].append(tf.make_ndarray(value.tensor))
            else:
                if hparams is None:
                    hparams = value.metadata.plugin_data.content
    steps = {key: np.stack(values, axis=0) for key, values in steps.items()}
    data = {key: np.stack(values, axis=0) for key, values in data.items()}
    return steps, data, hparams


def steps_equal(steps1, steps2):
    for key, value1 in steps1.items():
        value2 = steps2[key]
        if value1.shape != value2.shape:
            return False
        if not (value1 == value2).all():
            return False
    return True


def check_all_steps_the_same(all_steps):
    try:
        all_steps = iter(all_steps)
        first = next(all_steps)
        return all(steps_equal(first, rest) for rest in all_steps)
    except StopIteration:
        return True


def average_steps_data(all_steps, all_data):
    check_all_steps_the_same(all_steps)
    return all_steps[0], {
        key: np.mean([data[key] for data in all_data], axis=0)
        for key in all_data[0].keys()
    }


def get_event_files_path(multirun_path):
    multirun_path = Path(multirun_path)
    logdirs = [
        multirun_path / x / 'logs' for x in os.listdir(multirun_path)
        if os.path.isdir(multirun_path / x)
        and os.path.isdir(multirun_path / x / "logs")
    ]
    paths = []
    for logdir in logdirs:
        files = [
            f for f in os.listdir(logdir) if f.startswith("events.out.tfevents")
        ]
        for f in files:
            paths.append(logdir / f)
    return paths


def aggregate_to_summary(path, steps, averages, hparam_bytes, name):
    path = Path(path) / FOLDER_NAME / name
    path.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(path))
    print(str(path))
    hparams = decode_hparam_bytes(hparam_bytes)
    with writer.as_default():
        hparams = {key: value.ListFields()[0][1] for key, value in hparams.items()}
        hp.hparams(hparams)
        for key, means in averages.items():
            for step, mean in zip(steps[key], means):
                tf.summary.scalar(key, mean, step=step)
        writer.flush()


def group_by_hparams(all_steps, all_data, all_hparams):
    all_hparams = list(all_hparams)
    all_hparams_clear = [decode_hparam_bytes(h) for h in all_hparams]
    dones = [False for _ in all_hparams]
    for i, done_i in enumerate(dones):
        if not done_i:
            for j, done_j in enumerate(dones[i + 1:], i + 1):
                if all_hparams_clear[i] == all_hparams_clear[j]:
                    all_hparams[j] = all_hparams[i]
                    dones[j] = True
        dones[i] = True
    hparams_dict = defaultdict(list)
    for hparams, steps, data in zip(all_hparams, all_steps, all_data):
        hparams_dict[hparams].append((steps, data))
    return hparams_dict


def get_name_mapping(hparam_bytes_list):
    hparam_list = [decode_hparam_bytes(bytes) for bytes in hparam_bytes_list]
    different_keys = set()
    for i, hparam_i in enumerate(hparam_list):
        for j, hparam_j in enumerate(hparam_list[i + 1:], i + 1):
            for key, value in hparam_i.items():
                if hparam_j[key] != value:
                    different_keys.add(key)
    name_mapping = {}
    for bytes, hparam in zip(hparam_bytes_list, hparam_list):
        name = "__".join([
            key + '_' + str(hparam[key].ListFields()[0][1])
            for key in different_keys
        ])
        name_mapping[bytes] = name
    return name_mapping


if __name__ == '__main__':
    import sys


    # multirun_path = "/home/cwilmot/sshfs_mountpoint/experiments/2020-06-29/19-16-46/"
    multirun_path = sys.argv[1]
    paths = get_event_files_path(multirun_path)
    all_steps, all_data, all_hparams = tuple(zip(*[
        extract_steps_data_hparams(path) for path in paths
    ]))
    hparams_to_steps_data = group_by_hparams(all_steps, all_data, all_hparams)
    hparams_to_name = get_name_mapping(all_hparams)

    for hparams, all_steps_data in hparams_to_steps_data.items():
        all_steps, all_data = tuple(zip(*all_steps_data))
        steps, averages = average_steps_data(all_steps, all_data)
        name = hparams_to_name[hparams]
        aggregate_to_summary(multirun_path, steps, averages, hparams, name)
