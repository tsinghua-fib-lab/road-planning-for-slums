import yaml
import glob
import pickle
import os
import socket


def get_file_path(file_path):
    hostname = socket.gethostname()
    cwd = os.getcwd()
    if hostname == 'LAPTOP-V8NBPG91':
        file_path = os.path.join(cwd, file_path)
    elif hostname == 'rl2':
        file_path = os.path.join('/data1/suhongyuan/road_planning/code', file_path)
    elif hostname == 'rl3':
        file_path = os.path.join('/home/zhengyu/workspace/urban_planning', file_path)
    elif hostname == 'rl4':
        file_path = os.path.join('/home/zhengyu/workspace/urban_planning', file_path)
    elif hostname == 'DL4':
        file_path = os.path.join('/data2/zhengyu/workspace/urban_planning', file_path)
    else:
        raise ValueError('Unknown hostname: {}'.format(socket.gethostname()))
    return file_path


def load_yaml(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    print(file_path)
    assert(len(files) == 1)
    cfg = yaml.safe_load(open(files[0], 'r'))
    return cfg


def load_pickle(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    assert(len(files) == 1)
    data = pickle.load(open(files[0], 'rb'))
    return data
