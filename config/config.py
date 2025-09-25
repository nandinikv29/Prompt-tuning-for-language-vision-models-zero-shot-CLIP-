import os

import yaml


def get_config():
    file_yaml = 'config.yaml'
    rf = open(file=file_yaml, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()  # 关闭文件
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)

    cwd = os.getcwd()

    if cwd.startswith('/content/drive/MyDrive/'):
        yaml_data['dataset_root'] = './datas/'
    else:
        yaml_data['dataset_root'] = '/Users/lizx/python/datasets/'

    return yaml_data
