import os
import yaml

def get_config():
    file_yaml = 'config.yaml'
    with open(file_yaml, 'r', encoding='utf-8') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    cwd = os.getcwd()
    if cwd.startswith('/content/drive/MyDrive/'):
        yaml_data['dataset_root'] = './datas/'
    else:
        yaml_data['dataset_root'] = '/Users/lizx/python/datasets/'

    # If API key is blank, try environment variable
    if not yaml_data['openai'].get('api_key'):
        yaml_data['openai']['api_key'] = os.getenv("OPENAI_API_KEY", "")

    return yaml_data
