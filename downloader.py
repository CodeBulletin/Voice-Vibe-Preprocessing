import configparser
import os

env = configparser.ConfigParser()
env.read('.env')

os.environ['KAGGLE_USERNAME'] = env['Login']['kaggle_username']
os.environ['KAGGLE_KEY'] = env['Login']['kaggle_key']


import kaggle

config = configparser.ConfigParser()
config.read('settings.ini')

kaggle.api.authenticate()

datasets = config['Datasets']['datasets'].split(',')
dataset_names = config['Datasets']['datasets_names'].split(',')
download_loc = config['Main']['dataset_download_folder']

if not os.path.exists(download_loc):
    os.mkdir(download_loc)


print(datasets)

for dataset, name in zip(datasets, dataset_names):
    if not os.path.exists(download_loc + "/" + name):
        os.mkdir(download_loc + "/" + name)
        kaggle.api.dataset_download_files(
            dataset,
            path=download_loc + "/" + name,
            unzip=True,
            quiet=False
        )