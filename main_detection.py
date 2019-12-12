import os
import sys
import time
import zipfile
import argparse
import requests

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--INPUT_DIR', type=str, default=r'/path/to/input/dir', help='/path/to/input/dir')
parser.add_argument('--OUTPUT_DIR', type=str, default=r'/path/to/output/dir', help='/path/to/output/dir')
parser.add_argument('--DAPI', type=str, default='', help='/path/to/dapi.tif | None')
parser.add_argument('--HISTONES', type=str, default='', help='/path/to/dapi.tif | None')

args = parser.parse_known_args()[0]

channels = {'DAPI': args.DAPI, 'Histones': args.HISTONES}

# Add object_detection and slim folders to PATH
sys.path.append(os.path.join(os.getcwd(), 'DETECTION'))
sys.path.append(os.path.join(os.getcwd(), 'DETECTION', 'lib'))
sys.path.append(os.path.join(os.getcwd(), 'DETECTION', 'lib', 'slim'))


# DOWNLOAD PRE-TRAINED MODELS
def download_and_extract_models(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination + '.zip')

    with zipfile.ZipFile(destination + '.zip') as file:
        par_dir = os.path.dirname(destination)
        [file.extract(filename, par_dir) for filename in file.namelist()]
    os.remove(destination + '.zip')


file_id = '1xvkkOij38YCO2_tVm5qAs4Xup1zoWrwC'
target_dir = 'DETECTION/models'
if not os.path.exists(target_dir):
    download_and_extract_models(file_id, target_dir)

# DETECTION
num_channels = len([k for k, v in channels.items() if v])
model = '_'.join([k for k, v in channels.items() if v]).lower()
config_path = os.path.join('DETECTION', 'models', model, 'pipeline_config.config')
trained_checkpoint = os.path.join('DETECTION', 'models', model, 'model.ckpt')
from DETECTION.config import args as tf_args

tf_args.pipeline_config_path = config_path
tf_args.trained_checkpoint = trained_checkpoint
tf_args.input_dir = args.INPUT_DIR
tf_args.output_dir = args.OUTPUT_DIR
tf_args.channel = num_channels
tf_args.c1 = os.path.join(args.INPUT_DIR, channels['DAPI'])
tf_args.c2 = os.path.join(args.INPUT_DIR, channels['Histones'])
from DETECTION.model import JNet as detection_model
from DETECTION.DataLoader import DataLoader
start = time.time()
data = DataLoader(tf_args)
model = detection_model(tf_args)
model.test(data)
duration = time.time() - start
m, s = divmod(int(duration), 60)
h, m = divmod(m, 60)
print('Detection pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
