import os
import time
import subprocess
import requests
import zipfile

INPUT_DIR = r'/brazos/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#10_HC_12L/unsupervised/unmixed'
OUTPUT_DIR = r'/brazos/roysam/datasets/TBI/G3_mFPI_Vehicle/G3_BR#10_HC_12L/unsupervised/detection_results'
channels = {'DAPI': 'R2C1.tif', 'Histones': ''}

# Add object_detection and slim folders to PYTHONPATH
if os.environ.get("PYTHONPATH") is None:
    os.environ["PYTHONPATH"] = ''
os.environ["PYTHONPATH"] += os.pathsep + os.path.join(os.getcwd(), '2_DETECTION', 'lib')
os.environ["PYTHONPATH"] += os.pathsep + os.path.join(os.getcwd(), '2_DETECTION', 'lib', 'slim')

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
target_dir = '2_DETECTION/models'
if not os.path.exists(target_dir):
    download_and_extract_models(file_id, target_dir)

# DETECTION
num_channels = len([k for k, v in channels.items() if v])
model = '_'.join([k for k, v in channels.items() if v]).lower()
config_path = os.path.join('2_DETECTION/models', model, 'pipeline_config.config')
trained_checkpoint = os.path.join('2_DETECTION/models', model, 'model.ckpt')
command = ' '.join([r"python 2_DETECTION/main.py",
                    "--mode=test",
                    "--pipeline_config_path={}".format(config_path),
                    "--trained_checkpoint={}".format(trained_checkpoint),
                    "--input_dir={}".format(INPUT_DIR),
                    "--output_dir={}".format(OUTPUT_DIR),
                    "--channel={}".format(num_channels),
                    "--c1={}".format(os.path.join(INPUT_DIR, channels['DAPI'])),
                    "--c2={}".format(os.path.join(INPUT_DIR, channels['Histones']))])
start = time.time()
p = subprocess.call(command, shell=True)
print('Detection pipeline finished successfully in {:.2f} seconds.'.format(time.time() - start))
