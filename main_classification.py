import os
import sys
import time
import zipfile
import argparse
import requests
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--INPUT_DIR', type=str, help='/path/to/input/dir')
    parser.add_argument('--OUTPUT_DIR', type=str, help='/path/to/output/dir')
    parser.add_argument('--BBXS_FILE', type=str, help='/path/to/bbxs_detection.txt')
    parser.add_argument('--test_mode', type=str, default='first', help='first | adjust -> generate or use y.h5 for inference')
    parser.add_argument('--DAPI', type=str, default='', help='<dapi.tif> | None')
    parser.add_argument('--HISTONES', type=str, default='', help='<histones.tif> | None')
    parser.add_argument('--NEUN', type=str, default='', help='<NeuN.tif> | None')
    parser.add_argument('--S100', type=str, default='', help='<S100.tif> | None')
    parser.add_argument('--OLIG2', type=str, default='', help='<Olig2.tif> | None')
    parser.add_argument('--IBA1', type=str, default='', help='<Iba1.tif> | None')
    parser.add_argument('--RECA1', type=str, default='', help='<RECA1.tif> | None')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.5, .5, .5, .5, .5], help='[NeuN_thres, S100_thresh, Olig2_thresh, Iba1_thresh, RECA1_thresh]')

    args = parser.parse_known_args()[0]

    sys.path.append(os.path.join(os.getcwd(), 'CLASSIFICATION'))
    os.chdir(os.path.join(os.getcwd(), 'CLASSIFICATION'))

    # DOWNLOAD PRE-TRAINED MODEL
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

        response = session.get(URL, params={'id': id}, stream=True, verify=False)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True, verify=False)

        save_response_content(response, os.path.join(destination, 'model.zip'))

        with zipfile.ZipFile(os.path.join(destination, 'model.zip')) as file:
            [file.extract(filename, destination) for filename in file.namelist()]
        os.remove(os.path.join(destination, 'model.zip'))

    file_id = '13NzdlcbGowMQt_hWSpzZwU5QZCbQypc0'
    target_dir = os.path.join(os.getcwd(), 'Results', 'model_dir')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        download_and_extract_models(file_id, target_dir)

    start = time.time()
    # PREPARE DATASET FOR CLASSIFICATION
    if args.test_mode == 'first':
        from CLASSIFICATION.prepare_data.prepare_data_test import main as perpare_data
        perpare_data(args.INPUT_DIR, args.BBXS_FILE,
                     [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2, args.IBA1, args.RECA1],
                     args.OUTPUT_DIR, margin=5, crop_size=(50, 50), parallel=True)

    # CLASSIFICATION
    from CLASSIFICATION.models.Original_CapsNet import Orig_CapsNet as Model
    from CLASSIFICATION.config import args as tf_args

    if not os.path.exists(args.OUTPUT_DIR):
        os.makedirs(args.OUTPUT_DIR)

    model = Model(tf.Session(), tf_args)
    model.inference(tf_args.step_num)
    duration = time.time() - start
    m, s = divmod(int(duration), 60)
    h, m = divmod(m, 60)
    print('Classification pipeline finished successfully in {:d} hours, {:d} minutes and {:d} seconds.'.format(h, m, s))
