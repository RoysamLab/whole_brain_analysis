import os
import tensorflow as tf
from config import args
from utils import write_spec


if args.model == 'original_capsule':
    from models.Original_CapsNet import Orig_CapsNet as Model
elif args.model == 'matrix_capsule':
    from models.Matrix_Capsule_EM_routing import MatrixCapsNet as Model
elif args.model == 'vector_capsule':
    from models.Deep_CapsNet import CapsNet as Model


def main(_):
    if args.mode not in ['train', 'test']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train or test")
    else:
        model = Model(tf.Session(), args)

        if args.mode == 'train':
            # Write specification/network architecture in file
            write_spec(args)
            # Create necessary directories to save logs and model parameters
            if not os.path.exists(os.path.join(args.modeldir, args.run_name)):
                os.makedirs(os.path.join(args.modeldir, args.run_name))
            if not os.path.exists(os.path.join(args.logdir, args.run_name)):
                os.makedirs(os.path.join(args.logdir, args.run_name))
            # Create dataset for training from bounding boxes
            from prepare_data.prepare_data_train import main as perpare_data
            perpare_data(args.INPUT_DIR, args.BBXS_FILE,
                         [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2, args.IBA1, args.RECA1],
                         args.OUTPUT_DIR, inside_box=[8000, 4000, 34000, 24000], parallel=False, margin=5,
                         crop_size=(50, 50), topN=5000)
            # Train model
            model.train()
        elif args.mode == 'test':
            if not os.path.exists(args.OUTPUT_DIR):
                os.makedirs(args.OUTPUT_DIR)
            # Create dataset for training from bounding boxes if it is the first time for inference
            if args.test_mode == 'first':
                from prepare_data.prepare_data_test import main as perpare_data
                perpare_data(args.INPUT_DIR, args.BBXS_FILE,
                             [args.DAPI, args.HISTONES, args.NEUN, args.S100, args.OLIG2, args.IBA1, args.RECA1],
                             args.OUTPUT_DIR, margin=5, crop_size=(50, 50), parallel=True)
            model.inference(args.step_num)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
    tf.app.run()
