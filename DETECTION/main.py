import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from config import args
from model import JNet
from DataLoader import DataLoader


def main(_):
    if args.mode not in ['train', 'test', 'eval', 'write_crops', 'update_xmls', 'create_tfrecord']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: {}".format(x for x in ['train', 'test', 'write_crops', 'update_xmls']))
    else:

        data = DataLoader(args)

        if args.mode in ['train', 'test', 'eval']:
            model = JNet(args)
            if args.mode == 'train':
                model.train()
            elif args.mode == 'eval':
                model.eval()
            elif args.mode == 'test':
                model.test(data)

        elif args.mode == 'write_crops':
            data.write_crops(args.save_folder, args.crop_width, args.crop_height, args.crop_overlap,
                             adjust_hist=args.adjust_hist)

        elif args.mode == 'update_xmls':
            if args.xml_mode == 'update':
                data.update_xmls(xml_dir=args.xmls_dir, save_fname=args.new_bbxs)
            elif args.xml_mode == 'new':
                data.generate_new_table_from_xmls(xml_dir=args.xmls_dir, save_fname=args.new_bbxs)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    tf.app.run()
