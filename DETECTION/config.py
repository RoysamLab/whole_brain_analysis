import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='test', help='train | test | eval | write_crops | update_xmls | create_tfrecord')

# load model
parser.add_argument('--pipeline_config_path', type=str, default='models/dapi/pipeline_config.config', help='Path to detection config file')

# train
parser.add_argument('--model_dir', type=str, default='training/NeuN', help='Path to output model directory where event and checkpoint files will be written.')

# test
parser.add_argument('--trained_checkpoint', type=str, default='models/dapi/model.ckpt', help='Path to trained checkpoint')
parser.add_argument('--batch_size',type=int, default=2, help='training batch size')
parser.add_argument('--max_proposal', type=int, default=500, help='maximum proposal per image')
parser.add_argument('--score_threshold', type=int, default=.4, help='Threshold of score of detection box')
parser.add_argument('--skip_edge', type=int, default=10, help='skip object with this distance to edge')
parser.add_argument('--crop_augmentation', type=str, default=None, help='flip_left_right | rot90 | None')
parser.add_argument('--input_shape', type=str, default=None, help='Comma delimited input shape e.g 300,300,3')
# post processing
parser.add_argument('--nms_iou', type=int, default=.6, help='intersection over union of bbxs for non max suppression')
parser.add_argument('--close_centers_r', type=int, default=5, help='Minimum distance between two centers')

# data
parser.add_argument('--input_dir', type=str, default=r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L', help='Path to the directory of input data')
parser.add_argument('--output_dir', type=str, default=r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L\detection_results', help='Path to the directory to save results')
parser.add_argument('--channel', type=int, default=1, help='Network input channel size')
parser.add_argument('--c1', type=str, default='R2C1.tif', help='image 1 path')
parser.add_argument('--c2', type=str, default='R2C1.tif', help='image 2 path')
parser.add_argument('--c3', type=str, default='R2C3.tif', help='image 3 path')
parser.add_argument('--bbxs_file', type=str, default=None, help='txt file name of bounding boxes')
parser.add_argument('--centers_file', type=str, default=None, help='txt file name of centers')
parser.add_argument('--height', type=int, default=300, help='Network input height size - crop large image with this height')
parser.add_argument('--width', type=int, default=300, help='Network input width size - crop large image with this height')
parser.add_argument('--depth', type=int, default=None, help='Network input depth size (in the case of 3D input images)')
parser.add_argument('--overlap', type=int, default=100, help='Overlap of crops')

# write_crops parameters => write crops with specified size from large image
parser.add_argument('--save_folder', type=str, default='data/test/hpc_crop', help='Parent folder of imgs & xmls folders')
parser.add_argument('--crop_width', type=int, default=None, help='Crop large image with this width   | use "width" if None')
parser.add_argument('--crop_height', type=int, default=None, help='Crop large image with this height | use "height" if None')
parser.add_argument('--crop_overlap', type=int, default=None, help='Overlap between crops (in pixel) | use "overlap" if None')
parser.add_argument('--adjust_hist', type=str, default=True, help='Adjust histogram of crop for higher quality image')

# update_xmls parameters => updates the objects in bbxs.txt with new objects
parser.add_argument('--xmls_dir', type=str, default='data/test/hpc_crop/xmls', help='Parent folder of imgs & xmls folders')
parser.add_argument('--new_bbxs', type=str, default='updated_bbxs.txt', help='Save new bounding boxes filename')



args, _ = parser.parse_known_args()
