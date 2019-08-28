import tensorflow as tf
import time

flags = tf.app.flags
flags.DEFINE_string('mode', 'test', 'train | test | eval | write_crops | update_xmls | create_tfrecord')

# load model
flags.DEFINE_string('pipeline_config_path', 'models/dapi/pipeline_config.config', 'Path to detection config file')


# train
flags.DEFINE_string('model_dir', 'training/NeuN', 'Path to output model directory '
                                                  'where event and checkpoint files will be written.')

# test
flags.DEFINE_string('trained_checkpoint', 'models/dapi/model.ckpt-63896', 'Path to trained checkpoint')
flags.DEFINE_integer('batch_size', 2, 'training batch size')
flags.DEFINE_integer('max_proposal', 500, 'maximum proposal per image')
flags.DEFINE_float('score_threshold', .4, 'Threshold of score of detection box')
flags.DEFINE_integer('skip_edge', 10, 'skip object with this distance to edge')
flags.DEFINE_string('crop_augmentation', None, 'flip_left_right | rot90 | None')
flags.DEFINE_string('input_shape', None, 'Comma delimited input shape e.g 300,300,3')
# post processing
flags.DEFINE_float('nms_iou', .6, 'intersection over union of bbxs for non max suppression')
flags.DEFINE_integer('close_centers_r', 5, 'Minimum distance between two centers')

# data
flags.DEFINE_string('input_dir', r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L', 'Path to the directory of input data')
flags.DEFINE_string('output_dir', r'E:\jahandar\DashData\TBI\G3_BR#14_HC_11L\detection_results', 'Path to the directory to save results')
flags.DEFINE_integer('channel', 1, 'Network input channel size')
flags.DEFINE_string('c1', 'R2C1.tif', 'image 1 path')
flags.DEFINE_string('c2', 'R2C1.tif', 'image 3 path')
flags.DEFINE_string('c3', 'R2C3.tif', 'image 2 path')
flags.DEFINE_string('bbxs_file', None, 'txt file name of bounding boxes')
flags.DEFINE_string('centers_file', None, 'txt file name of centers')
flags.DEFINE_integer('height', 300, 'Network input height size - crop large image with this height')
flags.DEFINE_integer('width', 300, 'Network input width size - crop large image with this height')
flags.DEFINE_integer('depth', None, 'Network input depth size (in the case of 3D input images)')
flags.DEFINE_integer('overlap', 100, 'Overlap of crops')

# write_crops parameters => write crops with specified size from large image
flags.DEFINE_string('save_folder', 'data/test/hpc_crop', 'Parent folder of imgs & xmls folders')
flags.DEFINE_integer('crop_width', None, 'Crop large image with this width   | use "width" if None')
flags.DEFINE_integer('crop_height', None, 'Crop large image with this height | use "height" if None')
flags.DEFINE_integer('crop_overlap', None, 'Overlap between crops (in pixel) | use "overlap" if None')
flags.DEFINE_boolean('adjust_hist', True, 'Adjust histogram of crop for higher quality image')

# update_xmls parameters => updates the objects in bbxs.txt with new objects
flags.DEFINE_string('xmls_dir', 'data/test/hpc_crop/xmls', 'Parent folder of imgs & xmls folders')
flags.DEFINE_string('new_bbxs', 'updated_bbxs.txt', 'Save new bounding boxes filename')



args = tf.app.flags.FLAGS