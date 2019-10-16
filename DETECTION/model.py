import os
import sys
sys.path.append('lib')
sys.path.append('lib/slim')

import numpy as np
import progressbar
import tensorflow as tf

# train
from object_detection import model_hparams
from object_detection import model_lib

# test
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields

from DataLoader import DataLoader

from lib.image_uitls import visualize_bbxs, bbxs_image, center_image
from lib.ops import non_max_suppression_fast


class JNet(object):
    def __init__(self, conf):

        self.conf = conf

        self.batch_size = conf.batch_size
        self.skip_edge = conf.skip_edge
        if conf.input_shape is None:
            self.input_shape = (None, None, None, 3)
        else:
            self.input_shape = conf.input_shape

        self.input = None
        self.outputs = None

        if conf.mode == 'test':
            self.build_graph()

    def build_graph(self):
        # read pipeline config
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.gfile.GFile(self.conf.pipeline_config_path, 'r') as f:
            text_format.Merge(f.read(), pipeline_config)
        text_format.Merge('', pipeline_config)

        # check to make sure
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = self.conf.height
        pipeline_config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = self.conf.width

        # pipeline_config.model.faster_rcnn.first_stage_max_proposals = self.conf.max_proposal
        # pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_detections_per_class = self.conf.max_proposal
        # pipeline_config.model.faster_rcnn.second_stage_post_processing.batch_non_max_suppression.max_total_detections = self.conf.max_proposal

        if self.conf.mode == 'test':
            detection_model = model_builder.build(pipeline_config.model, is_training=False)
            self.build_test_graph(detection_model)

        self.saver = tf.train.Saver()


    def build_test_graph(self, detection_model):
        self.input = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
        preprocessed_inputs, true_image_shapes = detection_model.preprocess(self.input)
        output_tensors = detection_model.predict(preprocessed_inputs, true_image_shapes)
        postprocessed_tensors = detection_model.postprocess(output_tensors, true_image_shapes)

        detection_fields = fields.DetectionResultFields
        label_id_offset = 1
        boxes = postprocessed_tensors.get(detection_fields.detection_boxes)
        scores = postprocessed_tensors.get(detection_fields.detection_scores)
        classes = postprocessed_tensors.get(
            detection_fields.detection_classes) + label_id_offset
        masks = postprocessed_tensors.get(detection_fields.detection_masks)
        num_detections = postprocessed_tensors.get(detection_fields.num_detections)
        outputs = {}
        outputs[detection_fields.detection_boxes] = tf.identity(
            boxes, name=detection_fields.detection_boxes)
        outputs[detection_fields.detection_scores] = tf.identity(
            scores, name=detection_fields.detection_scores)
        outputs[detection_fields.detection_classes] = tf.identity(
            classes, name=detection_fields.detection_classes)
        outputs[detection_fields.num_detections] = tf.identity(
            num_detections, name=detection_fields.num_detections)
        if masks is not None:
            outputs[detection_fields.detection_masks] = tf.identity(
                masks, name=detection_fields.detection_masks)
        for output_key in outputs:
            tf.add_to_collection('inference_op', outputs[output_key])

        self.outputs = outputs

    def safe_run(self, sess, feed_dict=None, output_tensor=None):

        try:
            out_dict = sess.run(output_tensor, feed_dict=feed_dict)
        except Exception as e:
            if type(e).__name__ == 'ResourceExhaustedError':
                print('Ran out of memory !')
                print('decrease the batch size')
                sys.exit(-1)
            else:
                print('Error in running session:')
                print(e.message)
                sys.exit(-1)

        return out_dict

    def test(self, data):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.saver.restore(sess, self.conf.trained_checkpoint)

            crop_gen = data.next_crop()
            iterate = True
            bbxs = []
            scores = []
            while iterate:

                # make np arrays for generator
                batch_x = np.empty(shape=(self.batch_size, data.height, data.width, 3))
                corner = np.empty(shape=(self.batch_size, 2))

                try:
                    for i in range(self.batch_size):
                        corner[i, :], batch_x[i, :, :, :] = next(crop_gen)
                except StopIteration:
                    iterate = False

                # temp (for 16 bit image)
                #TODO: check dtype
                if data.image.dtype == 'uint16':
                    batch_x = batch_x / 256

                out_org = self.safe_run(sess, feed_dict={self.input: batch_x}, output_tensor=self.outputs)

                # crop augmentation
                if self.conf.crop_augmentation == 'rot90':
                    out_aug = self.safe_run(sess, feed_dict={self.input: tf.image.rot90(batch_x).eval()},
                                                 output_tensor=self.outputs)
                    out_aug['detection_boxes'][:, :, [0, 2]] = np.subtract(1, out_aug['detection_boxes'][:, :, [0, 2]])
                    out_aug['detection_boxes'] = out_aug['detection_boxes'][:, :, [1, 0, 3, 2]]

                elif self.conf.crop_augmentation == 'flip_left_right':
                    out_aug = self.safe_run(sess, feed_dict={self.input: tf.image.flip_left_right(batch_x).eval()},
                                            output_tensor=self.outputs)
                    out_aug['detection_boxes'][:, :, [1, 3]] = np.subtract(1, out_aug['detection_boxes'][:, :, [1, 3]])

                if self.conf.crop_augmentation is not None:
                    out_dict = {}
                    out_dict.update({'detection_boxes': np.concatenate((out_org['detection_boxes'],
                                                                        out_aug['detection_boxes']),
                                                                       axis=1)})
                    out_dict.update({'detection_scores': np.concatenate((out_org['detection_scores'],
                                                                         out_aug['detection_scores']),
                                                                        axis=1)})
                    out_dict.update({'detection_classes': np.concatenate((out_org['detection_classes'],
                                                                          out_aug['detection_classes']),
                                                                         axis=1)})
                else:
                    out_dict = out_org

                for i in range(self.batch_size):
                    keep_boxes = out_dict["detection_scores"][i, :] > self.conf.score_threshold

                    if not np.any(keep_boxes):
                        continue

                    box = out_dict["detection_boxes"][i, :][keep_boxes]
                    score = out_dict["detection_scores"][i, :][keep_boxes]

                    box = box[:, [1, 0, 3, 2]]      # reformat to: xmin, ymin, xmax, ymax
                    # rescale from [0-1] to the crop size
                    box[:, [0, 2]] = box[:, [0, 2]] * data.width
                    box[:, [1, 3]] = box[:, [1, 3]] * data.height

                    # remove very large bounding boxes
                    idx = np.where((box[:, 2] - box[:, 0] < 100) | (box[:, 3] - box[:, 1] < 100), True, False)
                    box = box[idx, :]
                    score = score[idx]
                    if not np.any(idx):    # if no bounding box after removing large ones
                        continue

                    # keep boxes inside the crop (not close to the edge)
                    idx = np.where((box[:, 0] >= self.skip_edge) &
                                   (box[:, 1] >= self.skip_edge) &
                                   (box[:, 2] <= data.width - self.skip_edge) &
                                   (box[:, 3] <= data.height - self.skip_edge), True, False)
                    box = box[idx, :]
                    score = score[idx]
                    if not np.any(idx):     # if no bounding box after removing edge ones
                        continue

                    #TODO: tensorflow version is slow... come up with alternative
                    # # non-max-suppression
                    # idx = tf.image.non_max_suppression(np.array(box), score, len(box), iou_threshold=0.5).eval()
                    # box = box[idx, :]
                    # score = score[idx]


                    # from skimage import exposure
                    # show_image = exposure.rescale_intensity((batch_x[i, :, :, :]).astype('uint8'), in_range='image', out_range='dtype')
                    # visualize_bbxs(show_image, bbxs=box, save=True)

                    box[:, [0, 2]] += corner[i][0]
                    box[:, [1, 3]] += corner[i][1]

                    bbxs.append(box.astype(int))
                    scores.append(score)

        # np.save(os.path.join(self.conf.output_dir, 'boxes.npy'), np.array(bbxs))
        # np.save(os.path.join(self.conf.output_dir, 'scores.npy'), np.array(scores))

        data.bbxs = np.concatenate(bbxs)
        data.scores = np.concatenate(scores)

        keep_idx = data.remove_close_centers(centers=data.centers, scores=data.scores, radius=3)
        data.bbxs = data.bbxs[keep_idx, :]
        data.scores = data.scores[keep_idx]

        keep_idx = data.nms(data.bbxs, overlapThresh=self.conf.nms_iou)
        data.bbxs = data.bbxs[keep_idx, :]
        data.scores = data.scores[keep_idx]

        # create output directory
        if not os.path.exists(self.conf.output_dir):
            os.makedirs(self.conf.output_dir)

        # save results in output_dir
        np.save(os.path.join(self.conf.output_dir, 'boxes.npy'), data.bbxs)
        np.save(os.path.join(self.conf.output_dir, 'scores.npy'), data.scores)

        data.save_bbxs(os.path.join(self.conf.output_dir, 'bbxs_detection.txt'))
        bbxs_image(os.path.join(self.conf.output_dir, 'bbxs_detection.tif'), data.bbxs, data.image.shape[:2][::-1])
        center_image(os.path.join(self.conf.output_dir, 'centers_detection.tif'), data.centers, data.image.shape[:2][::-1])

    def train(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        config = tf.estimator.RunConfig(model_dir=self.conf.model_dir)

        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=config,
            hparams=model_hparams.create_hparams(None),
            pipeline_config_path=self.conf.pipeline_config_path,
            train_steps=None,
            eval_steps=None)
        estimator = train_and_eval_dict['estimator']
        train_input_fn = train_and_eval_dict['train_input_fn']
        eval_input_fn = train_and_eval_dict['eval_input_fn']
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_steps = train_and_eval_dict['train_steps']
        eval_steps = train_and_eval_dict['eval_steps']

        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fn,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_steps,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

    def eval(self):

        tf.logging.set_verbosity(tf.logging.INFO)

        config = tf.estimator.RunConfig(model_dir=self.conf.model_dir)

        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=config,
            hparams=model_hparams.create_hparams(None),
            pipeline_config_path=self.conf.pipeline_config_path,
            train_steps=None,
            eval_steps=None)
        estimator = train_and_eval_dict['estimator']
        eval_input_fn = train_and_eval_dict['eval_input_fn']
        eval_steps = train_and_eval_dict['eval_steps']

        name = 'validation_data'
        input_fn = eval_input_fn

        estimator.evaluate(input_fn, eval_steps, checkpoint_path=tf.train.latest_checkpoint(self.conf.model_dir))
