from data_provider import data_generator
import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset_dir', '/path/to/your/training/set', '')

dataset_dir = FLAGS.dataset_dir
img_dir = "{}/JPEGImages/".format(dataset_dir)
gt_dir = "{}/Annotations/".format(dataset_dir)
geo_dir = "{}/geo_map".format(dataset_dir)
score_dir = "{}/score_map".format(dataset_dir)

if not os.path.exists(geo_dir):
    os.makedirs(geo_dir)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
data_generator.generator_all(input_images_dir=img_dir, input_gt_dir=gt_dir, output_geo_dir=geo_dir, output_score_dir= score_dir)
