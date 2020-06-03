import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
# from localization_test import detect_train

tf.app.flags.DEFINE_integer('class_num', 230, '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100001, '')
tf.app.flags.DEFINE_integer('max_restore_steps', 40, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint as pretrain model')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2500, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_integer('train_stage', 1,
                            '0-train detection only; 1-train recognition only; 2-train end-to-end; 3-train end-to-end absolutely')
tf.app.flags.DEFINE_string('training_data_dir',
                           default='/path/to/your/training/set/JPEGImages/',
                           help='training images dir')
tf.app.flags.DEFINE_string('training_gt_data_dir',
                           default='/path/to/your/training/set/Annotations/',
                           help='training gt dir')
# tf.app.flags.DEFINE_string('training_data_dir', default='D:/FOTS/FOTS_TF-dev/ICDAR2015/ch4_training_images/', help='training images dir')
# tf.app.flags.DEFINE_string('training_gt_data_dir', default='D:/FOTS/FOTS_TF-dev/ICDAR2015/ch4_training_gt/', help='training gt dir')

from data_provider import data_generator
from module import Backbone_branch, RoI_rotate, classification_branch

FLAGS = tf.app.flags.FLAGS

# gpus = list(range(len(FLAGS.gpu_list.split(','))))

detect_part = Backbone_branch.Backbone(is_training=True)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = classification_branch.Recognition(is_training=False)


def build_graph(input_images, input_transform_matrix, input_box_masks, input_box_widths, input_seq_len):
    shared_feature, f_score, f_geometry = detect_part.model(input_images)
    # if FLAGS.train_stage == 1:
    #     shared_feature = tf.stop_gradient(shared_feature, f_geometry, f_score)
    pad_rois = roi_rotate_part.roi_rotate_tensor_pad(shared_feature, input_transform_matrix, input_box_masks,
                                                     input_box_widths)
    recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths, class_num=FLAGS.class_num)
    return f_score, f_geometry, recognition_logits
    # return f_score, f_geometry


def compute_loss(f_score, f_geometry, input_score_maps, input_geo_maps, input_training_masks, logits, label, lamda=0.01):
    detection_loss = detect_part.loss(input_score_maps, f_score, input_geo_maps, f_geometry, input_training_masks)
    tf.summary.scalar('detect_loss', detection_loss)

    recognition_loss, recognition_accuracy = recognize_part.loss(label, logits)
    tf.summary.scalar('recognition_loss', recognition_loss)
    tf.summary.scalar('recognition_accuracy', recognition_accuracy)
    # print("label:{}, predict:{}".format(max(label), max(logits)))
    # if FLAGS.train_stage == 2:
    #     model_loss = detection_loss + lamda * recognition_loss
    # elif FLAGS.train_stage == 0:
    #     model_loss = detection_loss
    # elif FLAGS.train_stage == 1:
    #     model_loss = recognition_loss
    # model_loss = detection_loss + recognition_loss
    model_loss = detection_loss + lamda * recognition_loss

    return detection_loss, recognition_loss, recognition_accuracy, model_loss


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    # define the input with placeholder
    # localization
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')
    input_transcription = tf.sparse_placeholder(tf.int32, name='input_transcription')

    # roi_rotate
    input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
    input_transform_matrix = tf.stop_gradient(input_transform_matrix)
    input_box_masks = []

    input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')
    input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)

    label = tf.placeholder(tf.float32, shape=[None, FLAGS.class_num])

    for i in range(FLAGS.batch_size_per_gpu):
        input_box_masks.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_' + str(i)))

    # build the graph
    f_score, f_geometry, logits = build_graph(input_images, input_transform_matrix, input_box_masks, input_box_widths,
                                              input_seq_len)


    # set global step
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    # set learning rate
    # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    learning_rate = FLAGS.learning_rate

    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    # calculate loss
    d_loss, r_loss, r_acc, model_loss = compute_loss(f_score, f_geometry, input_score_maps, input_geo_maps, input_training_masks, logits, label)
    tf.summary.scalar('total_loss', model_loss)
    # add the L2 regularization loss
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # batch normalization
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    # if FLAGS.train_stage == 1:
    #     print("Train recognition branch only!")
    #     recog_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='recog')
    #     # grads = opt.compute_gradients(total_loss, recog_vars)
    #     grads = opt.compute_gradients(total_loss)
    # else:
    #     grads = opt.compute_gradients(total_loss)

    grads = opt.compute_gradients(total_loss)
    # greds clip
    for i, (g, v) in enumerate(grads):
        if g is not None:
            grads[i] = (tf.clip_by_norm(g, 1.0), v)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    # save weight (max=40)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.max_restore_steps)
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        if os.path.isdir(FLAGS.pretrained_model_path):
            print("Restore pretrained model from other datasets")
            ckpt = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt, slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
        else:  # is *.ckpt
            print("Restore pretrained model from imagenet")
            variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)

    image = tf.placeholder("uint8", [None, None, 3])
    # bright_image = tf.image.adjust_brightness(image, -0.2)
    bright_image = tf.image.random_brightness(image, 0.2)
    constrast_img = tf.image.random_contrast(bright_image, 0.8, 1.2)
    # constrast_img = tf.image.adjust_contrast(image, 0.8)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        dg = data_generator.get_batch(input_images_dir=FLAGS.training_data_dir, input_gt_dir=FLAGS.training_gt_data_dir,
                                      num_workers=FLAGS.num_readers,
                                      input_size=FLAGS.input_size,
                                      batch_size=FLAGS.batch_size_per_gpu,
                                      class_num=FLAGS.class_num)

        start = time.time()
        for step in range(FLAGS.max_steps):
            # data_prepare_time = time.time()
            data = next(dg)
            for im_n in range(FLAGS.batch_size_per_gpu):
                data[0][im_n] = sess.run(constrast_img, feed_dict={image: data[0][im_n]})

            inp_dict = {input_images: data[0],
                        input_score_maps: data[2],
                        input_geo_maps: data[3],
                        input_training_masks: data[4],
                        input_transform_matrix: data[5],
                        input_box_widths: data[7],
                        input_transcription: data[8],
                        label: data[10]}
            # print("\n data_peparetime : " + str(time.time()-data_prepare_time))

            for i in range(FLAGS.batch_size_per_gpu):
                inp_dict[input_box_masks[i]] = data[6][i]

            dl, rl, ra, tl, _ = sess.run([d_loss, r_loss, r_acc, total_loss, train_op], feed_dict=inp_dict)
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu) / (time.time() - start)
                start = time.time()
                print(
                    'Step {:06d}, detect_loss {:.4f}, recognize_loss {:.4f}, recognize_acc_{:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, dl, rl, ra, tl, avg_time_per_step, avg_examples_per_second))

                """
                print "recognition results: "
                for pred in result:
                    print icdar.ground_truth_to_word(pred)
                """

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=step)

            if step % FLAGS.save_summary_steps == 0:
                """
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
                """
                dl, rl, ra, tl, _, summary_str = sess.run([d_loss, r_loss, r_acc, total_loss, train_op, summary_op], feed_dict=inp_dict)

                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    tf.app.run()
