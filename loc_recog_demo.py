import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
from statistics import mean
from module.stn import spatial_transformer_network as transformer

import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_integer('class_num', 230, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './checkpoints/', '')
tf.app.flags.DEFINE_string('demo_img', './demo.jpg', '')
tf.app.flags.DEFINE_string('output_dir', 'outputs/', '')
tf.app.flags.DEFINE_bool('no_write_images', True, 'do not write images')
# tf.app.flags.DEFINE_bool('use_vacab', True, 'strong, normal or weak')
from module import Backbone_branch, RoI_rotate, classification_branch
from data_provider.data_utils import restore_rectangle

FLAGS = tf.app.flags.FLAGS
detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = classification_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

blister_pack_names_input_path = "./blister_class.txt"
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''

    input_size = 512
    new_h, new_w, _ = im.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    im_padded[:new_h, :new_w, :] = im.copy()
    im = im_padded
    # resize the image to input size
    new_h, new_w, _ = im.shape
    resize_h = input_size
    resize_w = input_size
    im = cv2.resize(im, dsize=(resize_w, resize_h))
    resize_ratio_3_x = resize_w / float(new_w)
    resize_ratio_3_y = resize_h / float(new_h)

    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    # print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def get_project_matrix_and_width(text_polyses, target_height=16):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, target_height)
        mapped_x2, mapped_y2 = (target_height, 0)

        # width_box = math.ceil(8 * box_w / box_h)
        # width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width: 
            max_width = width_box 
        """

        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        width_box = target_height
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def dst_cal(src_x, src_y, dst_x, dst_y):
    x = dst_x - src_x
    y = dst_y - src_y
    # 用math.sqrt（）求平方根
    len = math.sqrt((x ** 2) + (y ** 2))
    return len


def polygon_sort(poly):
    point_len = []
    for i in range(4):
        point_len.append(dst_cal(poly[i][0], poly[i][1], 512, 0))
    len_sort = sorted(range(len(point_len)), key=lambda k: point_len[k])
    up_right_point = len_sort[0]
    # up_right_point = point_len.index(min(point_len))
    if (up_right_point - 1) < 0:
        last_point_index = 3
    else:
        last_point_index = up_right_point - 1

    if (up_right_point + 1) > 3:
        next_point_index = 0
    else:
        next_point_index = up_right_point + 1

    last_point_dst = dst_cal(poly[up_right_point][0], poly[up_right_point][1], poly[last_point_index][0],
                             poly[last_point_index][1])
    next_point_dst = dst_cal(poly[up_right_point][0], poly[up_right_point][1], poly[next_point_index][0],
                             poly[next_point_index][1])
    if last_point_dst > next_point_dst:
        point_index = []
        for i in range(up_right_point, up_right_point + 4, 1):
            if i >= 4:
                point_index.append(i - 4)
            else:
                point_index.append(i)
        poly = poly[(point_index[0], point_index[1], point_index[2], point_index[3]), :]
    else:
        point_index = []
        # up_right_point += 1
        up_right_point = len_sort[1]
        if up_right_point == 4:
            up_right_point = 0
        for i in range(up_right_point, up_right_point + 4, 1):
            if i >= 4:
                point_index.append(i - 4)
            else:
                point_index.append(i)
        poly = poly[(point_index[0], point_index[1], point_index[2], point_index[3]), :]

    return poly


def bktree_search(bktree, pred_word, dist=5):
    return bktree.query(pred_word, dist)


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def point_change(point):
    n_point = int(((point + 1) * 512) / 2)
    return n_point


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    """
    if FLAGS.use_vacab and os.path.exists("./vocab.txt"):
        bk_tree = BKTree(levenshtein, list_words('./vocab.txt'))
        # bk_tree = bktree.Tree()
    """
    with tf.get_default_graph().as_default():

        # define the placeholder
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        input_feature_map = tf.placeholder(tf.float32, shape=[None, None, None, 32], name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32, shape=[None, 6], name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32, shape=[None], name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32, shape=[None], name='input_box_widths')

        # define the model
        # input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map, input_transform_matrix, input_box_mask,
                                                         input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois, input_box_widths, class_num=FLAGS.class_num)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            cap0 = cv2.VideoCapture(0)
            cap1 = cv2.VideoCapture(1)
            cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # im_fn_list = get_images()
            count = 0
            cost_time = []

            print("start webcam ~")
            while (True):
                # class_pred =[]
                # confidence_pred = 0
                # ret0, frame0 = cap0.read()
                # assert ret0
                # ret1, frame1 = cap1.read()
                # assert ret1
                # im = cv2.vconcat((frame0, frame1))
                im = cv2.imread("{}".format(FLAGS.demo_img))

                im = im[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)
                # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
                start = time.time()
                shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry],
                                                               feed_dict={input_images: [im_resized]})
                gray_score = score[0]
                gray_score = cv2.resize(gray_score[:, :85, :], (480, 720))
                # cv2.imshow("gray", gray_score)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                # sort the boxes to true direction
                try:
                    if boxes.shape[0] >= 2:
                        if boxes[0][1] > boxes[1][1]:
                            boxes[[0, 1], :] = boxes[[1, 0], :]

                    new_boxes = []
                    for i, b in enumerate(boxes):
                        b = b[:8].reshape((4, 2))
                        b = polygon_sort(b)
                        # b = b[(1,2,3,0), :]
                        if i == 1:
                            b = b[(2, 3, 0, 1), :]
                        new_boxes.append(b)
                    boxes = np.asarray(new_boxes, dtype=np.float32)
                    # print(len(boxes))
                except:
                    print("shape error!")

                if boxes is not None and boxes.shape[0] != 0:
                    input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                    if input_roi_boxes.shape[0] == 2:
                        tmp_roi_boxes = input_roi_boxes[0:2]
                        boxes_masks = [0] * tmp_roi_boxes.shape[0]
                        transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)

                        # run the recognition part
                        recog_logits = sess.run(recognition_logits, feed_dict={input_feature_map: shared_feature_map,
                                                                               input_transform_matrix: transform_matrixes,
                                                                               input_box_mask[0]: boxes_masks,
                                                                               input_box_widths: box_widths})
                        # part level
                        np_pred = np.asarray(recog_logits)
                        mean_pred = np.mean(np_pred, axis=0)
                        softmax_x = np.asarray(mean_pred).reshape(-1).tolist()
                        softmax_x = softmax(softmax_x)
                        softmax_x = softmax_x.reshape(-1, 1)
                        im_pred = np.argmax(softmax_x, 0)[0]
                        confidence_pred = softmax_x[im_pred][0]
                        class_pred = blister_names[im_pred]
                        # if (im_gt == im_pred[0]):
                        #     print('{} : detect {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                        #         im_fn, timer['detect'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
                        print(" predict result:{}, confidence score:{}".format(blister_names[im_pred], confidence_pred))
                    timer['recog'] = time.time() - start

                    # Preparing for draw boxes
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                    for i, box in enumerate(boxes):
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                            continue
                        box = box * 1.875

                        # Draw bounding box
                        cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                      color=(0, 0, 255), thickness=3)
                        # Draw recognition results area
                        text_area = box.copy()
                        text_area[2, 1] = text_area[1, 1]
                        text_area[3, 1] = text_area[0, 1]
                        text_area[0, 1] = text_area[0, 1] - 15
                        text_area[1, 1] = text_area[1, 1] - 15
                        # cv2.fillPoly(im[:, :, ::-1], [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                        im_txt = im[:, :, ::-1]
                        # cv2.imshow("result", im_txt)
                else:
                    im_txt = im[:, :, ::-1]
                    class_pred = ""
                    confidence_pred = 0

                # print('detect {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                #      timer['detect'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))
                cost_time.append(duration)
                im_txt = cv2.resize(im_txt, (480, 720), interpolation=cv2.INTER_CUBIC)
                blank = np.ones((170, 480, 3), np.uint8)
                im_txt = cv2.vconcat((im_txt, blank))
                # cv2.rectangle(im_txt, pt1=(0, 0), pt2=(200, 200), color=(255, 255, 255), thickness=-1)
                cv2.putText(im_txt, text="Blister class : ", org=(0, 760), fontFace=1, fontScale=2, thickness=2,
                            color=(255, 255, 255))
                cv2.putText(im_txt, class_pred, (20, 795), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(im_txt, text="confidence score : ", org=(0, 830), fontFace=1, fontScale=2, thickness=2,
                            color=(255, 255, 255))
                cv2.putText(im_txt, "{:.2f}%".format(confidence_pred * 100), (20, 865), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.imshow("demo", im_txt)
                key = cv2.waitKey(30) & 0xFF
                if (key == 27):
                    cap0.release()
                    cap1.release()
                    # videoWriter.release()
                    cv2.destroyAllWindows()
                    break

            # average_time = mean(cost_time)
            # print("average cost time", average_time)


if __name__ == '__main__':
    tf.app.run()
