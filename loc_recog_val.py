import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from statistics import mean
import lanms

'''
D:/FOTS/FOTS_TF-dev/datasets/PS1_PS2_vertical_dataset/test/JPEGImages/
D:/FOTS/FOTS_TF-dev/datasets/PS1_PS2_vertical_dataset/test/Annotations/
E:/FastROR/checkpoints/0521_loc_recog_imagnet_pre/
'''

tf.app.flags.DEFINE_integer('class_num', 230, '')
tf.app.flags.DEFINE_string('test_data_path', '/path/to/your/testing images/', '')
tf.app.flags.DEFINE_string('test_gt_path', '/path/to/your/testing annotations/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/path/to/your/checkpoints/', '')
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



def iou_cal(predict_box, gt_box):
    union = np.bitwise_or(predict_box, gt_box)
    inter = np.bitwise_and(predict_box, gt_box)
    # union = cv2.cvtColor(union, cv2.COLOR_GRAY2BGR)
    # inter = cv2.cvtColor(inter, cv2.COLOR_GRAY2BGR)
    contours_union, hierarchy_union = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_inter, hierarchy_inter = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_union = 0
    area_inter = 0
    for cnt_union in range(len(contours_union)):
        area_union += cv2.contourArea(contours_union[cnt_union])
    for cnt_inter in range(len(contours_inter)):
        area_inter += cv2.contourArea(contours_inter[cnt_inter])
    iou = float(area_inter / (area_union))

    return iou


def draw_gt_box(src):
    box = []
    with open(src, 'r')as fp:
        all_lines = fp.readlines()
        # print(all_lines[1])
        for b in all_lines:
            box.append(b.split(",")[:8])
    fp.close()
    gt_area = np.zeros((960, 640), np.uint8)
    for b in box:
        # box = np.asarray(b)
        box = np.array(b, np.int32).reshape((-1, 2))
        cv2.fillPoly(gt_area, [box], color=(255, 255, 255))
    return gt_area


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
    # print('Find {} images'.format(len(files)))
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


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.15, nms_thres=0.2):
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
    # filter the score map, output the index >score map threshhold
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



def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    # print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))



def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise
    with tf.get_default_graph().as_default() as graph:

        # define the placehodler
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
        # stats_graph(graph)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            weight_len = len(ckpt_state.all_model_checkpoint_paths)
            # print(weight_len)

            for w in range(0, weight_len):
                model_path = os.path.join(FLAGS.checkpoint_path,
                                          os.path.basename(ckpt_state.all_model_checkpoint_paths[w]))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)

                im_fn_list = get_images()
                all_pred = []
                all_gt = []
                iou_list = []
                img_per_cls = len(im_fn_list) / FLAGS.class_num
                # img_per_cls = 20
                print("Img_per_cls:", img_per_cls)
                for im_fn in im_fn_list:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(im)
                    # im_resized_d, (ratio_h_d, ratio_w_d) = resize_image_detection(im)

                    timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
                    start = time.time()
                    shared_feature_map, score, geometry = sess.run([shared_feature, f_score, f_geometry],
                                                                   feed_dict={input_images: [im_resized]})

                    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
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
                    except AttributeError:
                        print("shape not found")

                    # read the ground-truth as four points
                    # txt_dir = "{}gt_img{}".format(FLAGS.test_gt_path, (im_fn.split("img")[1]).split(".")[0]+".txt")
                    # boxes = []
                    # with open(txt_dir, 'r')as fp:
                    #     all_lines = fp.readlines()
                    #     # print(all_lines[1])
                    #     for i, b in enumerate(all_lines):
                    #     # for b in all_lines:
                    #         box = b.split(",")[:8]
                    #         box = np.asarray(box, dtype=np.float32)
                    #         box = box.reshape((4, 2))
                    #         box = polygon_sort(box)
                    #         if i == 1:
                    #             box = box[(2, 3, 0, 1), :]
                    #         # box = sort_poly(box)
                    #         # box = box[[2, 3, 0, 1]]
                    #         box = box.reshape((-1, 8, 1))
                    #         boxes.append(box)
                    # boxes = np.asarray(boxes, dtype=np.float32)
                    # boxes = boxes*0.53333333

                    timer['detect'] = time.time() - start
                    # print(im_fn)
                    im_num = int((im_fn.split(".")[0]).split("img_")[1])
                    if (im_num % img_per_cls == 0):
                        im_gt = int((im_num / img_per_cls) - 1)
                    else:
                        im_gt = int(im_num / img_per_cls)

                    # print(im_gt)
                    predict_area = np.zeros(im[:, :, ::-1].shape[:2], np.uint8)
                    if boxes is not None and boxes.shape[0] != 0:
                        res_file_path = os.path.join(FLAGS.output_dir,
                                                     'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                        input_roi_boxes = boxes[:, :8].reshape(-1, 8)
                        if input_roi_boxes.shape[0] == 2:
                            tmp_roi_boxes = input_roi_boxes[0:2]
                            boxes_masks = [0] * tmp_roi_boxes.shape[0]
                            transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)

                            # run the recognition part
                            recog_logits = sess.run(recognition_logits,
                                                    feed_dict={input_feature_map: shared_feature_map,
                                                               input_transform_matrix: transform_matrixes,
                                                               input_box_mask[0]: boxes_masks,
                                                               input_box_widths: box_widths})
                            # part level
                            np_pred = np.asarray(recog_logits)
                            mean_pred = np.mean(np_pred, axis=0)
                            # mean_pred = np.average(np_pred, axis=0, weights=[1, 1, 0, 1])
                            softmax_x = np.asarray(mean_pred).reshape(-1).tolist()

                            softmax_x = softmax(softmax_x)
                            softmax_x = softmax_x.reshape(-1, 1)
                            im_pred = np.argmax(softmax_x, 0)
                            all_pred.append(im_pred)
                            all_gt.append(im_gt)
                            # print("ground-truth:[{}] predict:{}".format(im_gt, im_pred))

                        timer['recog'] = time.time() - start

                        # Preparing for draw boxes
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h

                        # with open(res_file_path, 'w') as f:
                        for i, box in enumerate(boxes):
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                                continue

                            # f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                            #     box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                            #     box[3, 1]
                            # ))
                            box = box * (960 / 512)

                            # Draw bounding box
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                          color=(0, 0, 255), thickness=3)
                            cv2.fillPoly(predict_area, [box.astype(np.int32).reshape((-1, 1, 2))],
                                         color=(255, 255, 255))
                            # Draw recognition results area
                            text_area = box.copy()
                            text_area[2, 1] = text_area[1, 1]
                            text_area[3, 1] = text_area[0, 1]
                            text_area[0, 1] = text_area[0, 1] - 15
                            text_area[1, 1] = text_area[1, 1] - 15
                            # cv2.fillPoly(im[:, :, ::-1], [text_area.astype(np.int32).reshape((-1, 1, 2))], color=(255, 255, 0))
                            im_txt = im[:, :, ::-1]
                    else:
                        res_file = os.path.join(FLAGS.output_dir,
                                                'res_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                        f = open(res_file, "w")
                        im_txt = None
                        f.close()

                    # calculate the intersection of union
                    gt_file = os.path.join(FLAGS.test_gt_path,
                                           'gt_' + '{}.txt'.format(os.path.basename(im_fn).split('.')[0]))
                    gt_area = draw_gt_box(gt_file)
                    iou = iou_cal(predict_area, gt_area)
                    # print("IOU: ", iou)
                    iou_list.append(iou)

                    # print('{} : detect {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    #     im_fn, timer['detect'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

                    duration = time.time() - start_time
                    # print('[timing] {}'.format(duration))

                    if not FLAGS.no_write_images:
                        img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                        # cv2.imwrite(img_path, im[:, :, ::-1])
                        if im_txt is not None:
                            cv2.imwrite(img_path, im_txt)

                # print(confusion_matrix(all_gt, all_pred))
                # np.savetxt("confusion_matrix.csv", confusion_matrix(all_gt, all_pred), delimiter=" ")
                average_iou = mean(iou_list)
                accuracy = accuracy_score(all_gt, all_pred)
                precision = precision_score(all_gt, all_pred, average='macro')
                recall = recall_score(all_gt, all_pred, average='macro')
                F1_score = f1_score(all_gt, all_pred, average='macro')

                print(
                    "weight: {} || accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f}, f1_score:{:.4f}, IoU:{:.4f} \n".format(
                        (2500 * (w + 1)), accuracy, precision, recall, F1_score, average_iou))


if __name__ == '__main__':
    tf.app.run()
