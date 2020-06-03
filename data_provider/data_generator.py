import time
import os
import random
import numpy as np
import tensorflow as tf
import cv2
from itertools import compress
from data_provider.data_utils import check_and_validate_polys, crop_area, rotate_image, generate_rbox, \
    get_project_matrix, sparse_tuple_from, crop_area_fix, generate_maps
from data_provider.loader import Loader
import pickle
from data_provider.data_enqueuer import GeneratorEnqueuer


def dense_to_one_hot(labels, n_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((1, n_classes), dtype=np.float32)
    labels_one_hot[0][labels] = 1
    return labels_one_hot


def generator(input_images_dir, input_gt_dir, input_size=512, batch_size=12, class_num=230,
              random_scale=np.array([0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2]), ):
    data_loader = Loader(shuffle=True)
    # image_list = np.array(data_loader.get_images(FLAGS.training_data_dir))
    image_list = np.array(data_loader.get_images(input_images_dir))
    # print('{} training images in {} '.format(image_list.shape[0], FLAGS.training_data_dir))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        batch_images = []
        batch_image_fns = []
        batch_score_maps = []
        batch_geo_maps = []
        batch_training_masks = []

        batch_polyses = []
        batch_tagses = []
        batch_boxes_masks = []

        batch_labels = []
        count = 0
        for i in index:
            try:

                start_time = time.time()
                im_fn = image_list[i]
                # print(im_fn)
                # if im_fn.split(".")[0][-1] == '0' or im_fn.split(".")[0][-1] == '2':
                #     continue
                im = cv2.imread(os.path.join(input_images_dir, im_fn))
                h, w, _ = im.shape
                file_name = "gt_" + im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt').split('\\')[-1]
                txt_fn = os.path.join(input_gt_dir, file_name)
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                # print(txt_fn)
                polys, tags, labels = data_loader.load_annotation(
                    txt_fn)  # Change for load text transiption

                if polys.shape[0] == 0:
                    continue

                polys, tags, labels = check_and_validate_polys(polys, tags, labels,
                                                                              (h, w))

                ############################# Data Augmentation ##############################
                '''
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                polys *= rd_scale

                # rotate image from [-10, 10]
                angle = random.randint(-10, 10)
                im, polys = rotate_image(im, polys, angle)

                # 600×600 random samples are cropped.
                im, polys, tags, selected_poly = crop_area(im, polys, tags, crop_background=False)
                # im, polys, tags, selected_poly = crop_area_fix(im, polys, tags, crop_size=(600, 600))
                labels = [labels[i] for i in selected_poly]
                if polys.shape[0] == 0 or len(labels) == 0:
                    continue
                '''



                ################################################################################
                # prepare_one_img_time = time.time()
                # pad the image to the training input size or the longer side of image
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
                polys[:, :, 0] *= resize_ratio_3_x
                polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape


                training_mask, rectangles = generate_rbox((new_h, new_w), polys, tags)
                rectangles_list = []
                polys = polys.astype(np.float32)
                for i in range(2):
                    rectangles_list.append(polys[i].flatten())

                # read the geo_map and score map from the file
                im_name = (im_fn.split("\\")[-1]).split(".")[0]
                geo_dir = "{}/geo_map".format(im_fn.split("JPEGImages")[0])
                with open("{}/{}.pickle".format(geo_dir, im_name), "rb") as f_in:
                    geo_map = pickle.load(f_in)
                f_in.close()
                score_dir = "{}/score_map".format(im_fn.split("JPEGImages")[0])
                with open("{}/{}.pickle".format(score_dir, im_name), "rb") as f_in:
                    score_map = pickle.load(f_in)
                f_in.close()
                # print("\nprepare_one_img_cost:" + str(time.time() - start_time))

                mask = [not (word == [-1]) for word in labels]
                # remove the unreadable
                labels = list(compress(labels, mask))
                rectangles = list(compress(rectangles_list, mask))

                assert len(labels) == len(rectangles), "rotate rectangles' num is not equal to text label"

                if len(labels) == 0:
                    continue

                # turn the label to one hot representation
                labels = [str(i) for i in labels[0]]
                label = ''.join(labels)
                one_hot_label = dense_to_one_hot(int(label), n_classes=class_num)

                boxes_mask = np.array([count] * len(rectangles))

                count += 1

                batch_images.append(im[:, :, ::-1].astype(np.float32))
                batch_image_fns.append(im_fn)
                batch_score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                batch_geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                batch_training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                batch_polyses.append(rectangles)
                batch_boxes_masks.append(boxes_mask)
                batch_labels.extend(one_hot_label)
                batch_tagses.append(tags)


                if len(batch_images) == batch_size:
                    batch_polyses = np.concatenate(batch_polyses)
                    batch_tagses = np.concatenate(batch_tagses)
                    batch_transform_matrixes, batch_box_widths = get_project_matrix(batch_polyses,
                                                                                    batch_tagses)
                    # TODO limit the batch size of recognition 
                    batch_labels_sparse = sparse_tuple_from(np.array(batch_labels))

                    # yield images, image_fns, score_maps, geo_maps, training_masks
                    yield batch_images, batch_image_fns, batch_score_maps, batch_geo_maps, batch_training_masks, batch_transform_matrixes, batch_boxes_masks, batch_box_widths, batch_labels_sparse, batch_polyses, batch_labels
                    batch_images = []
                    batch_image_fns = []
                    batch_score_maps = []
                    batch_geo_maps = []
                    batch_training_masks = []
                    batch_polyses = []
                    batch_tagses = []
                    batch_boxes_masks = []
                    batch_labels = []
                    count = 0
            except Exception as e:
                import traceback
                print(im_fn)
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=False)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=10, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def test():
    font = cv2.FONT_HERSHEY_SIMPLEX
    dg = get_batch(num_workers=1, input_size=512, batch_size=4)
    for iter in range(2000):
        print("iter: ", iter)
        data = next(dg)
        imgs = data[0]
        imgs_name = data[1]
        polygons = data[-2]
        labels = data[-1]
        masks = data[6]
        prev_start_index = 0
        for i, (img, mask, img_name) in enumerate(zip(imgs, masks, imgs_name)):
            # img_name = ''
            im = img.copy()
            poly_start_index = len(masks[i - 1])
            poly_end_index = len(masks[i - 1]) + len(mask)
            for poly, la, in zip(polygons[prev_start_index:(prev_start_index + len(mask))],
                                 labels[prev_start_index:prev_start_index + len(mask)]):
                cv2.polylines(img, [poly.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                # trans = ground_truth_to_word(la)
                # img_name = img_name + trans + '_'
            img_name = img_name[:-1] + '.jpg'
            cv2.imwrite("./polygons/" + os.path.basename(img_name), img)

            prev_start_index += len(mask)


def generator_all(input_images_dir, input_gt_dir, output_geo_dir, output_score_dir, input_size=512, class_num=230):
    geo_dir = "./"
    print(input_images_dir)
    data_loader = Loader(shuffle=True)
    # image_list = np.array(data_loader.get_images(FLAGS.training_data_dir))
    image_list = np.array(data_loader.get_images(input_images_dir))
    print("img_num : ", len(image_list))
    # print('{} training images in {} '.format(image_list.shape[0], FLAGS.training_data_dir))
    index = np.arange(0, image_list.shape[0])
    # while True:
        # np.random.shuffle(index)
    for i in index:
            try:

                im_fn = image_list[i]
                # print(im_fn)
                # if im_fn.split(".")[0][-1] == '0' or im_fn.split(".")[0][-1] == '2':
                #     continue
                im = cv2.imread(os.path.join(input_images_dir, im_fn))
                h, w, _ = im.shape
                file_name = "gt_" + im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt').split('\\')[-1]

                txt_fn = os.path.join(input_gt_dir, file_name)
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue
                # print(txt_fn)
                polys, tags, labels = data_loader.load_annotation(
                    txt_fn)  # Change for load text transiption

                if polys.shape[0] == 0:
                    continue

                polys, tags, labels = check_and_validate_polys(polys, tags, labels,
                                                                              (h, w))

                # prepare_one_img_time = time.time()
                # pad the image to the training input size or the longer side of image
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
                polys[:, :, 0] *= resize_ratio_3_x
                polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape


                score_map, geo_map, training_mask, rectangles = generate_maps((new_h, new_w), polys, tags)
                # rectangles_list = []
                # polys = polys.astype(np.float32)
                # for i in range(2):
                #     rectangles_list.append(polys[i].flatten())
                im_name = (im_fn.split("\\")[-1]).split(".")[0]
                print(im_name)
                with open("{}/{}.pickle".format(output_geo_dir, im_name), "wb") as f_out:
                    pickle.dump(geo_map, f_out)
                f_out.close()
                with open("{}/{}.pickle".format(output_score_dir, im_name), "wb") as f_out:
                    pickle.dump(score_map, f_out)
                f_out.close()




            except Exception as e:
                import traceback
                print(im_fn)
                traceback.print_exc()
                continue


if __name__ == '__main__':
    test()
