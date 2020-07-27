# coding=utf-8

from datetime import datetime
import cv2
import os
import json
import numpy as np
from scipy.spatial import distance as dist

domainA_img_input_path = "./rand50_img_E3"
# domainA_img_input_path = "J:/domainAandB_dataset_make/domain_AandB_dataset/domainA_img"
domainA_img_output_path = "./img"
domainA_img_label_json_path = "./json"
domainB_img_output_path = "./label_img"

drawing = False  # true if mouse is pressed
mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1
s_points_L = []
s_points_R = []


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def sortpts_clockwise(A):
    # Sort A based on Y(col-2) coordinates
    sortedAc2 = A[np.argsort(A[:, 1]), :]
    # Get top two and bottom two points
    top2 = sortedAc2[0:2, :]
    bottom2 = sortedAc2[2:, :]
    # Sort top2 points to have the first row as the top-left one
    sortedtop2c1 = top2[np.argsort(top2[:, 0]), :]
    top_left = sortedtop2c1[0, :]
    # Use top left point as pivot & calculate sq-euclidean dist against
    # bottom2 points & thus get bottom-right, bottom-left sequentially
    sqdists = dist.cdist(top_left[None], bottom2, 'sqeuclidean')
    rest2 = bottom2[np.argsort(np.max(sqdists, 0))[::-1], :]
    # Concatenate all these points for the final output
    result = np.concatenate((sortedtop2c1, rest2), axis=0)
    return result


def findright4points(front4points):
    f_have_horizontal = checkhorizontal(front4points)
    if (not f_have_horizontal):
        f_bottom_right_index, f_bottom_left_index = lookfor_lowest2points(front4points)
    else:
        sort_f_index = np.argsort(front4points[:, 1])
        if (front4points[sort_f_index[1]][0] <= front4points[sort_f_index[2]][0]):
            f_bottom_left_index = sort_f_index[1]
        else:
            f_bottom_left_index = sort_f_index[2]

    refront4points = resortpoints(f_bottom_left_index, front4points)

    return refront4points


def checkhorizontal(matrix):
    state = False
    sort = np.argsort(matrix[:, 1])

    if abs(matrix[sort[1]][1] - matrix[sort[2]][1]) < 5:
        state = True

    return state


def lookfor_lowest2points(points):
    sort_index = np.argsort(points[:, 1])
    if (points[sort_index[2]][0] >= points[sort_index[3]][0]):
        # bottom_right = points[ sort_index[2] ]
        bottom_right_index = sort_index[2]
        # bottom_left = points[ sort_index[3] ]
        bottom_left_index = sort_index[3]
    elif (points[sort_index[2]][0] < points[sort_index[3]][0]):
        # bottom_right = points[sort_index[3]]
        bottom_right_index = sort_index[3]
        # bottom_left = points[sort_index[2]]
        bottom_left_index = sort_index[2]

    return bottom_right_index, bottom_left_index


def resortpoints(index, result):
    real_result = np.zeros(result.shape, result.dtype)
    for j in range(len(result)):
        index_temp = index + j
        if (index_temp >= len(result)):
            index_temp = index_temp - len(result)
        real_result[j] = result[index_temp]

    return real_result


def draw_circle_L(event, x, y, flags, param):
    global ix, iy, drawing, mode, s_points_L

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(s_points_L) < 4:
            # print("({0},{1})".format(x, y))
            cv2.circle(s_img_L, (x, y), 4, (0, 0, 255), -1)
            s_points_L.append([x, y])
        else:
            print("too many points")

    if event == cv2.EVENT_LBUTTONUP:
        if len(s_points_L) == 2:
            cv2.line(s_img_L, (s_points_L[0][0], s_points_L[0][1]), (s_points_L[1][0], s_points_L[1][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
        elif len(s_points_L) == 3:
            cv2.line(s_img_L, (s_points_L[1][0], s_points_L[1][1]), (s_points_L[2][0], s_points_L[2][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
        elif len(s_points_L) == 4:
            cv2.line(s_img_L, (s_points_L[0][0], s_points_L[0][1]), (s_points_L[3][0], s_points_L[3][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
            cv2.line(s_img_L, (s_points_L[2][0], s_points_L[2][1]), (s_points_L[3][0], s_points_L[3][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)


def draw_circle_R(event, x, y, flags, param):
    global ix, iy, drawing, mode, s_points_R

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(s_points_R) < 4:
            # print("({0},{1})".format(x, y))
            cv2.circle(s_img_R, (x, y), 4, (0, 0, 255), -1)
            s_points_R.append([x, y])
        else:
            print("too many points")

    if event == cv2.EVENT_LBUTTONUP:
        if len(s_points_R) == 2:
            cv2.line(s_img_R, (s_points_R[0][0], s_points_R[0][1]), (s_points_R[1][0], s_points_R[1][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
        elif len(s_points_R) == 3:
            cv2.line(s_img_R, (s_points_R[1][0], s_points_R[1][1]), (s_points_R[2][0], s_points_R[2][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
        elif len(s_points_R) == 4:
            cv2.line(s_img_R, (s_points_R[0][0], s_points_R[0][1]), (s_points_R[3][0], s_points_R[3][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)
            cv2.line(s_img_R, (s_points_R[2][0], s_points_R[2][1]), (s_points_R[3][0], s_points_R[3][1]), (0, 0, 255),
                     1,
                     cv2.LINE_AA)


def write_json(poly, min_rect, bbox, path):
    # print poly

    # min_rect = np
    # poly = poly.astype(int)
    # bbox = bbox.astype(int)
    data = [{
        "bbox": [
            bbox[0][0],
            bbox[0][1],
            bbox[1][0],
            bbox[1][1]
        ],
        "poly": [
            [poly[0][0], poly[0][1]],
            [poly[1][0], poly[1][1]],
            [poly[2][0], poly[2][1]],
            [poly[3][0], poly[3][1]]
        ],
        "min_rect": [
            [min_rect[0][0], min_rect[0][1]],
            [min_rect[1][0], min_rect[1][1]],
            [min_rect[2][0], min_rect[2][1]],
            [min_rect[3][0], min_rect[3][1]]
        ]
    }]

    with open(path, 'w') as fw:
        json.dump(data, fw, indent=3, cls=MyEncoder)
    fw.close()


def main():
    global s_img_L, s_img_R, s_points_L, s_points_R, ix, iy
    #########################################
    cls_num = 0
    dir_num = 0
    #########################################
    domainA_imgs = os.listdir(domainA_img_input_path)
    cls_flag = False

    for cls in range(cls_num, len(domainA_imgs)):

        img_path = domainA_img_output_path + "/" + domainA_imgs[cls]
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        label_img_path = domainB_img_output_path + "/" + domainA_imgs[cls]
        if not os.path.exists(label_img_path):
            os.makedirs(label_img_path)

        json_path = domainA_img_label_json_path + "/" + domainA_imgs[cls]
        if not os.path.exists(json_path):
            os.makedirs(json_path)

        # next_flag = False
        # back label ###################################################################################################
        count = len(os.listdir(domainB_img_output_path + "/" + domainA_imgs[cls])) / 2
        print(count)
        if count < 15:
            myclass = domainA_img_input_path + "/" + domainA_imgs[cls]
            print(domainA_imgs[cls])
            L_file = myclass + "/L_back"
            R_file = myclass + "/R_back"
            L_img = os.listdir(L_file)
            # dir_num = 0
            if cls_flag == True:
                dir_num = 0
                cls_flag = False

            for dir in range(dir_num, len(L_img)):
                next_flag = False
                L_img_dir = L_file + "/" + L_img[dir]
                R_img_dir = R_file + "/" + L_img[dir]
                # print(L_img_dir)
                domainA_img_L = cv2.imread(L_img_dir)
                domainA_img_R = cv2.imread(R_img_dir)
                # domainA_img = cv2.imread("{0}/{1}.jpg".format(domainA_img_input_path, num))
                s_img_L = domainA_img_L.copy()
                s_img_R = domainA_img_R.copy()
                cv2.namedWindow("img_" + str(dir) + "_L")
                cv2.namedWindow("img_" + str(dir) + "_R")
                cv2.setMouseCallback("img_" + str(dir) + "_L", draw_circle_L)
                cv2.setMouseCallback("img_" + str(dir) + "_R", draw_circle_R)
                while True:
                    cv2.imshow("img_" + str(dir) + "_L", s_img_L)
                    cv2.moveWindow("img_" + str(dir) + "_L", 100, 100)
                    cv2.imshow("img_" + str(dir) + "_R", s_img_R)
                    cv2.moveWindow("img_" + str(dir) + "_R", 740, 100)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('s'), ord('S')]:  # save
                        if len(s_points_R) < 4:
                            print("img_" + str(count) + "_R_back label unsussesful")
                            print("cls = " + str(cls), "dir = " + str(dir))
                        elif len(s_points_L) < 4:
                            print("img_" + str(count) + "_L_back label unsussesful")
                            print("cls = " + str(cls), "dir = " + str(dir))
                        else:
                            print("img_" + str(count) + "_back_save sussesful")
                        break

                    if key in [ord('q'), ord('Q')]:  # next picture
                        next_flag = True
                        break
                    if key in [ord('r'), ord('R')]:  # relabel
                        cv2.destroyAllWindows()
                        s_points_L = []
                        s_points_R = []
                        s_img_L = domainA_img_L.copy()
                        s_img_R = domainA_img_R.copy()
                        cv2.namedWindow("img_" + str(dir) + "_L")
                        cv2.namedWindow("img_" + str(dir) + "_R")
                        cv2.setMouseCallback("img_" + str(dir) + "_L", draw_circle_L)
                        cv2.setMouseCallback("img_" + str(dir) + "_R", draw_circle_R)
                        continue
                    if key == 27:  # esc and print num
                        print("cls = " + str(cls), "dir = " + str(dir))
                        cv2.destroyAllWindows()
                        exit(0)

                cv2.destroyAllWindows()

                if next_flag:
                    s_points_L = []
                    s_points_R = []
                    continue

                s_points_L_tmp = np.array(s_points_L, np.int32)
                s_points_L = []
                s_points_L_tmp = sortpts_clockwise(s_points_L_tmp)
                poly_4points = findright4points(s_points_L_tmp)
                min_rect = cv2.minAreaRect(poly_4points)
                min_rect_4point = cv2.boxPoints(min_rect)
                min_rect_4point = np.array(min_rect_4point, np.int32)
                count += 1

                cv2.imwrite(img_path + "/" + "{:06d}".format(int(count)) + "_L.jpg", domainA_img_L)

                domainB_img = np.zeros(domainA_img_L.shape, np.uint8)
                domainB_img = cv2.fillPoly(domainB_img, [poly_4points], (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite(label_img_path + "/" + "{:06d}".format(int(count)) + "_L.jpg", domainB_img)

                x_min = np.min(poly_4points[:, 0])
                x_max = np.max(poly_4points[:, 0])
                y_min = np.min(poly_4points[:, 1])
                y_max = np.max(poly_4points[:, 1])

                bbox_topleft_bottomright = np.array([[x_min, y_min], [x_max, y_max]], np.int32)
                domainA_img_label_json_output_path_L = json_path + "/" + "{:06d}".format(int(count)) + "_L.json"
                write_json(poly_4points, min_rect_4point, bbox_topleft_bottomright,
                           domainA_img_label_json_output_path_L)

                # right scene
                s_points_R_tmp = np.array(s_points_R, np.int32)
                s_points_R = []
                s_points_R_tmp = sortpts_clockwise(s_points_R_tmp)
                poly_4points = findright4points(s_points_R_tmp)
                min_rect = cv2.minAreaRect(poly_4points)
                min_rect_4point = cv2.boxPoints(min_rect)
                min_rect_4point = np.array(min_rect_4point, np.int32)

                cv2.imwrite(img_path + "/" + "{:06d}".format(int(count)) + "_R.jpg", domainA_img_R)

                domainB_img = np.zeros(domainA_img_R.shape, np.uint8)
                domainB_img = cv2.fillPoly(domainB_img, [poly_4points], (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite(label_img_path + "/" + "{:06d}".format(int(count)) + "_R.jpg", domainB_img)

                x_min = np.min(poly_4points[:, 0])
                x_max = np.max(poly_4points[:, 0])
                y_min = np.min(poly_4points[:, 1])
                y_max = np.max(poly_4points[:, 1])

                bbox_topleft_bottomright = np.array([[x_min, y_min], [x_max, y_max]], np.int32)
                domainA_img_label_json_output_path_R = json_path + "/" + "{:06d}".format(int(count)) + "_R.json"
                write_json(poly_4points, min_rect_4point, bbox_topleft_bottomright,
                           domainA_img_label_json_output_path_R)
                if count >= 15:
                    break

        # for label ###################################################################################################
        count = len(os.listdir(domainB_img_output_path + "/" + domainA_imgs[cls])) / 2
        print(count)
        if count >= 15:
            myclass = domainA_img_input_path + "/" + domainA_imgs[cls]
            print(domainA_imgs[cls])
            L_file = myclass + "/L_for"
            R_file = myclass + "/R_for"
            L_img = os.listdir(L_file)
            # dir_num = 0
            if cls_flag == True:
                dir_num = 0
                cls_flag = False

            for dir in range(dir_num, len(L_img)):
                next_flag = False
                L_img_dir = L_file + "/" + L_img[dir]
                R_img_dir = R_file + "/" + L_img[dir]
                # print(L_img_dir)
                domainA_img_L = cv2.imread(L_img_dir)
                domainA_img_R = cv2.imread(R_img_dir)
                # domainA_img = cv2.imread("{0}/{1}.jpg".format(domainA_img_input_path, num))
                s_img_L = domainA_img_L.copy()
                s_img_R = domainA_img_R.copy()
                cv2.namedWindow("img_" + str(dir) + "_L")
                cv2.namedWindow("img_" + str(dir) + "_R")
                cv2.setMouseCallback("img_" + str(dir) + "_L", draw_circle_L)
                cv2.setMouseCallback("img_" + str(dir) + "_R", draw_circle_R)
                while True:
                    cv2.imshow("img_" + str(dir) + "_L", s_img_L)
                    cv2.moveWindow("img_" + str(dir) + "_L", 100, 100)
                    cv2.imshow("img_" + str(dir) + "_R", s_img_R)
                    cv2.moveWindow("img_" + str(dir) + "_R", 740, 100)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('s'), ord('S')]:  # save
                        if len(s_points_R) < 4:
                            print("img_" + str(count) + "_R_for label unsussesful")
                            print("cls = " + str(cls), "dir = " + str(dir))
                        elif len(s_points_L) < 4:
                            print("img_" + str(count) + "_L_for label unsussesful")
                            print("cls = " + str(cls), "dir = " + str(dir))
                        else:
                            print("img_" + str(count) + "_for save sussesful")
                        break

                    if key in [ord('q'), ord('Q')]:  # next picture
                        next_flag = True
                        break
                    if key in [ord('r'), ord('R')]:  # relabel
                        cv2.destroyAllWindows()
                        s_points_L = []
                        s_points_R = []
                        s_img_L = domainA_img_L.copy()
                        s_img_R = domainA_img_R.copy()
                        cv2.namedWindow("img_" + str(dir) + "_L")
                        cv2.namedWindow("img_" + str(dir) + "_R")
                        cv2.setMouseCallback("img_" + str(dir) + "_L", draw_circle_L)
                        cv2.setMouseCallback("img_" + str(dir) + "_R", draw_circle_R)
                        continue
                    if key == 27:  # esc and print num
                        print("cls = " + str(cls), "dir = " + str(dir))
                        cv2.destroyAllWindows()
                        exit(0)

                cv2.destroyAllWindows()

                if next_flag:
                    s_points_L = []
                    s_points_R = []
                    continue

                s_points_L_tmp = np.array(s_points_L, np.int32)
                s_points_L = []
                s_points_L_tmp = sortpts_clockwise(s_points_L_tmp)
                poly_4points = findright4points(s_points_L_tmp)
                min_rect = cv2.minAreaRect(poly_4points)
                min_rect_4point = cv2.boxPoints(min_rect)
                min_rect_4point = np.array(min_rect_4point, np.int32)
                count += 1

                cv2.imwrite(img_path + "/" + "{:06d}".format(int(count)) + "_L.jpg", domainA_img_L)

                domainB_img = np.zeros(domainA_img_L.shape, np.uint8)
                domainB_img = cv2.fillPoly(domainB_img, [poly_4points], (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite(label_img_path + "/" + "{:06d}".format(int(count)) + "_L.jpg", domainB_img)

                x_min = np.min(poly_4points[:, 0])
                x_max = np.max(poly_4points[:, 0])
                y_min = np.min(poly_4points[:, 1])
                y_max = np.max(poly_4points[:, 1])

                bbox_topleft_bottomright = np.array([[x_min, y_min], [x_max, y_max]], np.int32)
                domainA_img_label_json_output_path_L = json_path + "/" + "{:06d}".format(int(count)) + "_L.json"
                write_json(poly_4points, min_rect_4point, bbox_topleft_bottomright,
                           domainA_img_label_json_output_path_L)

                # right scene
                s_points_R_tmp = np.array(s_points_R, np.int32)
                s_points_R = []
                s_points_R_tmp = sortpts_clockwise(s_points_R_tmp)
                poly_4points = findright4points(s_points_R_tmp)
                min_rect = cv2.minAreaRect(poly_4points)
                min_rect_4point = cv2.boxPoints(min_rect)
                min_rect_4point = np.array(min_rect_4point, np.int32)

                cv2.imwrite(img_path + "/" + "{:06d}".format(int(count)) + "_R.jpg", domainA_img_R)

                domainB_img = np.zeros(domainA_img_R.shape, np.uint8)
                domainB_img = cv2.fillPoly(domainB_img, [poly_4points], (255, 255, 255), cv2.LINE_AA)
                cv2.imwrite(label_img_path + "/" + "{:06d}".format(int(count)) + "_R.jpg", domainB_img)

                x_min = np.min(poly_4points[:, 0])
                x_max = np.max(poly_4points[:, 0])
                y_min = np.min(poly_4points[:, 1])
                y_max = np.max(poly_4points[:, 1])

                bbox_topleft_bottomright = np.array([[x_min, y_min], [x_max, y_max]], np.int32)
                domainA_img_label_json_output_path_R = json_path + "/" + "{:06d}".format(int(count)) + "_R.json"
                write_json(poly_4points, min_rect_4point, bbox_topleft_bottomright,
                           domainA_img_label_json_output_path_R)
                if count >= 30:
                    break

        cls_flag = True


if __name__ == "__main__":

    if not os.path.exists(domainA_img_input_path):
        print("domainA_img_input_path error")
        exit(0)
    if not os.path.exists(domainA_img_output_path):
        os.makedirs(domainA_img_output_path)
    if not os.path.exists(domainA_img_label_json_path):
        os.makedirs(domainA_img_label_json_path)
    if not os.path.exists(domainB_img_output_path):
        os.makedirs(domainB_img_output_path)
    main()
