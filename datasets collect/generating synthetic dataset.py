import cv2 as cv
from PIL import Image
import numpy as np
import math
from scipy.spatial import distance as dist
import json
import random
import os


def img_label_sep(label):
    # label = img[:,0:256]
    # pic = img[:,256:]
    gray = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

    ret, gray = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cv.drawContours(label,contours,-1,(0,0,255),1)
    max_area = 0
    for cnt2 in range(len(contours)):
        area = cv.contourArea(contours[cnt2])
        if area > max_area:
            max_area = area
            index = cnt2
    hull2 = cv.convexHull(contours[index])
    rect = cv.minAreaRect(hull2)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # cv.drawContours(pic,[box],-1,(0,0,255),1)
    # cv.imshow("im",pic)
    # cv.waitKey()
    return box


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


def hand_rotate(image, angle, center=None, scale=1):
    # 取得圖片尺寸
    (h, w) = image.shape[:2]
    # hand_rgb = image[:, :, ::-1]
    # plt.imshow(hand_rgb)
    # plt.show()

    # 設定旋轉中心錨點
    if center is None:
        center = (198, 230)

    # 進行旋轉
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))

    # 回傳旋轉後的圖片
    return rotated


def rotate(image, angle, center=None, scale=1, white_bg=0):
    # 取得圖片尺寸
    (h, w) = image.shape[:2]
    # hand_rgb = image[:, :, ::-1]
    # plt.imshow(hand_rgb)
    # plt.show()

    # 設定旋轉中心錨點
    if center is None:
        center = (w / 2, h / 2)

    # 進行旋轉
    if white_bg == 1:
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    else:
        M = cv.getRotationMatrix2D(center, angle, scale)
        rotated = cv.warpAffine(image, M, (w, h))

    # 回傳旋轉後的圖片
    return rotated


def addhand(drug, rotate_hand, gt_x, gt_y):
    # 決定手圖片的裁切範圍
    drug_row, drug_col, drug_channel = drug.shape

    rotate_hand_col = gt_x - 198  # 插入圖片的起始點 = 221(label)-198(圖片角落的x)
    rotate_hand_row = gt_y - 230  # 插入圖片的起始點 = 383(label)-212(圖片的角落的y)
    # print(rotate_hand_col, rotate_hand_row)
    if rotate_hand_col < 0 and rotate_hand_row < 0:
        reg_rotate_hand_col = rotate_hand_col
        rotate_hand_col = 0
        reg_rotate_hand_row = rotate_hand_row
        rotate_hand_row = 0
        rotate_hand = rotate_hand[abs(reg_rotate_hand_row):(drug_row - rotate_hand_row),
                      abs(reg_rotate_hand_col):(drug_col - rotate_hand_col)]

    elif rotate_hand_col >= 0 and rotate_hand_row >= 0:
        rotate_hand = rotate_hand[0:(drug_row - rotate_hand_row), 0:(drug_col - rotate_hand_col)]
        # cv.imshow(rotate_hand)

    elif rotate_hand_col < 0 and rotate_hand_row >= 0:
        reg_rotate_hand_col = rotate_hand_col
        rotate_hand_col = 0
        rotate_hand = rotate_hand[0:(drug_row - rotate_hand_row), abs(reg_rotate_hand_col):(drug_col - rotate_hand_col)]

    elif rotate_hand_col >= 0 and rotate_hand_row < 0:
        reg_rotate_hand_row = rotate_hand_row
        rotate_hand_row = 0
        rotate_hand = rotate_hand[abs(reg_rotate_hand_row):(drug_row - rotate_hand_row), 0:(drug_col - rotate_hand_col)]

    rows, cols, channels = rotate_hand.shape
    # cv.imshow('1',rotate_hand)
    # 建立插入圖片的區域
    roi = drug[rotate_hand_row:rows + rotate_hand_row, rotate_hand_col:cols + rotate_hand_col]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(rotate_hand, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 178, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Now black-out the area of logo in ROI 插圖區加上一層mask
    img1_bg = cv.bitwise_and(roi, roi, mask=mask)

    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(rotate_hand, rotate_hand, mask=mask_inv)
    # cv.imshow("5", img1_bg)

    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg, img2_fg)
    drug[rotate_hand_row:rows + rotate_hand_row, rotate_hand_col:cols + rotate_hand_col] = dst
    # cv.imshow('res', drug)  # 顯示結果圖片
    return drug


def hand_arc(point1, point2):
    gt_point1_x, gt_point1_y = point1
    gt_point2_x, gt_point2_y = point2
    dis_x = math.fabs(gt_point1_x - gt_point2_x)
    dis_y = math.fabs(gt_point1_y - gt_point2_y)
    radius = 180 / math.pi
    # print(dis_x, dis_y)
    if (gt_point1_x - gt_point2_x) >= 0 and (gt_point1_y - gt_point2_y) <= 0:
        arc = 90 - (math.atan(dis_y / dis_x) * radius)
        arc += 2
        style = 0
    elif (gt_point1_x - gt_point2_x) >= 0 and (gt_point1_y - gt_point2_y) >= 0:
        arc = 90 + (math.atan(dis_y / dis_x) * radius)
        arc += 2
        style = 1
    elif (gt_point1_x - gt_point2_x) <= 0 and (gt_point1_y - gt_point2_y) <= 0:
        arc = -90 + (math.atan(dis_y / dis_x) * radius)
        arc += 2
        style = 2
    elif (gt_point1_x - gt_point2_x) <= 0 and (gt_point1_y - gt_point2_y) >= 0:
        arc = -90 - (math.atan(dis_y / dis_x) * radius)
        arc += 2
        style = 3
    # print(style)
    return arc


def random_paste_with_hand(img, rect, bg, hand, random_scale, random_angle, side):
    # cv.imshow("1",label)
    bg = cv.resize(bg, (640, 480))
    label_img = np.zeros(img.shape, np.uint8)
    label_img = cv.fillPoly(label_img, [rect], (255, 255, 255), cv.LINE_AA)
    # cv.imshow("2", label_img)

    # label = label[:,:,1]
    # cv.imshow("1",img)
    # cv.imshow("2",label)
    label_not = cv.bitwise_not(label_img)
    crop = cv.bitwise_and(img, label_img)
    bg_crop = cv.bitwise_and(label_not, bg)
    new_img = cv.bitwise_or(crop, bg_crop)
    cv.imshow("8",crop)

    # 旋轉縮放任意拼貼
    # print(random_scale)
    rotate_crop = rotate(crop, angle=random_angle, scale=random_scale, white_bg=0)
    rotate_label_not = rotate(label_not, angle=random_angle, scale=random_scale, white_bg=1)
    rotate_bg_crop = cv.bitwise_and(rotate_label_not, bg)
    rotate_new_img = cv.bitwise_or(rotate_crop, rotate_bg_crop)
    rotate_label = cv.bitwise_not(rotate_label_not)
    blister_result = rotate_crop

    # cv.imshow("2", rotate_new_img)

    rect = img_label_sep(rotate_label)
    s_points_tmp = sortpts_clockwise(rect)
    rect = findright4points(s_points_tmp)
    # print(rect)

    if side == "R":
        arc_R = hand_arc(rect[3], rect[0])
        rotate_hand = hand_rotate(hand, -(arc_R))
        # cv.imshow("1",rotate_hand)
        # 加上手
        result = addhand(rotate_crop, rotate_hand, rect[0][0], rect[0][1])
        # cv.imshow("3", result)
    else:
        arc_L = hand_arc(rect[1], rect[2])
        rotate_hand = hand_rotate(hand, -(arc_L))
        # cv.imshow("1",rotate_hand)
        # 加上手
        result = addhand(rotate_crop, rotate_hand, rect[2][0], rect[2][1])
        # cv.imshow("3", result)
    # 旋轉手的角度

    img2gray1 = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray1, 0, 255, cv.THRESH_BINARY)
    mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    # cv.imshow("A", mask)

    mask_not = cv.bitwise_not(mask)
    mask_crop = cv.bitwise_and(img, mask)
    bg_crop = cv.bitwise_and(mask_not, bg)
    result = cv.bitwise_or(rotate_crop, bg_crop)
    # cv.imshow("2", blister_result)
    cv.waitKey(0)
    return crop, rotate_crop, result, rect


# img_R = cv.imread("./img/B1_ZEFFIX 100MG (LAMIVUDINE)/000001_R.jpg")
# gt_R = "./json/B1_ZEFFIX 100MG (LAMIVUDINE)/000001_R.json"
# img_L = cv.imread("./img/B1_ZEFFIX 100MG (LAMIVUDINE)/000001_L.jpg")
# gt_L = "./json/B1_ZEFFIX 100MG (LAMIVUDINE)/000001_L.json"
# bg = cv.imread("./NB/1.jpg")
img_src = "./img"
gt_src = "./json_new"
bg_src = "./NB"
hand = cv.imread("./HAND2.jpg")
bg_num = 1
im_num = 1

# img_dst = "F:/FastROR_synthesis/PS1/JPEGImages"
# gt_dst = "F:/FastROR_synthesis/PS1/Annotations"
# if not os.path.exists(img_dst):
#     os.makedirs(img_dst)
# if not os.path.exists(gt_dst):
#     os.makedirs(gt_dst)

cls = os.listdir(img_src)
for i, c in enumerate(cls):
    img_dir = "{}/{}".format(img_src, c)
    gt_dir = "{}/{}".format(gt_src, c)
    num = os.listdir(img_dir)
    print(c)
    for t in range(10):
        for im in range(1, 31):
            img_L_path = "{}/{:06}_R.jpg".format(img_dir, im)
            gt_L = "{}/{:06}_R.json".format(gt_dir, im)
            img_R_path = "{}/{:06}_L.jpg".format(img_dir, im)
            gt_R = "{}/{:06}_L.json".format(gt_dir, im)
            img_L = cv.imread(img_L_path)
            img_R = cv.imread(img_R_path)
            bg_L = cv.imread("{}/{}.jpg".format(bg_src, bg_num))
            bg_num += 1
            if bg_num == 2621:
                bg_num = 1
            bg_R = cv.imread("{}/{}.jpg".format(bg_src, bg_num))
            bg_num += 1
            if bg_num == 2621:
                bg_num = 1
            gt_rect_x = []
            gt_rect_y = []
            orgin_img = cv.vconcat((img_L, img_R))
            # cv.imshow("or",orgin_img)

            with open(gt_R, "r", encoding="utf-8") as reader:
                gt = json.loads(reader.read())
            gt_rect_R = np.array(gt[0]['poly'], np.int32)

            with open(gt_L, "r", encoding="utf-8") as reader:
                gt = json.loads(reader.read())
            gt_rect_L = np.array(gt[0]['poly'], np.int32)

            s_points_tmp = sortpts_clockwise(gt_rect_R)
            rect_R = findright4points(s_points_tmp)
            s_points_tmp = sortpts_clockwise(gt_rect_L)
            rect_L = findright4points(s_points_tmp)

            random_scale = random.uniform(0.9, 1.1)
            random_angle = random.randint(-45, 45)

            crop_R, crop_hand_R, result_R, rand_rect_R = random_paste_with_hand(img=img_R, rect=rect_R, bg=bg_R, hand=hand,
                                                           random_scale=random_scale,
                                                           random_angle=-random_angle, side="R")
            crop_L, crop_hand_L, result_L, rand_rect_L = random_paste_with_hand(img=img_L, rect=rect_L, bg=bg_L, hand=hand,
                                                           random_scale=random_scale,
                                                           random_angle=random_angle, side="L")
            crop = cv.vconcat((crop_L, crop_R))
            crop_hand = cv.vconcat((crop_hand_L, crop_hand_R))
            result = cv.vconcat((result_L, result_R))
            # cv.imwrite("{}/img_{}.jpg".format(img_dst, im_num), result)
            # dst_txt_file = "{}/gt_img_{}.txt".format(gt_dst, im_num)
            # # print(rand_rect_L, rand_rect_R)
            # with open(dst_txt_file, 'w') as f:
            #     f.write('{},{},{},{},{},{},{},{},{}\n'.format(
            #         rand_rect_L[0][0], rand_rect_L[0][1], rand_rect_L[1][0], rand_rect_L[1][1], rand_rect_L[2][0],
            #         rand_rect_L[2][1], rand_rect_L[3][0], rand_rect_L[3][1], str(i)
            #     ))
            #     f.write('{},{},{},{},{},{},{},{},{}'.format(
            #         rand_rect_R[0][0], str(int(rand_rect_R[0][1]) + 480), rand_rect_R[1][0],
            #         str(int(rand_rect_R[1][1]) + 480), rand_rect_R[2][0], str(int(rand_rect_R[2][1]) + 480),
            #         rand_rect_R[3][0], str(int(rand_rect_R[3][1]) + 480), str(i)
            #     ))
            # im_num += 1
            cv.imshow("result", result)
            cv.imshow("crop", crop)
            cv.imshow("crop_hand", crop_hand)
            cv.waitKey(0)
            cv.destroyAllWindows()
# print(rect)


# z = np.ones((4, 1), np.float32)
# max_area_point_combinations_float_tmp = np.concatenate((max_area_point_combinations_float, z), axis=1)
# max_area_point_combinations_float_tmp_trans = max_area_point_combinations_float_tmp.transpose()
# max_area_point_combinations_float_ori = np.matmul(M_resize2ori, max_area_point_combinations_float_tmp_trans)
# four_points_op_t = max_area_point_combinations_float_ori.transpose()
# four_points = np.array([[round((four_points_op_t[0][0] / four_points_op_t[0][2]), 1),
#                          round((four_points_op_t[0][1] / four_points_op_t[0][2]), 1)],
#                         [round((four_points_op_t[1][0] / four_points_op_t[1][2]), 1),
#                          round((four_points_op_t[1][1] / four_points_op_t[1][2]), 1)],
#                         [round((four_points_op_t[2][0] / four_points_op_t[2][2]), 1),
#                          round((four_points_op_t[2][1] / four_points_op_t[2][2]), 1)],
#                         [round((four_points_op_t[3][0] / four_points_op_t[3][2]), 1),
#                          round((four_points_op_t[3][1] / four_points_op_t[3][2]), 1)]], np.float32)
#
# ori_frame_size = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], np.float32)
# resize_frame_size = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], np.float32)
# M_resize2ori = cv2.getPerspectiveTransform(resize_frame_size, ori_frame_size)
