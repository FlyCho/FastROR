import random
import os
from shutil import copy2
import cv2
# from scipy import *
from os import listdir

myimg = "./blister_test_img"
myrand = "./rand50_img/"
if not os.path.exists(os.path.dirname(myrand)):
    os.makedirs(os.path.dirname(myrand))

# list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# slice = random.sample(list,5)
# print(slice)

vid_files = listdir(myimg)
print(vid_files)
num = 1
for vif in vid_files:

    myclass = myimg + "/" + vif
    print(myclass)
    L_b_file = myclass + "/L_back"
    R_b_file = myclass + "/R_back"
    L_b_img = listdir(L_b_file)

    L_b_rand = random.sample(L_b_img, 50)
    num = 1
    for dir in L_b_rand:
        L_b_img_src = L_b_file + "/" + str(dir)
        R_b_img_src = R_b_file + "/" + str(dir)

        L_b_img = cv2.imread(L_b_img_src)
        R_b_img = cv2.imread(R_b_img_src)
        # img_pair = cv2.vconcat([L_b_img, R_b_img])
        L_back_dst_dir = "{}{}/L_back/".format(myrand, vif)
        if not os.path.exists(os.path.dirname(L_back_dst_dir)):
            os.makedirs(os.path.dirname(L_back_dst_dir))
        R_back_dst_dir = "{}{}/R_back/".format(myrand, vif)
        if not os.path.exists(os.path.dirname(R_back_dst_dir)):
            os.makedirs(os.path.dirname(R_back_dst_dir))
        cv2.imwrite("{}img_{}.jpg".format(L_back_dst_dir, num), L_b_img)
        cv2.imwrite("{}img_{}.jpg".format(R_back_dst_dir, num), R_b_img)
        num += 1

    L_f_file = myclass + "/L_for"
    R_f_file = myclass + "/R_for"
    L_f_img = listdir(L_f_file)

    L_f_rand = random.sample(L_f_img, 50)
    num = 1
    for dir in L_f_rand:
        L_f_img_src = L_f_file + "/" + str(dir)
        R_f_img_src = R_f_file + "/" + str(dir)

        L_f_img = cv2.imread(L_f_img_src)
        R_f_img = cv2.imread(R_f_img_src)
        # img_pair = cv2.vconcat([L_f_img, R_f_img])
        L_for_dst_dir = "{}{}/L_for/".format(myrand, vif)
        if not os.path.exists(os.path.dirname(L_for_dst_dir)):
            os.makedirs(os.path.dirname(L_for_dst_dir))
        R_for_dst_dir = "{}{}/R_for/".format(myrand, vif)
        if not os.path.exists(os.path.dirname(R_for_dst_dir)):
            os.makedirs(os.path.dirname(R_for_dst_dir))
        cv2.imwrite("{}img_{}.jpg".format(L_for_dst_dir, num), L_f_img)
        cv2.imwrite("{}img_{}.jpg".format(R_for_dst_dir, num), R_f_img)
        num += 1
