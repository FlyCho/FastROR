import cv2
from os import listdir, mkdir
import os
import numpy as np
myvideo = "./blister_test"
myimg = "./blister_test_img/"

def get_images_from_video(video_name, time_F):

    video_images = []
    vc = cv2.VideoCapture(video_name)
    c = 1

    if vc.isOpened():  # 判斷是否開啟影片
        rval, video_frame = vc.read()

    else:
        rval = False

    while rval:  # 擷取視頻至結束
        rval, video_frame = vc.read()
        if (c % time_F == 0):  # 每隔幾幀進行擷取
            video_images.append(video_frame)
        c = c + 1

    vc.release()

    return video_images



time_F = 5  # time_F越小，取樣張數越多

vid_files = listdir(myvideo)
print(vid_files)
im_len = []
# for vif in vid_files:
vif = "O1_10mg_Abilify_10mg"
myclass = myvideo + "/" + vif
video_name_L_back = myclass + "/L_cam_backward.mp4"
video_name_R_back = myclass + "/R_cam_backward.mp4"
video_name_L_for = myclass + "/L_cam_forward.mp4"
video_name_R_for = myclass + "/R_cam_forward.mp4"
# print(video_name_L)
video_images_L_back = get_images_from_video(video_name_L_back, time_F)
video_images_R_back = get_images_from_video(video_name_R_back, time_F)
video_images_L_for = get_images_from_video(video_name_L_for, time_F)
video_images_R_for = get_images_from_video(video_name_R_for, time_F)
# print(vif)
# cv2.imshow("1", video_images_R_back[1])
# cv2.imshow("2", video_images_L_back[1])
# cv2.waitKey(100)
L_dir_back = myimg + vif + '/L_back'
R_dir_back = myimg + vif + '/R_back'
if not os.path.exists(L_dir_back):
    os.makedirs(L_dir_back)
if not os.path.exists(R_dir_back):
    os.makedirs(R_dir_back)

L_dir_for = myimg + vif + '/L_for'
R_dir_for = myimg + vif + '/R_for'
if not os.path.exists(L_dir_for):
    os.makedirs(L_dir_for)
if not os.path.exists(R_dir_for):
    os.makedirs(R_dir_for)

for i in range(len(video_images_L_back)-1):  # 顯示出所有擷取之圖片
    cv2.imwrite(L_dir_back + '/img_' + str(i) + '.jpg', video_images_L_back[i])
    cv2.imwrite(R_dir_back + '/img_' + str(i) + '.jpg', video_images_R_back[i])
for i in range(len(video_images_L_for)-1):  # 顯示出所有擷取之圖片
    cv2.imwrite(L_dir_for + '/img_' + str(i) + '.jpg', video_images_L_for[i])
    cv2.imwrite(R_dir_for + '/img_' + str(i) + '.jpg', video_images_R_for[i])
im_len.append(len(video_images_L_back))
im_len.append(len(video_images_L_for))
print("{} suesscess".format(vif))


# video_name = 'D:/blister/Blister_test_video/blister_test/B1_ZEFFIX 100MG (LAMIVUDINE)/L_cam.mp4'  # 影片名稱
# video_images = get_images_from_video(video_name, time_F)  # 讀取影片並轉成圖片
#
# for i in range(0, len(video_images)):  # 顯示出所有擷取之圖片
#     # cv2.imshow('windows', video_images[i])
#     cv2.imwrite('./blister_test_img/img_'+str(i)+'.jpg',video_images[i])
#     cv2.waitKey(100)
print("min_img:" + str(min(im_len)))
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()