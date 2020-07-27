import cv2
import numpy as np
import sys
import os

test_vid = "./blister_test"
blister_check_example_img_input_path = "./blister_ex"
blister_pack_names_input_path = "./blister_names.txt"

if not os.path.exists(test_vid):
    os.mkdir(test_vid)
blister_names = []
f = open(blister_pack_names_input_path, 'r')
for l in f:
    l = l.strip('\n')
    blister_names.append(l)
f.close()

# =============================================================================
check_class_index = 0
# =============================================================================

RTT_imread_index = 1
start_vedio = 0
count = 0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

blister_class = test_vid + "/" + blister_names[check_class_index - 1]
if not os.path.exists(blister_class):
    os.mkdir(blister_class)
out0 = cv2.VideoWriter(blister_class + "/L_cam_forward.mp4", fourcc, 30.0, (640, 480))
out1 = cv2.VideoWriter(blister_class + "/R_cam_forward.mp4", fourcc, 30.0, (640, 480))
out2 = cv2.VideoWriter(blister_class + "/L_cam_backward.mp4", fourcc, 30.0, (640, 480))
out3 = cv2.VideoWriter(blister_class + "/R_cam_backward.mp4", fourcc, 30.0, (640, 480))
print("check_class_index :", check_class_index)
print("class name : ", blister_names[0])

while True:
    blister_ex_img = cv2.imread("{}/{}.jpg".format(blister_check_example_img_input_path, check_class_index))
    # blister_ex_img_win_name = "{}_{}".format(blister_names[check_class_index - 1], check_class_index)
    blister_ex_img_win_name = blister_names[check_class_index - 1]
    RTT_ori_img_win_name = "{}.jpg".format(RTT_imread_index)
    cv2.imshow(blister_ex_img_win_name, blister_ex_img)
    cv2.moveWindow(blister_ex_img_win_name, 1100, 550)
    ret0, frame0 = cap0.read()
    assert ret0
    ret1, frame1 = cap1.read()
    assert ret1

    key = cv2.waitKey(1) & 0xFF

    result = np.concatenate((frame1, frame0), axis=1)
    cv2.imshow('Video', result)
    cv2.moveWindow('Video', 0, 0)

    if key % 256 == 27:
        print("Escape hit, closing...")
        print("check_class_index :", check_class_index)
        break
    elif key in [ord('e') or ord('E')]:  # next blister_pack_ex
        if count ==1:
            print("please take the backward video!!")
        # elif count ==0:
        #     print("please take the video")
        else:
            cv2.destroyWindow(blister_ex_img_win_name)
            check_class_index += 1
            if check_class_index > len(os.listdir(blister_check_example_img_input_path)):
                # check_class_index = 1
                print("finish~~~~")
                break
            blister_ex_img_win_name = blister_names[check_class_index - 1]
            blister_class = test_vid + "/" + blister_ex_img_win_name
            if not os.path.exists(blister_class):
                os.mkdir(blister_class)
            out0 = cv2.VideoWriter(blister_class + "/L_cam_forward.mp4", fourcc, 30.0, (640, 480))
            out1 = cv2.VideoWriter(blister_class + "/R_cam_forward.mp4", fourcc, 30.0, (640, 480))
            out2 = cv2.VideoWriter(blister_class + "/L_cam_backward.mp4", fourcc, 30.0, (640, 480))
            out3 = cv2.VideoWriter(blister_class + "/R_cam_backward.mp4", fourcc, 30.0, (640, 480))
            print("class index = " + blister_class + "/cam.mp4")
            print("check_class_index :", check_class_index)
            print("class name : ", blister_ex_img_win_name)


    elif key in [ord('w') or ord('W')]:  # pre blister_pack_ex
        cv2.destroyWindow(blister_ex_img_win_name)
        check_class_index -= 1
        if check_class_index < 1:
            check_class_index = len(os.listdir(blister_check_example_img_input_path))
    elif key in [ord('d') or ord('D')]:
        if start_vedio == 1:
            print("press wrong key please press s")
        else:
            start_vedio = 1
            count += 1
            print("start taking vedio")


    elif key in [ord('s') or ord('S')]:
        if start_vedio ==1:
            start_vedio = 0
        if count ==1:
            print("end taking vedio forward video")
        if count == 2:
            print("end taking vedio backward video")
            count = 0

    elif key in [ord('f') or ord('F')]:
        cv2.imwrite(blister_class + "/L_cam.jpg", frame0)
        cv2.imwrite(blister_class + "/R_cam.jpg", frame1)
        print("write the background picture!")

    if count == 1:
        if start_vedio == 1:
            out0.write(frame0)
            out1.write(frame1)

    if count == 2:
        if start_vedio == 1:
            out2.write(frame0)
            out3.write(frame1)

# When everything is done, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
