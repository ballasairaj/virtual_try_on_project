from doctest import master
import os, sys
import cv2
from PIL import Image
import numpy as np
import glob
import warnings
import argparse
from cloths_segmentation.pre_trained_models import create_model
import time
import time
import subprocess
from get_cloth_mask import main_get_cloth_mask
from posenet_exe import process_posenet_main
from get_seg_grayscale import seg_gray_scale_main

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--background', type=bool, default=True, help='Define removing background or not')
    opt = parser.parse_args()

    # Read input image
    img=cv2.imread("./static/origin_web.jpg")
    ori_img=cv2.resize(img,(768,1024))
    cv2.imwrite("./origin.jpg",ori_img)
    
    # img=cv2.imread("./static/cloth_web.jpg")
    # ori_img=cv2.resize(img,(768,1024))
    # cv2.imwrite("./static/cloth_web.jpg",ori_img)


    # Resize input image
    img=cv2.imread('origin.jpg')
    img=cv2.resize(img,(384,512))
    cv2.imwrite('resized_img.jpg',img)
    
    
    # Get mask of cloth
    print("Get mask of cloth\n")
    # time_mask_of_cloth1 = time.time()
    # terminnal_command = "python get_cloth_mask.py" 
    # os.system(terminnal_command)
    # total_time_mask_of_cloth1 = time.time() - time_mask_of_cloth1
    # print(f"Total execution time for mask of cloth1: {total_time_mask_of_cloth1:.2f} seconds")
    time_mask_of_cloth2 = time.time()
    main_get_cloth_mask()
    total_time_mask_of_cloth2 = time.time() - time_mask_of_cloth2
    print(f"Total execution time for mask of cloth2: {total_time_mask_of_cloth2:.2f} seconds")

    # Get openpose coordinate using posenet
    total_time_posenet = time.time()
    print("Get openpose coordinate using posenet\n")
    process_posenet_main()
    total_time_posenet = time.time() - total_time_posenet
    print(f"Total execution time for posenet: {total_time_posenet:.2f} seconds")

    # Generate semantic segmentation using Graphonomy-Master library
    total_time_graphonomy = time.time()
    print("Generate semantic segmentation using Graphonomy-Master library\n")
    os.chdir("./Graphonomy_master")
    # sys.path.append("./Graphonomy_master")
    from Graphonomy_master.exp.inference.inference import graphonomy_main
    graphonomy_main(r'D:\VTO\inference.pth', r'D:\VTO\resized_img.jpg', r'../', 'resized_segmentation_img')
    # terminnal_command ="python exp/inference/inference.py --loadmodel ../inference.pth --img_path ../resized_img.jpg --output_path ../ --output_name /resized_segmentation_img"
    # os.system(terminnal_command)
    os.chdir("../")
    total_time_graphonomy = time.time() - total_time_graphonomy
    print(f"Total execution time for Graphonomy: {total_time_graphonomy:.2f} seconds")

    segmentation_mask_time = time.time()
    # Remove background image using semantic segmentation mask
    mask_img=cv2.imread('./resized_segmentation_img.png',cv2.IMREAD_GRAYSCALE)
    mask_img=cv2.resize(mask_img,(768,1024))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_img = cv2.erode(mask_img, k)
    img_seg=cv2.bitwise_and(ori_img,ori_img,mask=mask_img)
    back_ground=ori_img-img_seg
    img_seg=np.where(img_seg==0,215,img_seg)
    cv2.imwrite("./seg_img.png",img_seg)
    img=cv2.resize(img_seg,(768,1024))
    cv2.imwrite('./HR_VITON_main/test/test/image/00001_00.jpg',img)
    time_segmentation_mask = time.time() - segmentation_mask_time
    print(f"Total execution time for Segmentation Mask: {time_segmentation_mask:.2f} seconds")


    # Generate grayscale semantic segmentation image
    total_time_semantic_segmentation = time.time()
    # terminnal_command ="python get_seg_grayscale.py"
    # os.system(terminnal_command)
    seg_gray_scale_main()
    total_time_semantic_segmentation = time.time() - total_time_semantic_segmentation
    print(f"Total execution time for Semantic Segmentation: {total_time_semantic_segmentation:.2f} seconds")

    # exit('DONEEEE')
    # os.chdir(r"D:\VTO")
    # Generate Densepose image using detectron2 library
    total_time_detectron2 = time.time()
    print("\nGenerate Densepose image using detectron2 library\n")
    terminnal_command ="python detectron2/projects/DensePose/apply_net.py dump detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
    https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
    origin.jpg --output output.pkl -v"
    os.system(terminnal_command)
    total_time_detectron2 = time.time() - total_time_detectron2
    print(f"Total execution time for Detectron2: {total_time_detectron2:.2f} seconds")

    # terminnal_command ="python get_densepose.py" #??????????????????
    # os.system(terminnal_command)

    # Run HR-VITON to generate final image
    total_time_hrviton = time.time()
    print("\nRun HR-VITON to generate final image\n")
    os.chdir("./HR_VITON_main")
    # sys.path.append(r'.\HR_VITON_main')
    # sys.path.append(r'D:\VTO\HR_VITON_main')

    # from test_generator import main
    # from HR_VITON_main.test_generator import main
    # main('test1', 'eval_models/weights/v0.1/mtviton.pth', '0', 'eval_models/weights/v0.1/gen.pth', 'unpaired', r"D:\VTO\HR_VITON_main\test\t2.txt", './test')

    # main(
    #     'test1',
    #     r"D:\VTO\HR_VITON_main\eval_models\weights\v0.1\mtviton.pth",  # model weights
    #     '0',  # GPU id (or 'cpu')
    #     r"D:\VTO\HR_VITON_main\eval_models\weights\v0.1\gen.pth",     # generator weights
    #     'unpaired',
    #     r"D:\VTO\HR_VITON_main\test\t2.txt",                          # data list file
    #     r"D:\VTO\HR_VITON_main\test"                                  # output folder
    # )
    terminnal_command = "python test_generator.py --test_name test1 --tocg_checkpoint eval_models/weights/v0.1/mtviton.pth --gpu_ids 0 --gen_checkpoint eval_models/weights/v0.1/gen.pth --datasetting unpaired --data_list t2.txt --dataroot ./test" 
    os.system(terminnal_command)
    total_time_hrviton = time.time() - total_time_hrviton
    print(f"Total execution time for HR-VITON: {total_time_hrviton:.2f} seconds")
    # Add Background or Not
    l=glob.glob("./Output/*.png")
    # Add Background
    if opt.background:
        for i in l:
            img=cv2.imread(i)
            img=cv2.bitwise_and(img,img,mask=mask_img)
            img=img+back_ground
            cv2.imwrite(i,img)

    # Remove Background
    else:
        for i in l:
            img=cv2.imread(i)
            cv2.imwrite(i,img)

    os.chdir("../")
    cv2.imwrite("./static/finalimg.png", img)
    # Total execution time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")