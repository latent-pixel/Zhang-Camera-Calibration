import numpy as np
import cv2
from scipy.optimize import least_squares
import os
import argparse
import helpers.helper as helper 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--ImageDirectory', default="Calibration_Imgs/", help="Folder where the calibration images are stored")
    parser.add_argument('-s', '--SavePath', default="Undistorted_Output/", help="Folder where the output images are to be stored")
    
    args = parser.parse_args()
    input_dir = args.ImageDirectory
    save_path = args.SavePath
    
    image_stack = helper.getImageStack(input_dir)
    all_image_coords = helper.getImageCorners(image_stack)
    # print(all_image_coords[0])
    world_coords = helper.getWorldCorners(edge=21.5, n_X=9, n_Y=6)
    all_homographies = helper.findAllHomographies(world_coords, all_image_coords)
    B = helper.findB(all_homographies)
    A = helper.get_IntrinsicMatrix(B) # Calibration matrix
    print("Initial estimate for the A matrix: \n", A)
    all_Rt = helper.get_ExtrinsicMatrix(A, all_homographies)    
    k_c = np.array([0, 0]).T
    a, r, b, u0, v0 = A[0, 0], A[0,  1], A[1, 1], A[0, 2], A[1, 2]
    intrinsic_params = np.array([a, r, b, u0, v0, k_c[0], k_c[1]])
    mean_error = helper.get_ReprojectionError(intrinsic_params, all_Rt, all_image_coords, world_coords, ret_reprojection=True)[0]
    print("Error before optimization: ", mean_error)
    res = least_squares(fun=helper.get_ReprojectionError, x0=intrinsic_params, method="lm", args=[all_Rt, all_image_coords, world_coords])
    
    intrinsic_params_new = res.x
    a_new, r_new, b_new, u0_new, v0_new, k1_new, k2_new = intrinsic_params_new
    A_new = np.array([[a_new, r_new, u0_new], [0, b_new, v0_new], [0, 0, 1]])
    print("A matrix after optimization: \n", A_new)
    kc_new = np.array([k1_new, k2_new]).T
    print("Distortion coefficients obtained: ", kc_new)
    
    all_Rt_new = helper.get_ExtrinsicMatrix(A_new, all_homographies)
    
    mean_error_new = helper.get_ReprojectionError(intrinsic_params_new, all_Rt_new, all_image_coords, world_coords, True)[0]
    all_reprojected_pts = helper.get_ReprojectionError(intrinsic_params_new, all_Rt_new, all_image_coords, world_coords, True)[1]
    print("Mean error after optimization: ", mean_error_new)
    
    D = np.array([kc_new[0], kc_new[1], 0, 0] , np.float32)
    for i in range(len(image_stack)):
        curr_img = image_stack[i]
        reprojected_pts = all_reprojected_pts[i]
        curr_img = cv2.undistort(curr_img, A_new, D)
        for pt in reprojected_pts:
            # print(pt)
            curr_img = cv2.circle(curr_img, (int(pt[0]), int(pt[1])), 8, (149, 0, 255), 2)
        # cv2.imshow("Corrected", curr_img)
        # cv2.waitKey(0)
        cv2.imwrite(save_path+str(i)+"_corr.jpg", curr_img)
    # cv2.destroyAllWindows()
        
    
if __name__ == "__main__":
    main()