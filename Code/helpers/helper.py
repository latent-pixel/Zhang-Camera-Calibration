import numpy as np
import cv2
import scipy
import os
import copy


# Resizing the image
def resize_image(image_array, scale=1):
    scaled = cv2.resize(image_array, (0,0), fx=scale, fy=scale)
    return scaled 


# Collecting all the images and placing them into an array
def getImageStack(image_folder):
    image_stack = list()
    for file_name in os.listdir(image_folder):
        if ".jpg" in file_name:
            file_path = os.path.join(image_folder, file_name)   
            image = cv2.imread(file_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_stack.append(image)
    return image_stack


# Getting the corners from images
def getImageCorners(calib_images, save_path="/initial_outputs/", save=False):
    corner_stack = list()
    for i in range(len(calib_images)):
        gray = cv2.cvtColor(calib_images[i], cv2.COLOR_BGR2GRAY)
        X_n, Y_n = 9, 6
        pattern_found, det_corners = cv2.findChessboardCorners(gray, (X_n, Y_n), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if pattern_found:
            corners = cv2.cornerSubPix(gray, det_corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            corners = corners.reshape(-1, 2)
            corner_stack.append(corners)
            
        if save == True:
            image_copy = copy.deepcopy(calib_images[i])
            cv2.drawChessboardCorners(image_copy, (X_n, Y_n), corners, True)
            image_copy = resize_image(image_copy, 1/3)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path+str(i)+'.jpg', image_copy)
    return corner_stack            


# Getting the world coordinates
def getWorldCorners(edge, n_X, n_Y):
    x_vals = [edge*int(val) for val in np.linspace(0, n_X-1, n_X)]
    y_vals = [edge*int(val) for val in np.linspace(0, n_Y-1, n_Y)]
    w_corners = list()
    for yval in y_vals:
        for xval in x_vals:
            w_corners.append((xval, yval))    
    w_corners = np.stack(w_corners)
    # print(w_corners)
    return w_corners


def get_Arows(u, v, X, Y):
    """Returns the rows for A matrix given a point correspondence

    Args:
        u, v (float): Image coordinates along x and y axes 
        X, Y (float): World coordinates along x and y axes

    Returns:
        list: 2 lists pertaining to the correspondence constraints
    """
    row1 = [X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u]
    row2 = [0, 0, 0, X, Y, 1, -v*X, -v*Y, -v]
    return row1, row2


# Finding the homography between world coordinates and image coordinates
def findAllHomographies(world_coords, image_coords):
    all_homographies = list()
    X = world_coords[:, 0]
    Y = world_coords[:, 1]
    for corner_set in image_coords:
        A = list()
        u = corner_set[:, 0]
        v = corner_set[:, 1]
        for i in range(len(corner_set)):
            row1, row2 = get_Arows(u[i], v[i], X[i], Y[i])
            A.append(row1)
            A.append(row2)
        
        _, _, V = np.linalg.svd(np.array(A), full_matrices=True)
        # print(V)
        H = V[-1, :].reshape((3, 3))
        H = H / H[2,2]
        all_homographies.append(H)
        
    return all_homographies
    

def get_Vrow(h_i, h_j):
    """Returns a row for the V matrix given a homographt

    Args:
        homography_mtrx (array): Homography matrix corresponding to a pair of images

    Returns:
        list: v_ij
    """
    vrow = [h_i[0]*h_j[0], h_i[0]*h_j[1]+h_i[1]*h_j[0], h_i[1]*h_j[1], h_i[2]*h_j[0]+h_i[0]*h_j[2], h_i[2]*h_j[1]+h_i[1]*h_j[2], h_i[2]*h_j[2]]
    return np.transpose(vrow)


def get_V_matrix(homography_list):   # Can be merged with findB ---> Check again!
    V = list()
    for H in homography_list:
        h1 = H[:, 0]
        h2 = H[:, 1]
        v_12 = get_Vrow(h1, h2)
        v_11 = get_Vrow(h1, h1)
        v_22 = get_Vrow(h2, h2)
        V.append(np.transpose(v_12))
        V.append(np.transpose(v_11 - v_22))
    return np.array(V)


def findB(homography_list):
    V_matrix = get_V_matrix(homography_list)
    # print(V_matrix)
    _, _, V = np.linalg.svd(V_matrix)
    B11 , B12 , B22, B13 , B23, B33 = V[-1, :]
    B = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])
    return B


def get_IntrinsicMatrix(B):
    # print(B)
    v0 = (B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2])/(B[0, 0]*B[1, 1] - B[0, 1]**2)
    lamda = B[2, 2] - (B[0, 2]**2 + v0*(B[0, 1]*B[0, 2] - B[0, 0]*B[1, 2]))/B[0, 0]
    a = np.sqrt(lamda/B[0, 0])
    b = np.sqrt(lamda*B[0, 0]/(B[0, 0]*B[1, 1] - B[0, 1]**2))
    r = -B[0, 1]*a*a*b/lamda
    u0 = (r*v0/b) - (B[0, 2]*a*a/lamda)
    K = np.array([[a, r, u0], [0, b, v0], [0, 0, 1]])
    return K

    
def get_ExtrinsicMatrix(K, homography_list):
    all_Rt = []
    for H in homography_list:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lamda = 1/np.linalg.norm(np.dot(np.linalg.inv(K), h1), 2)

        r1 = lamda*np.dot(np.linalg.inv(K), h1)
        r2 = lamda*np.dot(np.linalg.inv(K), h2)
        r3 = np.cross(r1, r2)
        t = lamda*np.dot(np.linalg.inv(K), h3)
        Rt = np.vstack((r1, r2, r3, t)).T
        all_Rt.append(Rt)

    return all_Rt  


def get_ReprojectionError(intrinsic_params, all_extrinsic_matrices, all_image_coords, world_coords, ret_reprojection=False):
    
    a, r, b, u0, v0, k1, k2 = intrinsic_params
    
    A = np.array([[a, r, u0], [0, b, v0], [0, 0, 1]])
    k_c = np.array([k1, k2])
    # print(A)
    
    all_errors = list()
    all_reprojected_coords = list()
    for i in range(len(all_image_coords)):
        image_coords = all_image_coords[i]
        Rt = all_extrinsic_matrices[i]

        Rt_Z = np.array([Rt[:, 0], Rt[:, 1], Rt[:, 3]]).reshape(3, 3).T
        projection_Z = np.matmul(A, Rt_Z)

        error = 0
        reprojected_coords = list()
        for j in range(len(image_coords)):
            homogeneous_image_coords = np.array([image_coords[j][0], image_coords[j][1], 1]).T

            homogeneous_world_coords = np.array([world_coords[j][0], world_coords[j][1], 1]).T
            homogeneous_world_coords_3D = np.array([world_coords[j][0], world_coords[j][1], 0, 1]).T
            transformed_point = np.matmul(Rt, homogeneous_world_coords_3D)

            x = transformed_point[0]/transformed_point[2]
            y = transformed_point[1]/transformed_point[2]

            projected_image_coords = np.matmul(projection_Z, homogeneous_world_coords)

            u = projected_image_coords[0]/projected_image_coords[2]
            v = projected_image_coords[1]/projected_image_coords[2]
            
            u_d = u + (u - u0)*(k1*(x**2 + y**2) + k2*(x**2 + y**2)**2)     # radial distortion
            v_d = v + (v - v0)*(k1*(x**2 + y**2) + k2*(x**2 + y**2)**2)
            
            estimated_image_coords = np.array([u_d, v_d, 1]).T
            reprojected_coords.append(estimated_image_coords)
            error += np.linalg.norm((homogeneous_image_coords - estimated_image_coords))
            
        all_errors.append(error/len(image_coords))
        all_reprojected_coords.append(reprojected_coords)
    # print(all_errors) 
    if ret_reprojection:
        return np.mean(all_errors), np.array(all_reprojected_coords, dtype=np.float32)
    else:
        return all_errors
