import cv2
import numpy as np
import random
import math
import sys
import os

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Stitcher:
    def __init__(self):
        pass

    def stitch(self, imgs, grays, SIFT_Detector, threshold = 0.75, blend = 'linearBlendingWithConstantWidth'):
        # Step1 - extract the keypoints and features by SIFT
        key_points_1, descriptors_1 = SIFT_Detector.detectAndCompute(grays[0], None)
        key_points_2, descriptors_2 = SIFT_Detector.detectAndCompute(grays[1], None)
        
        # Step2 - extract the match point with threshold (David Lowe's ratio test)
        print('start match key points')
        matches = self.matchKeyPoint(key_points_1, descriptors_1, key_points_2, descriptors_2, threshold)
        print('finish match key points')
        
        # Step3 - fit the homography model with RANSAC algorithm
        print('start RANSAC')
        H = self.RANSAC_get_H(matches)
        print('finish RANSAC')

        # Step4 - Warp image to create panoramic image
        print('start warp image')
        warp_img = self.warp(imgs[0], imgs[1], H, blend)
        print('finish warp image')
        
        return warp_img

    def matchKeyPoint(self, kps_1, features_1, kps_2, features_2, threshold):
        '''
        Match the Keypoints beteewn two image
        '''
        # matches = []
        # for i in range(len(features_1)):
        #     min_index, min_distance = -1, np.inf
        #     sec_index, sec_distance = -1, np.inf
            
        #     for j in range(len(features_2)):
        #         distance = np.linalg.norm(features_1[i] - features_2[j])
                
        #         if distance < min_distance:
        #             sec_index, sec_distance = min_index, min_distance
        #             min_index, min_distance = j, distance
                    
        #         elif distance < sec_distance and sec_index != min_index:
        #             sec_index, sec_distance = j, distance
                    
        #     matches.append([min_index, min_distance, sec_index, sec_distance])

        # good_matches = []
        # for i in range(len(matches)):
        #     if matches[i][1] <= matches[i][3] * threshold:
        #         good_matches.append([(int(kps_1[i].pt[0]), int(kps_1[i].pt[1])), 
        #                              (int(kps_2[matches[i][0]].pt[0]), int(kps_2[matches[i][0]].pt[1]))])


        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(features_1, features_2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        good_matches = []
        for m in matches:
            if m.distance < 300:
                good_matches.append([(int(kps_1[m.queryIdx].pt[0]), int(kps_1[m.queryIdx].pt[1])), 
                                    (int(kps_2[m.trainIdx].pt[0]), int(kps_2[m.trainIdx].pt[1]))])
        

        return good_matches
    
    def RANSAC_get_H(self, matches):
        img1_kp = []
        img2_kp = []
        for kp1, kp2 in matches:
            img1_kp.append(list(kp1))
            img2_kp.append(list(kp2))
        img1_kp = np.array(img1_kp)
        img2_kp = np.array(img2_kp)
        
        # homography = Homography()
        # threshold = 5
        # iteration_num = 2000
        # max_inliner_num = 0
        # best_H = None
        
        # for iter in range(iteration_num):
        #     random_sample_idx = random.sample(range(len(matches)), 4)
        #     H = homography.solve_homography(img1_kp[random_sample_idx], img2_kp[random_sample_idx])

        #     # find the best Homography have the the maximum number of inlier
        #     inliner_num = 0
            
        #     for i in range(len(matches)):
        #         if i not in random_sample_idx:
        #             concateCoor = np.hstack((img1_kp[i], [1])) # add z-axis as 1
        #             dstCoor = H @ concateCoor.T
                    
        #             # avoid divide zero number, or too small number cause overflow
        #             if dstCoor[2] <= 1e-8: 
        #                 continue
                    
        #             dstCoor = dstCoor / dstCoor[2]
        #             if (np.linalg.norm(dstCoor[:2] - img2_kp[i]) < threshold):
        #                 inliner_num = inliner_num + 1
            
        #     if (max_inliner_num < inliner_num):
        #         max_inliner_num = inliner_num
        #         best_H = H

        # Use OpenCV's findHomography function
        best_H, _ = cv2.findHomography(img1_kp, img2_kp, cv2.RANSAC, 5.0)

        return best_H
                
    def warp(self, img1, img2, H, blendType):
        left_down = np.hstack(([0], [0], [1]))
        left_up = np.hstack(([0], [img1.shape[0]-1], [1]))
        right_down = np.hstack(([img1.shape[1]-1], [0], [1]))
        right_up = np.hstack(([img1.shape[1]-1], [img1.shape[0]-1], [1]))
        
        warped_left_down = H @ left_down.T
        warped_left_up = H @ left_up.T
        warped_right_down =  H @ right_down.T
        warped_right_up = H @ right_up.T

        x1 = int(min(min(min(warped_left_down[0],warped_left_up[0]),min(warped_right_down[0], warped_right_up[0])), 0))
        y1 = int(min(min(min(warped_left_down[1],warped_left_up[1]),min(warped_right_down[1], warped_right_up[1])), 0))
        size = (img2.shape[1] + abs(x1), img2.shape[0] + abs(y1))

        A = np.float32([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]])
        warped1 = cv2.warpPerspective(src=img1, M=A@H, dsize=size)
        warped2 = cv2.warpPerspective(src=img2, M=A, dsize=size)
        
        blender = Blender()
        print('start blending')
        if blendType == 'linearBlendingWithConstantWidth':
            result = blender.linearBlendingWithConstantWidth([warped1, warped2])
        elif blendType == 'MultiBandBlending':
            result = blender.MultiBandBlending([warped1, warped2])
        else:
            result = blender.Blending([warped1, warped2])
        
        return result

class Blender:
    def __init__(self):
        pass
    
    def Blending(self, imgs):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]

        # # create the mask
        # img_left_mask = np.zeros((hr, wr), dtype="int")
        # img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # # find the left image and right image mask region(Those not zero pixels)
        # for i in range(hl):
        #     for j in range(wl):
        #         if np.count_nonzero(img_left[i, j]) > 0:
        #             img_left_mask[i, j] = 1
        # for i in range(hr):
        #     for j in range(wr):
        #         if np.count_nonzero(img_right[i, j]) > 0:
        #             img_right_mask[i, j] = 1
        
        # # find the overlap mask(overlap region of two image)
        # overlap_mask = np.zeros((hr, wr), dtype="int")
        # for i in range(hr):
        #     for j in range(wr):
        #         if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
        #             overlap_mask[i, j] = 1
        
        # # compute the alpha mask to linear blending the overlap region
        # alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        # for i in range(hr): 
        #     minIdx = maxIdx = -1
        #     for j in range(wr):
        #         if (overlap_mask[i, j] == 1 and minIdx == -1):
        #             minIdx = j
        #         if (overlap_mask[i, j] == 1):
        #             maxIdx = j
            
        #     if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
        #         continue
                
        #     decrease_step = 1 / (maxIdx - minIdx)
        #     for j in range(minIdx, maxIdx + 1):
        #         alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        # blending_img = np.copy(img_right)
        # blending_img[:hl, :wl] = np.copy(img_left)
        # # linear blending
        # for i in range(hr):
        #     for j in range(wr):
        #         if ( np.count_nonzero(overlap_mask[i, j]) > 0):
        #             blending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        #         elif(np.count_nonzero(img_left_mask[i, j]) > 0):
        #             blending_img[i, j] = img_left[i, j]
        #         else:
        #             blending_img[i, j] = img_right[i, j]

        # Create masks
        img_left_mask = np.any(img_left > 0, axis=-1)
        img_right_mask = np.any(img_right > 0, axis=-1)

        # Find overlap
        overlap_mask = img_left_mask & img_right_mask

        # Compute alpha mask
        rows, cols = np.where(overlap_mask)
        min_cols = np.min(cols)
        max_cols = np.max(cols)
        alpha_mask = np.zeros((hr, wr))
        alpha_mask[overlap_mask] = 1 - (cols - min_cols) / (max_cols - min_cols)

        # Create blending image
        blending_img = np.copy(img_right)
        blending_img[img_left_mask] = img_left[img_left_mask]  # Copy non-overlapping region of left image

        # Apply linear blending
        blending_mask = overlap_mask & img_right_mask
        blending_img[blending_mask] = alpha_mask[blending_mask, None] * img_left[blending_mask] + (1 - alpha_mask[blending_mask, None]) * img_right[blending_mask]
        
        return blending_img

    def linearBlendingWithConstantWidth(self, imgs):
        '''
        linear Blending with Constat Width, avoiding ghost region
        # you need to determine the size of constant with
        '''
        img_left, img_right = imgs
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        constant_width = 10 # constant width
        
        # # find the left image and right image mask region(Those not zero pixels)
        # for i in range(hl):
        #     for j in range(wl):
        #         if np.count_nonzero(img_left[i, j]) > 0:
        #             img_left_mask[i, j] = 1
        # for i in range(hr):
        #     for j in range(wr):
        #         if np.count_nonzero(img_right[i, j]) > 0:
        #             img_right_mask[i, j] = 1
                    
        # # find the overlap mask(overlap region of two image)
        # overlap_mask = np.zeros((hr, wr), dtype="int")
        # for i in range(hr):
        #     for j in range(wr):
        #         if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
        #             overlap_mask[i, j] = 1
        
        # # compute the alpha mask to linear blending the overlap region
        # alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        # for i in range(hr):
        #     minIdx = maxIdx = -1
        #     for j in range(wr):
        #         if (overlap_mask[i, j] == 1 and minIdx == -1):
        #             minIdx = j
        #         if (overlap_mask[i, j] == 1):
        #             maxIdx = j
            
        #     if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
        #         continue
                
        #     decrease_step = 1 / (maxIdx - minIdx)
            
        #     # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
        #     middleIdx = int((maxIdx + minIdx) / 2)
            
        #     # left 
        #     for j in range(minIdx, middleIdx + 1):
        #         if (j >= middleIdx - constant_width):
        #             alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        #         else:
        #             alpha_mask[i, j] = 1
        #     # right
        #     for j in range(middleIdx + 1, maxIdx + 1):
        #         if (j <= middleIdx + constant_width):
        #             alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        #         else:
        #             alpha_mask[i, j] = 0

        
        # linearBlendingWithConstantWidth_img = np.copy(img_right)
        # linearBlendingWithConstantWidth_img[:hl, :wl] = np.copy(img_left)
        # # linear blending with constant width
        # for i in range(hr):
        #     for j in range(wr):
        #         if (np.count_nonzero(overlap_mask[i, j]) > 0):
        #             linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        #         elif(np.count_nonzero(img_left_mask[i, j]) > 0):
        #             linearBlendingWithConstantWidth_img[i, j] = img_left[i, j]
        #         else:
        #             linearBlendingWithConstantWidth_img[i, j] = img_right[i, j]

        img_left_mask = np.any(img_left > 0, axis=-1)
        img_right_mask = np.any(img_right > 0, axis=-1)

        # Find overlap
        overlap_mask = img_left_mask & img_right_mask

        # Compute alpha mask
        alpha_mask = np.zeros((hr, wr))
        for i in range(hr):
            overlap_cols = np.where(overlap_mask[i, :])[0]
            if overlap_cols.size != 0:
                minIdx = np.min(overlap_cols)
                maxIdx = np.max(overlap_cols)
                middleIdx = (maxIdx + minIdx) // 2
                decrease_step = 1 / (maxIdx - minIdx)

                # left
                alpha_mask[i, minIdx:middleIdx+1] = 1
                alpha_mask[i, middleIdx-constant_width:middleIdx+1] = 1 - decrease_step * (np.arange(middleIdx-constant_width, middleIdx+1) - minIdx)

                # right
                alpha_mask[i, middleIdx+1:maxIdx+1] = 0
                alpha_mask[i, middleIdx+1:middleIdx+constant_width+1] = 1 - decrease_step * (np.arange(middleIdx+1, middleIdx+constant_width+1) - minIdx)

        # Create blending image
        blending_img = np.copy(img_right)
        blending_img[img_left_mask] = img_left[img_left_mask]  # Copy non-overlapping region of left image

        # Apply linear blending
        blending_img[overlap_mask] = alpha_mask[overlap_mask, None] * img_left[overlap_mask] + (1 - alpha_mask[overlap_mask, None]) * img_right[overlap_mask]


        return blending_img

class Homography:
    def __init__(self):
        pass
    
    def solve_homography(self, kps_1, kps_2):
        A = []
        for i in range(len(kps_1)):
            A.append([kps_1[i, 0], kps_1[i, 1], 1, 0, 0, 0, -kps_1[i, 0] * kps_2[i, 0], -kps_1[i, 1] * kps_2[i, 0], -kps_2[i, 0]])
            A.append([0, 0, 0, kps_1[i, 0], kps_1[i, 1], 1, -kps_1[i, 0] * kps_2[i, 1], -kps_1[i, 1] * kps_2[i, 1], -kps_2[i, 1]])

        # Solve system of linear equations Ah = 0 using SVD
        u, sigma, vt = np.linalg.svd(A)
        
        # pick H from last line of vt
        H = np.reshape(vt[8], (3, 3))
        
        # normalization, let H[2,2] equals to 1
        H = (1/H.item(8)) * H
        
        return H

def cylindricalWarp(img):
    """
    This function returns the cylindrical warp for a given image and intrinsics matrix K
    """

    h_,w_ = img.shape[:2]
    focal = w_
    K = np.array([[focal,0,w_/2],[0,focal,h_/2],[0,0,1]])
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
  

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)

    SIFT_Detector = cv2.SIFT_create()
    stitcher = Stitcher()
    fileList = ['Base']
    blendType = None
    
    for filename in fileList:
        if filename == 'Base':
            imgNameList = ['Base1.jpg', 'Base2.jpg', 'Base3.jpg']
            blendType = 'linearBlending'
        else:
            imgNameList = ['Challenge1.jpg', 'Challenge2.jpg', 'Challenge3.jpg', 'Challenge4.jpg', 'Challenge5.jpg', 'Challenge6.jpg']
            blendType = 'linearBlendingWith'
            

        img1, img_gray1 = read_img(os.path.join(filename, imgNameList[0]))
        img1 = cylindricalWarp(img1)

        img2, img_gray2 = read_img(os.path.join(filename, imgNameList[1]))
        img2 = cylindricalWarp(img2)
        
        result_img = stitcher.stitch([img1, img2], [img_gray1, img_gray2],SIFT_Detector, threshold = 0.75, blend = blendType)
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite(os.path.join(filename, 'First_2_result.jpg'), result_img)
        
        for idx, img_name in enumerate(imgNameList[2:], start = 3):
            next_img, next_img_gray = read_img(os.path.join(filename, img_name))
            next_img = cylindricalWarp(next_img)
            result_img = stitcher.stitch([result_img, next_img], [result_gray, next_img_gray], SIFT_Detector, threshold = 0.75, blend = blendType)
            result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
            
            if idx != len(imgNameList): 
                cv2.imwrite(os.path.join(filename, f'First_{idx}_result.jpg'), result_img)
            
        cv2.imwrite(os.path.join(filename, 'Final_result.jpg'), result_img)