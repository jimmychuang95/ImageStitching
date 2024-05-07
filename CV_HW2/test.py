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

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(features_1, features_2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        good_matches = []
        for m in matches:
            if m.distance < 150:
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

        # Use OpenCV's findHomography function
        best_H, _ = cv2.findHomography(np.array(img1_kp), np.array(img2_kp), cv2.RANSAC, 5.0, maxIters=5000, confidence=0.99)

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
        elif blendType == 'multiBandBlending':
            result = blender.multiBandBlending([warped1, warped2])
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