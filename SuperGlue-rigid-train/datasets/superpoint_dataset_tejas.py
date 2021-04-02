'''
to test 1024 x 1024 points in warped and source image
, image'''

import sys

sys.path.append('../')
sys.path.append('./')


import numpy as np
import torch
import os
import cv2
import math
import datetime
import matplotlib.pyplot as plt


from torch.utils.data import Dataset


from models.superpoint import SuperPoint
from models.utils import frame2tensor, array2tensor,\
                         plot_image_with_kpts, generateRandomWarp,\
                         make_matching_plot_generic


from scipy.spatial import KDTree
 

class SuperPointDataset(Dataset):

    def __init__(self, dataset_path,\
                    image_list=None,\
                    device='cpu',\
                    superpoint_config={},\
                    DEBUG=False,\
                    NUM_KPS_PER_IMAGE = 512,\
                    theta_amp = 2*np.pi):

        print('Using SuperPoint dataset')

        self.DEBUG = DEBUG
        self.NUM_KPS_PER_IMAGE = NUM_KPS_PER_IMAGE
        self.dataset_path = dataset_path
        self.device = device
        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(dataset_path)
                if name.endswith('jpg') or name.endswith('png') ]

        # Load SuperPoint model
        self.superpoint = SuperPoint(superpoint_config)
        self.superpoint.to(device)
        self.theta_amp = theta_amp   ## in radians
        self.boundary_pad_hack = 10
        self.MATCHING_THRESH = 4 #pixels
        self.image_shape = (640,480) # width, height

    def __len__(self):
        return len(self.image_names)
    
    def randomize_prediction(self, kps, scores, descriptors):

        num_kps=len(kps)
        idx=torch.randperm(num_kps)
        scores = scores[idx]
        kps = kps[idx,:]
        descriptors = descriptors[:,idx]
        return kps, scores, descriptors
    

    def superpoint_pred_from_cv_image(self, image):
         
        data_torch = frame2tensor(image, self.device)
        pred = self.superpoint({ 'image': data_torch })
        kps = pred['keypoints'][0]  
        scores = pred['scores'][0] 
        descriptors = pred['descriptors'][0] 
        return kps , scores, descriptors
    

    def adjusted_superpoint_pred_from_cv_image(self, image, num_kps):

        height, width = image.shape[:2]
        kps, scores, descriptors = self.superpoint_pred_from_cv_image(image)
        kps, scores, descriptors = self.randomize_prediction(kps, scores, descriptors)


        # print("keypoints detected with superpoint: ", len(kps))
        # boundary_pad_hack = 10

        if len(kps) > num_kps:
            topk_idx = torch.topk(scores, num_kps, dim=0)
            descriptors_adjusted = descriptors[:,topk_idx.indices]
            kps_adjusted = kps[topk_idx.indices,:]
            scores_adjusted = scores[topk_idx.indices]

        if len(kps) <= num_kps:
            num_extra = num_kps - len(kps)
            y_extra = torch.randint( self.boundary_pad_hack, width - self.boundary_pad_hack,[num_extra,1],dtype=torch.float32)
            x_extra = torch.randint( self.boundary_pad_hack, height - self.boundary_pad_hack,[num_extra,1],dtype=torch.float32)

            pts_extra = torch.cat((y_extra,x_extra),1).to(self.device)
            kps_adjusted = torch.cat((kps,pts_extra),0)

            kps_adjusted = kps_adjusted[torch.randperm(num_kps),:]

            data_temp = {}
            data_temp['image'] = frame2tensor(image, self.device)
            data_temp['keypoints'] = [kps_adjusted]

            descriptors_adjusted,scores_adjusted = self.superpoint.computeDescriptorsAndScores(data_temp)
            descriptors_adjusted = descriptors_adjusted[0]
            scores_adjusted = scores_adjusted[0]

        # print("keypoints adjusted for efficient batching: ", len(kps_adjusted))
        # print(type(scores_adjusted),"type(scores_adjusted)")
        # print(type(kps_adjusted),"type(kps_adjusted)")
        # print(type(descriptors_adjusted),"type(descriptors_adjusted)")
        
        return kps_adjusted, scores_adjusted, descriptors_adjusted
    
    def nearest_neighbours(self, src_points,tgt_points):
        tree_tgt = KDTree(tgt_points)
        dd, ii = tree_tgt.query(src_points, k=1)

        outl_idx = np.where(dd> self.MATCHING_THRESH)
        ii[outl_idx] = -1

        return ii


    def find_matches(self, kps_src_transf_np, kps_tgt_np):
        
        ### input: kps_src_transf_np :  NUM_KPS_PER_IMAGE x 2  numpy array
        ### input: kps_src_np  :        NUM_KPS_PER_IMAGE x 2  numpy array

        ### output: list []
        
        num_pnts_src = len(kps_src_transf_np)
        num_pnts_tgt = len(kps_tgt_np)

        matches_src2tgt = self.nearest_neighbours(kps_src_transf_np,kps_tgt_np)
        matches_tgt2src = self.nearest_neighbours(kps_tgt_np, kps_src_transf_np)
        
        matches_list = [matches_src2tgt,matches_tgt2src]

        return matches_list

    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.dataset_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image,self.image_shape)
        # print(image.shape,"image.shape")       
        height, width = image.shape[:2]
        min_size = min(height, width)

        corners_gt = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)

        M_gt2src = generateRandomWarp(corners_gt, min_size/4, theta_amp=self.theta_amp)
        image_src = cv2.warpPerspective(image, M_gt2src, (width, height))

        M_gt2tgt = generateRandomWarp(corners_gt, min_size/4, theta_amp=self.theta_amp)
        image_tgt = cv2.warpPerspective(image, M_gt2tgt, (width, height))

        M_src2tgt = M_gt2tgt@np.linalg.inv(M_gt2src)
        M_src2tgt = M_src2tgt/M_src2tgt[-1,-1]

        image_src_transf = cv2.warpPerspective(image_src, M_src2tgt, (width, height))

        kps_src, scores_src, descriptors_src = self.adjusted_superpoint_pred_from_cv_image(image_src, self.NUM_KPS_PER_IMAGE)
        kps_tgt, scores_tgt, descriptors_tgt = self.adjusted_superpoint_pred_from_cv_image(image_tgt, int(self.NUM_KPS_PER_IMAGE/1.5))


        kps_src_transf_np = cv2.perspectiveTransform(kps_src.cpu().numpy()[None], M_src2tgt)
        kps_src_transf_np = kps_src_transf_np[0]

        kps_tgt_np = kps_tgt.cpu().numpy()
        kps_src_np = kps_src.cpu().numpy()

        matches_list = self.find_matches(kps_src_transf_np, kps_tgt_np)
        # print("superpoint dataset tejas")

        matches0_torch = torch.tensor(matches_list[0])
        matches1_torch = torch.tensor(matches_list[1])


        if self.DEBUG:
            make_matching_plot_generic(image_src=image_src, image_tgt=image_tgt,\
                                       kps_src_np=kps_src_np, kps_tgt_np=kps_tgt_np,\
                                       matches=matches_list)

            # kps_srccv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps_src.cpu().numpy().squeeze() ]
            # kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps_tgt.cpu().numpy().squeeze() ]
            # matchescv = [ cv2.DMatch(k0, k1, 0) for k0,k1 in zip(matches[0], matches[1]) ]
            # outimg = None
            # outimg = cv2.drawMatches(image_src, kps_srccv, image_tgt, kps1cv, matchescv, outimg)
            # cv2.imwrite('./temp/matches.jpg', outimg)
            # outimg = cv2.drawKeypoints(image_src, kps_srccv, outimg)
            # cv2.imwrite('./temp/kps_src.jpg', outimg)
            # outimg = cv2.drawKeypoints(image_tgt, kps1cv, outimg)
            # cv2.imwrite('./temp/kps_tgt.jpg', outimg)

 

        try:
            return {
                ## original format
                'keypoints0': kps_src,
                'keypoints1': kps_tgt,
                'descriptors0': descriptors_src,
                'descriptors1': descriptors_tgt,
                'scores0': scores_src,
                'scores1': scores_tgt,
                'image0': frame2tensor(image_src,self.device).squeeze(0),
                'image1': frame2tensor(image_tgt,self.device).squeeze(0),
                'matches0_gt': matches0_torch,
                'matches1_gt': matches1_torch,
                'file_name': self.image_names[idx],
            }
        except:
            print("error in superpoint dataset generation")
            import ipdb; ipdb.set_trace()




if __name__ == '__main__':
    # file_name = "sample.png"
    # dataset_path = "../assets/freiburg_sequence"
    dataset_path = "../assets/scannet_sample_images"
    
    sp = SuperPointDataset(dataset_path,DEBUG=True)
    pred = sp[10] 


    # import ipdb; ipdb.set_trace()
