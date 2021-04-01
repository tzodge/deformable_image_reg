"""
@author: Bassam Bikdash, Tejas Zodage

Opencv2 Feature Matching, Feature Detection, and Pose Estimation
"""
import argparse
import numpy as np
import numpy.random as random
import cv2
from super_matching import SuperMatching

# TODO: Useful link for code organization: https://www.programcreek.com/python/example/89440/cv2.FlannBasedMatcher

""" Compute transformation matrix (rotation + translation) """
def transf_matrix(theta=0, translation=[0,0]):
    assert len(translation) == 2
    tx, ty  = translation

    # First two columns correspond to the rotation b/t images
    M = np.zeros((2,3))
    M[:,0:2] = np.array([[np.cos(theta), np.sin(theta)],\
                         [ -np.sin(theta), np.cos(theta)]])

    # Last column corresponds to the translation b/t images
    M[0,2] = tx
    M[1,2] = ty
    return M

""" Convert the 2x3 rot/trans matrices to a 3x3 matrix """
def transf_mat_3x3(M):
    M_out = np.eye(3)
    M_out[0:2,0:3] = M
    return M_out

"""
Use ORB Feature Detection to find keypoints in the image
"""
def TraditionalDetection(img1, img2, debug=False):
    orb = cv2.ORB_create()
    # Find keypoints and respective descriptors for each image
    ref_keypoints, ref_descriptors = orb.detectAndCompute(img1, None)
    align_keypoints, align_descriptors = orb.detectAndCompute(img2, None)

    ref_cloud = cv2.drawKeypoints(img1, ref_keypoints, outImage=None, color=(255, 0, 0), flags=0)
    align_cloud = cv2.drawKeypoints(img2, align_keypoints[0:100], outImage=None, color=(0, 0, 255), flags=0)

    # Display the images with key points highlighted
    if debug:
        displayImages(ref_cloud, 'Reference Image Keypoints',
                  align_cloud, 'Aligned Image Keypoints')

    return ref_keypoints, ref_descriptors, align_keypoints, align_descriptors,\
           ref_cloud, align_cloud

"""
Use SuperPoint and SuperGlue to detect and match keypoints between images.
"""
def SuperGlueDetection(img1, img2, debug=False):
    # path = 'test_images/iphone3_OG_iphone3_warped_matches.npz'
    # npz = np.load(path)
    # kp1 = npz['keypoints0']
    # kp2 = npz['keypoints1']
    # matches = npz['matches']
    # match_confidence = npz['match_confidence']
    
    sp = SuperMatching()
    kp1, kp2, matches1, matches2 = sp.detectAndMatch(img1, img2)
    
    
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    rgb2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # Show keypoints
    for x,y in kp1:
        cv2.circle(rgb1, (x,y), 2, (255,0,0), -1)
    for x,y in kp2:
        cv2.circle(rgb2, (x,y), 2, (0,0,255), -1)
    # displayImages(rgb1, 'Reference Image Keypoints',
    #               rgb2, 'Aligned Image Keypoints', 'results/superpoint_detection.jpg')
    
    # Show matches
    if debug:
        sp.plot_matches(img1, img2, kp1, kp2, matches1, matches2)

    return kp1, kp2, matches1, matches2

"""
Use traditional feature matching technique, either FLANN or brute force matcher
"""
def TraditionalMatching(ref_descriptors, align_descriptors, bf=False):

    if bf == False:
        # ASSERT: Use FLANN Matcher
        index_params= dict(algorithm=6,  # FLANN_INDEX_LSH
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2

        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        # import ipdb; ipdb.set_trace()
        
        matches = flann.knnMatch(ref_descriptors, align_descriptors, k=2)
        # Filter matches using the Lowe's ratio test
        ratio_thresh = 0.7
        # import ipdb; ipdb.set_trace()
        good_matches = []
        
        try:
            for m,n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        except:
            for i in range(len(matches)):
 
                try:
                    m = matches[i][0]
                    n = matches[i][1]
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                except:
                    ""

    else:
        # Hamming norm is what we want since we're using ORB. crossCheck is on for better results
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match the descriptors
        matches = bf.match(ref_descriptors, align_descriptors)
        # Sort them in the order of their distance.
        good_matches = sorted(matches, key = lambda x:x.distance)

    return good_matches

"""
Overlay the transformed noisy image with the original and estimate the affine
transformation between the two
# """
# def generateComposite(ref_keypoints, align_keypoints, ref_cloud, align_cloud,
#                       matches, rows, cols):
def displayComposite(ref,align,M_est):
    # Converts vector of keypoints to vector of points or the reverse, where each keypoint is assigned the same size and the same orientation.
    # ref_keypoints_np = cv2.KeyPoint_convert(ref_keypoints)
    # align_keypoints_np = cv2.KeyPoint_convert(align_keypoints)

    # reordered_ref = np.zeros((len(matches), 2))
    # reordered_align = np.zeros((len(matches), 2))
    
    rows, cols = ref.shape

    # debug = False
    # for i, m in enumerate(matches):
    #     # I had to adjust the indices for m here too
    #     reordered_ref[i,:] = ref_keypoints_np[m.queryIdx,:]
    #     reordered_align[i,:] = align_keypoints_np[m.trainIdx,:]

    # M_est = cv2.estimateAffinePartial2D(reordered_ref, reordered_align)[0]

    # # M to go from reference image to the aligned image; should be = to OG M up above
    M_est_inv = np.linalg.inv(transf_mat_3x3(M_est))[0:2,:]

    beta_img = cv2.warpAffine(align, M_est_inv, (cols, rows))
    alpha_img = np.copy(ref)
    alpha = 0.5
    composed_img = cv2.addWeighted(alpha_img, alpha, beta_img, 1-alpha, 0.0)
    displayImages(composed_img, 'Composite Image')
    # return 

"""
Compute the translation/rotation pixel error between the estimated RANSAC
transformation and the true transformation done on the image.
"""
def computeError(M, M_est, M_est_inv):
    print('\nEstimated M\n', M_est)
    print('\nTrue M\n', M)

    # Add error
    error = M @ transf_mat_3x3(M_est_inv)
    R_del = error[0:2,0:2]
    t_del = error[0:2,2]

    print('\nTranslation Pixel Error: ', np.linalg.norm(t_del))
    print('Rotation Pixel Error: ', np.linalg.norm(R_del))
    
"""
Display a single image or display two images conatenated together for comparison
Specifying a path will save whichever image is displayed (the single or the
composite).
"""
def displayImages(img1, name1='Image 1', img2=None, name2='Image2', path=None):
    if img2 is None:
        # ASSERT: Display only 1 image
        output = img1
        # cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
        cv2.imshow(name1, img1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Display both images concatenated
        output = np.concatenate((img1, img2), axis=1)
        # cv2.namedWindow(name1 + ' and ' + name2, cv2.WINDOW_NORMAL)
        cv2.imshow(name1 + ' and ' + name2, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path is None:
        # Save the image at the current path
        print("")
    else:
        cv2.imwrite(path, output)

"""
Test feature detection, feature matching, and pose estimation on an image.
"""
def cv_kp_to_np(cv_keypoints):
    list_kp_np = []
    for idx in range(0, len(cv_keypoints)):
        list_kp_np.append(cv_keypoints[idx].pt)
    
    return np.array(list_kp_np).astype(np.int64)        
    # ref_cloud = np.float([cv_keypoints[idx].pt for idx in range(0, len(cv_keypoints))]).reshape(-1, 1, 2)


def find_transformation_keypoints_image(ref, align, debug=False):
    # try:
    
    if debug:
        displayImages(ref, 'Reference Image', align, 'Aligned Image')
        # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
        
        
        # Apply Feature Detection and Feature Matching
    ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align)

    # ref_keypoints, ref_descriptors, align_keypoints, align_descriptors, \
    # ref_cloud_overlay, align_cloud_overlay = TraditionalDetection(ref, align)

    ref_kp_image = np.zeros((np.shape(ref)),  dtype=np.uint8)
    align_kp_image = np.zeros((np.shape(align)),  dtype=np.uint8)
 
    # ref_cloud = np.float([ref_keypoints[idx].pt for idx in range(0, len(ref_keypoints))]).reshape(-1, 1, 2)
    # ref_cloud = np.float([ref_keypoints[idx].pt for idx in range(0, len(ref_keypoints))]).reshape(-1, 1, 2)
    ref_cloud = cv_kp_to_np(ref_keypoints)
    align_cloud = cv_kp_to_np(align_keypoints)

    ref_kp_image[ref_cloud[:,0], ref_cloud[:,1]] = 255    
    align_kp_image[align_cloud[:,0], align_cloud[:,1]] = 255    

    # import ipdb; ipdb.set_trace()    

    cv2.imshow("ref_kp_image",ref_kp_image)
    cv2.imshow("align_kp_image",align_kp_image)
    cv2.waitKey(0)
    return np.zeros((3,3)), 0
    # except:
    #     return np.zeros((3,3)), 0



def find_transformation_ORB(ref, align, debug=False):

    try:
        if debug:
            displayImages(ref, 'Reference Image', align, 'Aligned Image')
        # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
        
        # ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align)
        
        # Apply Feature Detection and Feature Matching
        ref_keypoints, ref_descriptors, align_keypoints, align_descriptors, \
            ref_cloud, align_cloud = TraditionalDetection(ref, align)


        # cv2.imshow("ref_cloud",ref_cloud)
        # cv2.waitKey(0)

        # print(ref_cloud,"ref_cloud")
        # import ipdb; ipdb.set_trace()
        matches = TraditionalMatching(ref_descriptors, align_descriptors, bf=False)

        ref_keypoints_np = cv2.KeyPoint_convert(ref_keypoints)
        align_keypoints_np = cv2.KeyPoint_convert(align_keypoints)

        reordered_ref = np.zeros((len(matches), 2))
        reordered_align = np.zeros((len(matches), 2))

        for (i, m) in enumerate(matches):
            # I had to adjust the indices for m here too
            reordered_ref[i,:] = ref_keypoints_np[m.queryIdx,:]
            reordered_align[i,:] = align_keypoints_np[m.trainIdx,:]

        M_est = cv2.estimateAffinePartial2D(reordered_ref, reordered_align)[0]


        # Draw matches
        img3 = cv2.drawMatches(ref, ref_keypoints,
                               align, align_keypoints,
                               matches, outImg=None, flags=2)
        if debug:
            displayImages(img3, 'Draw Matches', path='results/Test Image Feature Matching.jpg')

        print(len(matches),"num of matches")
        return M_est, len(matches)

    except:
        return np.zeros((3,3)), 0

def draw_src_and_dst_pnts(img_debug, points_src, points_dst):
    # img_debug = np.copy(img_debug)
    img_debug = get_debug_image(img_debug)
    for point_src,point_dst in zip(points_src, points_dst):
        cv2.circle(img_debug, tuple(point_src), 5, (0, 255, 0), 1)
        cv2.circle(img_debug, tuple(point_dst), 3, (0, 0, 255), 1)
        cv2.line(img_debug, tuple(point_src), tuple(point_dst), (255, 255, 0), 1)

    return img_debug


def get_tps_from_matches(points_src, points_dst):
    matches =[]
    for i in range(len(points_src)):
        matches.append(cv2.DMatch(i,i,0))


    tps = cv2.createThinPlateSplineShapeTransformer()
     

    # img_dst = np.copy(img_grid)

    tps.estimateTransformation(points_dst.reshape(1,-1,2),\
                               points_src.reshape(1,-1,2),
                               matches)
    return tps

def get_debug_image(img):
    img_debug = np.copy(img)
    if len(img_debug.shape)==2:
        img_debug = np.dstack((img_debug, img_debug,img_debug))

    return img_debug


def get_diff_img(img_dst, align):
    diff_image = img_dst.astype(np.float32) - align.astype(np.float32)
    diff_image = abs(diff_image).astype(np.uint8)

    return diff_image

def remove_border_kps (ref_matched, align_matched):
    assert len(ref_matched) == len(align_matched)

    delete_idx_1 = np.where(ref_matched[:,0]<60)[0]
    delete_idx_2 = np.where(ref_matched[:,0]>530)[0]

    delete_idx = np.hstack((delete_idx_1, delete_idx_2))
    ref_matched = np.delete(ref_matched, delete_idx, axis=0)
    align_matched = np.delete(align_matched, delete_idx, axis=0)

    return ref_matched, align_matched

def find_tps_warp_SuperGlue(ref, align, debug=False):
    ### ref : src
    ### align : tgt
        # displayImages(ref, 'Reference Image', align, 'Aligned Image')
    # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
    
    ref_keypoints, align_keypoints, ref_matched, align_matched = SuperGlueDetection(ref, align, debug)

    # import ipdb; ipdb.set_trace()
    # ref_matched, align_matched = remove_border_kps (ref_matched, align_matched)
    ref_debug = get_debug_image(ref)
    align_debug = get_debug_image(align)

    if debug:

        for point in ref_matched:
            cv2.circle(ref_debug, tuple(point), 5, (0, 255, 0), 1)
        cv2.imshow("src_debug", ref_debug)
        cv2.imwrite("./temp_results/src_debug.jpg", ref_debug)

        for point in align_matched:
            cv2.circle(align_debug, tuple(point), 3, (0, 0, 255), 1)
        cv2.imshow("target_debug", align_debug)
        cv2.imwrite("./temp_results/target_debug.jpg", align_debug)

        for point_ref, point_align in zip(ref_matched, align_matched):
            cv2.circle(ref_debug, tuple(point_align), 3, (0, 0, 255), 1)
            cv2.circle(ref_debug, tuple(point_ref), 5, (0, 255, 0), 1)

            cv2.line(ref_debug, tuple(point_align), tuple(point_ref), (255, 255, 0), 1)

        cv2.imshow("target points on src", ref_debug)
        cv2.imwrite("./temp_results/target points on src.jpg", ref_debug)

        cv2.waitKey(0)
   

    tps = get_tps_from_matches(ref_matched, align_matched)

    img_dst = np.copy(ref)
    tps.warpImage(ref, img_dst,cv2.INTER_CUBIC,cv2.BORDER_REPLICATE)

    img_dst_debug = get_debug_image(img_dst)
    img_dst_debug = draw_src_and_dst_pnts(img_dst_debug, ref_matched, align_matched)

    cv2.imshow("img_dst", img_dst_debug)
    cv2.imwrite("./temp_results/img_dst.jpg", img_dst_debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    diff_image = get_diff_img(img_dst, align)
    cv2.imshow("difference image tgt and warped src", diff_image)
    cv2.imwrite("./temp_results/difference image tgt and warped src.jpg", diff_image)

    diff_image = get_diff_img(ref, align)
    cv2.imshow("difference image tgt and src", diff_image)
    cv2.imwrite("./temp_results/difference image tgt and src.jpg", diff_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # import ipdb; ipdb.set_trace()
    # try :
    #     M_est = cv2.estimateAffinePartial2D(ref_matched, align_matched)[0]



    # except:
    #     print("could not find matches")
    #     M_est = np.array([[1,0,0],
    #                       [0,1,0]])

    # Draw matches
    # if debug:
    #     img3 = cv2.drawMatches(ref, ref_keypoints,
    #                            align, align_keypoints,
    #                            matches1, outImg=None, flags=2)
    #     displayImages(img3, 'Draw Matches', path='results/Test Image Feature Matching.jpg')

    # print(len(matches1),"num of matches")
    # return M_est, len(matches1)



def find_transformation_SuperGlue(ref, align, debug=False):

    if debug:
        displayImages(ref, 'Reference Image', align, 'Aligned Image')
    # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
    
    ref_keypoints, align_keypoints, matches1, matches2 = SuperGlueDetection(ref, align, debug)
    
    import ipdb; ipdb.set_trace()
    try :
        M_est = cv2.estimateAffinePartial2D(matches1, matches2)[0]



    except:
        print("could not find matches")
        M_est = np.array([[1,0,0],
                          [0,1,0]])

    # Draw matches
    # if debug:
    #     img3 = cv2.drawMatches(ref, ref_keypoints,
    #                            align, align_keypoints,
    #                            matches1, outImg=None, flags=2)
    #     displayImages(img3, 'Draw Matches', path='results/Test Image Feature Matching.jpg')

    print(len(matches1),"num of matches")
    return M_est, len(matches1)

    # except:
    #     return np.zeros((3,3)), 0



def test_single(image_path):
    ref = cv2.imread(image_path, 0)
    print(ref,"ref")
    # Transformations are proportional to the scale of the image
    # M1 = transf_matrix(random.uniform(low=-np.pi/4, high=np.pi/4),
    #                    [random.randint(low=-ref.shape[0]/20, high=ref.shape[0]/20),
    #                     random.randint(low=-ref.shape[1]/20, high=ref.shape[1]/20)])

    # M = transf_mat_3x3(M1)
    theta = -np.pi/3

    M1 = transf_matrix(0, [-ref.shape[0]/2,-ref.shape[1]/2])
    M2 = transf_matrix(theta , [0,0])
    M3 = transf_matrix(0, [ref.shape[0]/2,ref.shape[1]/2])

    M = transf_mat_3x3(M3)@transf_mat_3x3(M2)@transf_mat_3x3(M1)

    # Perform affine transform and add noise to the original image
    rows, cols = ref.shape
    align = cv2.warpAffine(ref, M[0:2,:], (cols, rows))
    # align += np.random.randint(20, size=align.shape, dtype=align.dtype)

    # M_est,_ = find_transformation_ORB(ref, align, debug=True)
    M_est, num_matches = find_transformation_SuperGlue(ref, align, debug=True)

    displayComposite(ref,align,M_est)

def test_two_iterative(ref_path, align_path):

    ref = cv2.imread(ref_path, 0)
    align = cv2.imread(align_path, 0)
    # align = 255- align
    align=cv2.resize(align,(ref.shape[1],ref.shape[0]))
    print(ref.shape,"ref.shape")
    print(align.shape,"align.shape")

    # Transformations are proportional to the scale of the image
    # M1 = transf_matrix(random.uniform(low=-np.pi/4, high=np.pi/4),
    #                    [random.randint(low=-ref.shape[0]/20, high=ref.shape[0]/20),
    #                     random.randint(low=-ref.shape[1]/20, high=ref.shape[1]/20)])
    # theta = -np.pi/2+np.pi/30
    # theta = -np.pi/2
    theta = 0

    M1 = transf_matrix(0, [-ref.shape[0]/2,-ref.shape[1]/2])
    M2 = transf_matrix(theta , [0,0])
    M3 = transf_matrix(0, [ref.shape[0]/2,ref.shape[1]/2])

    M_init = transf_mat_3x3(M3)@transf_mat_3x3(M2)@transf_mat_3x3(M1)

    M = M_init
    

    rows, cols = ref.shape
    for i in range(1):
        ref = cv2.warpAffine(ref, M[0:2,:], (cols, rows))
        # M_est, num_matches = find_tps_warp_SuperGlue(ref, align, debug=True)
        find_tps_warp_SuperGlue(ref, align, debug=True)
        
        # displayComposite(ref,align,M_est)
        # M = M_est



def test_two(ref_path, align_path):
    ref = cv2.imread(ref_path, 0)
    align = cv2.imread(align_path, 0)

    align=cv2.resize(align,ref.shape)

    # Transformations are proportional to the scale of the image
    # M1 = transf_matrix(random.uniform(low=-np.pi/4, high=np.pi/4),
    #                    [random.randint(low=-ref.shape[0]/20, high=ref.shape[0]/20),
    #                     random.randint(low=-ref.shape[1]/20, high=ref.shape[1]/20)])
    # theta = -np.pi/2+np.pi/30
    # theta = -np.pi/2
    theta = 0

    M1 = transf_matrix(0, [-ref.shape[0]/2,-ref.shape[1]/2])
    M2 = transf_matrix(theta , [0,0])
    M3 = transf_matrix(0, [ref.shape[0]/2,ref.shape[1]/2])

    M_init = transf_mat_3x3(M3)@transf_mat_3x3(M2)@transf_mat_3x3(M1)

    M = M_init
    # Perform affine transform and add noise to the original image
    rows, cols = ref.shape
    ref = cv2.warpAffine(ref, M[0:2,:], (cols, rows))
    # align += np.random.randint(20, size=align.shape, dtype=align.dtype)

    # find_transformation_ORB(ref, align, debug=True)
    M_est, num_matches = find_transformation_SuperGlue(ref, align, debug=True)
    displayComposite(ref,align,M_est)


# Main Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('-ref', '--reference_path',
    #                     type=str, default='data/iphone3_OG.jpg',
    #                     help='Reference Image')
    # parser.add_argument('-align', '--align_path',
    #                     type=str, default='data/iphone3_rotated90CW.jpg',
    #                     help='Image to align')
    

    parser.add_argument('-ref', '--reference_path',
                        type=str, default='data/sample_data_lowres/real_image.png',
                        help='Reference Image')
    parser.add_argument('-align', '--align_path',
                        type=str, default='data/sample_data_lowres/xray_image_simulated.png',
                        help='Image to align')
    
    args = parser.parse_args()
    # test_single(args.reference_path)
    # test_two(args.reference_path, args.align_path)
    test_two_iterative(args.reference_path, args.align_path)
    
    