"""
@author: Bassam Bikdash, Tejas Zodage

Opencv2 Feature Matching, Feature Detection, and Pose Estimation
"""
import argparse
import numpy as np
import numpy.random as random
import cv2
from super_matching import SuperMatching


def setup_sg_class(args):
    sg_matching = SuperMatching()
    sg_matching.weights = 'custom'
    sg_matching.weights_path = args.superglue_weights_path
    return sg_matching
 
def SuperGlueDetection(img1, img2, sg_matching, debug=False):
    # path = 'test_images/iphone3_OG_iphone3_warped_matches.npz'
    # npz = np.load(path)
    # kp1 = npz['keypoints0']
    # kp2 = npz['keypoints1']
    # matches = npz['matches']
    # match_confidence = npz['match_confidence']
    
    # sg_matching = SuperMatching()
    kp1, kp2, matches1, matches2 = sg_matching.detectAndMatch(img1, img2)
    
    
    rgb1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    rgb2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    # Show keypoints
    for x,y in kp1.astype(np.int64):
        cv2.circle(rgb1, (x,y), 2, (255,0,0), -1)
    for x,y in kp2.astype(np.int64):
        cv2.circle(rgb2, (x,y), 2, (0,0,255), -1)
    # displayImages(rgb1, 'Reference Image Keypoints',
    #               rgb2, 'Aligned Image Keypoints', 'results/superpoint_detection.jpg')
    
    # Show matches
    if debug:
        sg_matching.plot_matches(img1, img2, kp1, kp2, matches1, matches2)

    return kp1, kp2, matches1, matches2

 
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

  

def draw_src_and_dst_pnts(img_debug, points_src, points_dst):
    # img_debug = np.copy(img_debug)
    img_debug = get_debug_image(img_debug)
    for point_src,point_dst in zip(points_src, points_dst):
        point_src = point_src.astype(np.int64)
        point_dst = point_dst.astype(np.int64)
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

def find_tps_warp_SuperGlue(ref, align, sg_matching, debug=False):
    ### ref : src
    ### align : tgt
        # displayImages(ref, 'Reference Image', align, 'Aligned Image')
    # displayImages(ref, 'Reference Image', align, 'Aligned Image','results/TIF Image Transformation.jpg' )
    
    ref_keypoints, align_keypoints, ref_matched, align_matched = SuperGlueDetection(ref, align, sg_matching, debug)

    # import ipdb; ipdb.set_trace()
    # ref_matched, align_matched = remove_border_kps (ref_matched, align_matched)
    ref_debug = get_debug_image(ref)
    align_debug = get_debug_image(align)

    if debug:

        for point in ref_matched:
            point = point.astype(np.int64)
            cv2.circle(ref_debug, tuple(point), 5, (0, 255, 0), 1)
        cv2.imshow("src_debug", ref_debug)
        cv2.imwrite("./temp_results/src_debug.jpg", ref_debug)

        for point in align_matched:
            point = point.astype(np.int64)
            cv2.circle(align_debug, tuple(point), 3, (0, 0, 255), 1)
        cv2.imshow("target_debug", align_debug)
        cv2.imwrite("./temp_results/target_debug.jpg", align_debug)

        for point_ref, point_align in zip(ref_matched, align_matched):
            point_ref = point_ref.astype(np.int64)
            point_align = point_align.astype(np.int64)
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
 


 
 
def draw_rects(img_debug, points_src, ):
    rect_size = img_debug.shape[0]/30
    for point_src in points_src:
        point_src = point_src.astype(np.int64)
        cv2.rectangle(img_debug, tuple(point_src-   int(rect_size/2)), tuple(point_src+ int(rect_size/2)), color = (0, 255, 0), thickness=2)
        
    return img_debug


def add_edge_pnts(image, s_x = 0.05, s_y = 0.05):
    h,w = image.shape[:2]
    start_y, end_y = int(s_y*h), int((1-s_y)*h)
    start_x, end_x = int(s_x*w), int((1-s_x)*w)
    from IPython import embed
    divx = 10
    divy = 10
    
    x_coords = np.linspace(start_x,end_x, divx)
    y_coords = np.linspace(start_y,end_y, divy)

    left_pts = np.zeros((divy,2)) + start_x
    left_pts[:,0] = y_coords

    right_pts = np.zeros((divy,2)) + end_x
    right_pts[:,0] = y_coords

    up_pts = np.zeros((divx,2)) + start_y
    up_pts[:,1] = x_coords    

    down_pts = np.zeros((divx,2)) + end_y
    down_pts[:,1] = x_coords    

    pts = np.vstack((left_pts,right_pts,up_pts,down_pts))

    pts = np.fliplr(pts)    
    rect_size = image.shape[0]/30
    for pt in pts:
        pt = pt.astype(np.int64)
        cv2.rectangle(image, tuple(pt-   int(rect_size/2)), tuple(pt+ int(rect_size/2)), color = (255, 255, 255), thickness=2)

    return image

def pad_image(image, s_x = 0.1, s_y = 0.1):

    h,w = image.shape[:2]
    
    image_mask = np.zeros(image.shape, dtype=np.uint8)
    
    start_y, end_y = int(s_y*h), int((1-s_y)*h)
    start_x, end_x = int(s_x*w), int((1-s_x)*w)

    image_mask[start_y : end_y, start_x : end_x] = 1

    image_out = image_mask*image
    return image_out

def crop_image(image, s_x = 0.1, s_y = 0.1):
    h,w = image.shape[:2]
    
    start_y, end_y = int(s_y*h), int((1-s_y)*h)
    start_x, end_x = int(s_x*w), int((1-s_x)*w)
    
    image_out = image[start_y : end_y, start_x : end_x]
    return image_out


def test_two(args):

    ref_path, align_path = args.reference_path, args.align_path
 
    ref = cv2.imread(ref_path, 0)
    align = cv2.imread(align_path, 0)

    ref = pad_image(ref)
    align = pad_image(align)

    ref = add_edge_pnts(ref)
    align = add_edge_pnts(align)

    align=cv2.resize(align,(ref.shape[1],ref.shape[0]))

    sg_matching = setup_sg_class(args)

    find_tps_warp_SuperGlue(ref, align, sg_matching, debug=True)

 

# Main Code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

 
    parser.add_argument('-ref', '--reference_path',
                        type=str, default='data/sample_data_lowres/real_image.png',
                        help='Reference Image')
    parser.add_argument('-align', '--align_path',
                        type=str, default='data/sample_data_lowres/xray_image_simulated.png',
                        help='Image to align')
    parser.add_argument('--superglue', choices={'indoor', 'outdoor', 'custom'}, 
                        default='custom',
                        help='SuperGlue weights')

    parser.add_argument('-weights', '--superglue_weights_path', default='./models/weights/superglue_indoor.pth',
                        help='SuperGlue weights path')
    
    args = parser.parse_args()
    # test_single(args.reference_path)
    # test_two(args.reference_path, args.align_path)
    test_two(args)
    # test_two_iterative(args.reference_path, args.align_path)
    
    