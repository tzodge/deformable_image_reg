import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./pydemons')
import demons
import argparse
import cv2
import os


diff_save = 0 
count = 0
def get_diff_img(img_dst, align):
    diff_image = img_dst.astype(np.float32) - align.astype(np.float32)
    diff_image = abs(diff_image).astype(np.uint8)
    return diff_image

def run_demons(moving, fixed, **kwargs):
    # plot input images
    plt.ion()
    plt.figure(figsize=(13.5, 7))
    plt.gray()
    ax = plt.subplot(221)
    ax.set_title("fixed")
    plt.axis("off")
    ax.imshow(fixed)
    ax = plt.subplot(222)
    ax.set_title("moving")
    plt.axis("off")
    ax.imshow(moving)

    # run demons
    warped = moving
    # diff = warped - fixed
    diff = get_diff_img(warped , fixed)


    ax = plt.subplot(223)
    ax.set_title("warped")
    ax.axis("off")
    warped_thumb = ax.imshow(warped)
    ax = plt.subplot(224)
    ax.set_title("diff")
    ax.axis("off")
    diff_thumb = ax.imshow(diff)
    # plt.show()

    def _callback(variables):

        global diff_save, count
        warped = variables["warped"]
        fixed = variables['fixed']
        diff = warped - fixed
        diff_save = get_diff_img(warped,fixed)
        warped_thumb.set_data(warped)
        plt.draw()
        diff_thumb.set_data(diff)
        plt.draw()
        count +=1
        print(count)
        plt.pause(1)
        cv2.imwrite("./diff_image_demons.png",diff_save)

    # return demons.demons(fixed, moving, callback=_callback, **kwargs)
    # return demons(fixed, moving, callback=_callback, **kwargs)
    demons.demons(fixed, moving, callback=_callback, **kwargs)

def mask_cv2_img(img, idx_2d,c='r'):
    r_chan = img[:,:,2]
    g_chan = img[:,:,1]
    b_chan = img[:,:,0]

    if c=='r':
        r_chan[idx_2d] = 255
    else:
        r_chan[idx_2d] = 0

    if c=='g':
        g_chan[idx_2d] = 255
    else:
        g_chan[idx_2d] = 0

    if c=='b':
        b_chan[idx_2d] = 255
    else:
        b_chan[idx_2d] = 0

    return np.dstack((b_chan,g_chan,r_chan))

def mask_cv2_img_by_arr(img, idx_2d,c=[255,0,0]):
    img  = img.copy()
    r_chan = img[:,:,2]
    g_chan = img[:,:,1]
    b_chan = img[:,:,0]

    r_chan[idx_2d] = c[2]
    g_chan[idx_2d] = c[1]
    b_chan[idx_2d] = c[0]
    return np.dstack((b_chan,g_chan,r_chan))

def remove_noise(img, ksize=2):
    kernel = np.ones((ksize,ksize),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def save_and_disp(img, disp_name='disp_name'):
    if not os.path.exists('./results'):
        os.mkdir('./results')
    cv2.imwrite('./results/'+'{}.png'.format(disp_name),img)
    cv2.namedWindow(disp_name, cv2.WINDOW_NORMAL)
    cv2.imshow(disp_name,img); 
    





def get_demons_warp(moving, fixed):
    sx, sy, vx, vy = demons.demons(fixed, moving)
    warped = demons.iminterpolate(moving, sx=sx, sy=sy)
    from IPython import embed

    # embed()
    thresh = 18
    # remove_noise_ksize = 1

    fixed_float = fixed.astype(np.float32)
    moving_float = moving.astype(np.float32)
    warped_float = warped.astype(np.float32)
 
    total_diff = abs(fixed_float-moving_float)  ## total difference
    changed_pix = abs(fixed_float-warped_float)  ## pixel change
    displaced_pix = abs(total_diff-changed_pix)  ## only displacement

    total_diff_thresh = cv2.threshold(total_diff,thresh,255, cv2.THRESH_BINARY)[1]
    changed_pix_thresh = cv2.threshold(changed_pix,thresh,255, cv2.THRESH_BINARY)[1]
    displaced_pix_thresh = cv2.threshold(displaced_pix,thresh,255, cv2.THRESH_BINARY)[1]

    changed_pix_thresh = remove_noise(changed_pix_thresh,ksize=2)
    # displaced_pix_thresh = remove_noise(displaced_pix_thresh,ksize=1)

    ## Total difference
    fixed_color = np.dstack((fixed,fixed,fixed))
    totalDiff_img = mask_cv2_img_by_arr(fixed_color,np.where(total_diff_thresh), c=[0,255,0])
    save_and_disp(totalDiff_img,'totalDiff_img')

    ## Disentangled difference
    fixed_color = np.dstack((fixed,fixed,fixed))
    displacement_img = mask_cv2_img_by_arr(fixed_color,np.where(displaced_pix_thresh), c=[0,255,255])
    disentangled_img = mask_cv2_img_by_arr(displacement_img,np.where(changed_pix_thresh), c=[0,0,255])

    save_and_disp(disentangled_img,'disentangled_img')
 
    cv2.waitKey(0)

if __name__ == "__main__":
    # load data

    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   

    parser.add_argument('-fixed', '--fixed_path',
                        # type=str, default='../data/pydemons_data/lenag2.png',
                        type=str, default='../data/pydemons_data/lenag2.png',
                        help='fixed Image')
    parser.add_argument('-moving', '--moving_path',
                        type=str, default='../data/pydemons_data/lenag1.png',
                        help='moving')
    
    args = parser.parse_args()


    fixed = cv2.imread(args.fixed_path, 0)
    moving = cv2.imread(args.moving_path, 0)

    # run_demons(moving, fixed)
    get_demons_warp(moving, fixed)
