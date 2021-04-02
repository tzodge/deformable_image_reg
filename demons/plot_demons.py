import numpy as np
import matplotlib.pyplot as plt
from pydemons import demons
import argparse
import cv2


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
    return demons(fixed, moving, callback=_callback, **kwargs)
    # demons.demons(fixed, moving, callback=_callback, **kwargs)

if __name__ == "__main__":
    # load data

    parser = argparse.ArgumentParser(
        description='Image pair pose estimation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   

    parser.add_argument('-fixed', '--fixed_path',
                        type=str, default='../data/pydemons_data/lenag2.png',
                        help='fixed Image')
    parser.add_argument('-moving', '--moving_path',
                        type=str, default='../data/pydemons_data/lenag1.png',
                        help='moving')
    
    args = parser.parse_args()


    fixed = cv2.imread(args.fixed_path, 0)
    moving = cv2.imread(args.moving_path, 0)

    run_demons(moving, fixed)
