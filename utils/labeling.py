import os
import numpy as np
from PIL import Image

def labeling(label_pre_path, label_post_path):
    label_pre = Image.open(label_pre_path)
    label_post = Image.open(label_post_path)

    label_pre = np.array(label_pre)
    label_post = np.array(label_post)

    labeled_mask = np.zeros_like(label_pre, dtype=np.uint8)
    labeled_mask[(label_pre == 255) & (label_post == 255)] = 1 # undamaged road
    labeled_mask[(label_pre == 255) & (label_post != 255)] = 2 # damaged road

    # color mask visualization
    color_mask = np.zeros((*label_pre.shape, 3), dtype=np.uint8)
    color_mask[labeled_mask == 1] = [0, 255, 0] # green for undamaged road
    color_mask[labeled_mask == 2] = [255, 0, 0] # red for damaged road

    color_mask = Image.fromarray(color_mask, mode='RGB')
    # labeled_mask = Image.fromarray(labeled_mask, mode='L')

    return color_mask

def main(pre_dir, post_dir, save_dir):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    lst_pre_label = [f for f in os.listdir(pre_dir) if f.startswith('test_pred_batch')]

    for pre_label_filename in lst_pre_label:
        label_pre_path = os.path.join(pre_dir, pre_label_filename)
        label_post_path = os.path.join(post_dir, pre_label_filename)
        
        labeled_mask = labeling(label_pre_path, label_post_path)

        labeled_mask.save(os.path.join(save_dir, pre_label_filename))

if __name__ == '__main__':
    pre_dir = './png-pre'
    post_dir = './png-post'
    save_dir = './png-color'
    main(pre_dir, post_dir, save_dir)