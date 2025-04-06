import os
from utils.tar import extract_tar

dir_data = './xbd/road'
save_data = './datasets'

if not os.path.exists(save_data):
    os.makedirs(save_data)

extract_tar(os.path.join(dir_data, 'road_train_images_targets_512.tar'), save_data)
extract_tar(os.path.join(dir_data, 'road_hold_images_targets_512.tar'), save_data)
extract_tar(os.path.join(dir_data, 'road_test_images_targets_512.tar'), save_data)

lst_train_images = [f for f in os.listdir(os.path.join(save_data, 'train', 'images')) if f.endswith('.png')]
lst_train_targets = [f for f in os.listdir(os.path.join(save_data, 'train', 'targets')) if f.endswith('.png')]
lst_hold_images = [f for f in os.listdir(os.path.join(save_data, 'hold', 'images')) if f.endswith('.png')]
lst_hold_targets = [f for f in os.listdir(os.path.join(save_data, 'hold', 'targets')) if f.endswith('.png')]
lst_test_images = [f for f in os.listdir(os.path.join(save_data, 'test', 'images')) if f.endswith('.png')]
lst_test_targets = [f for f in os.listdir(os.path.join(save_data, 'test', 'targets')) if f.endswith('.png')]

lst_train_images.sort()
lst_train_targets.sort()
lst_hold_images.sort()
lst_hold_targets.sort()
lst_test_images.sort()
lst_test_targets.sort()

print(f"Number of training images: {len(lst_train_images)}")
print(f"Number of training targets: {len(lst_train_targets)}")
print(f"Number of hold images: {len(lst_hold_images)}")
print(f"Number of hold targets: {len(lst_hold_targets)}")
print(f"Number of test images: {len(lst_test_images)}")
print(f"Number of test targets: {len(lst_test_targets)}")

print(lst_train_images[0])
print(lst_train_targets[0])
print(lst_hold_images[0])
print(lst_hold_targets[0])
print(lst_test_images[0])
print(lst_test_targets[0])