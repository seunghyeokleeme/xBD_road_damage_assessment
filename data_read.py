import os
import tarfile

def extract_tar(tar_path, extract_to_path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to_path)
    print(f"{extract_to_path} path is extracted")

def compress_to_tar(output_filename, source_path):
    with tarfile.open(output_filename, "w") as tar:
        tar.add(source_path, arcname=os.path.basename(source_path))
    print(f"{output_filename} is created")

if __name__ == "__main__":
    train_source = "./datasets/test"
    train_output = "road_test_images_targets_512.tar"
    compress_to_tar(train_output, train_source)