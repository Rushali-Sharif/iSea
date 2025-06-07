import os
import time
import ntpath
import numpy as np
from PIL import Image
from os.path import join, exists, basename
from keras.models import model_from_json

# Function to find 'test/images' folders
def find_test_images_folders(root_dir):
    test_image_folders = []
    for subdir, dirs, files in os.walk(root_dir):
        if basename(subdir) == 'images' and 'test' in subdir:
            test_image_folders.append(subdir)
    return test_image_folders

# Define root directory
root_dir = "D:/objectDetection/objectDetection/"

# Find all 'test/images' folders
test_images_dirs = find_test_images_folders(root_dir)
if not test_images_dirs:
    print("No 'test/images' folders found.")
else:
    print(f"Found {len(test_images_dirs)} 'test/images' folders.")

    # Model paths
    checkpoint_dir = 'models/gen_p/'
    model_name_by_epoch = "model_15320_"
    model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"
    model_json = checkpoint_dir + model_name_by_epoch + ".json"
    assert (exists(model_h5) and exists(model_json))

    # Load model
    with open(model_json, "r") as json_file:
        loaded_model_json = json_file.read()
    funie_gan_generator = model_from_json(loaded_model_json)
    funie_gan_generator.load_weights(model_h5)
    print("\nLoaded model")

    # Functions to preprocess and deprocess images
    def preprocess(img):
        img = img / 127.5 - 1
        return img

    def deprocess(img):
        img = (img + 1) * 127.5
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    # Process each dataset
    for images_dir in test_images_dirs:
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(images_dir)))
        samples_dir = os.path.join("../data/testEnhanced/", dataset_name)

        # Create directory for enhanced images
        if not exists(samples_dir):
            os.makedirs(samples_dir)

        test_paths = [join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print("{0} test images are loaded from {1}".format(len(test_paths), dataset_name))

        # Testing loop
        times = []
        for img_path in test_paths:
            try:
                inp_img = Image.open(img_path).resize((256, 256)).convert('RGB')
                im = preprocess(np.array(inp_img))
                im = np.expand_dims(im, axis=0)  # (1,256,256,3)

                s = time.time()
                gen = funie_gan_generator.predict(im)
                gen_img = deprocess(gen)[0]
                tot = time.time() - s
                times.append(tot)

                img_name = ntpath.basename(img_path)
                out_img = gen_img.astype('uint8')
                Image.fromarray(out_img).save(join(samples_dir, img_name))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Some statistics    
        num_test = len(test_paths)
        if num_test == 0:
            print(f"\nFound no images for test in {dataset_name}")
        else:
            print(f"\nTotal images in {dataset_name}: {num_test}") 
            Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
            print("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
            print(f"\nSaved generated images for {dataset_name} in {samples_dir}\n")
