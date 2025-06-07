import os
import shutil

def merge_folders(root_input_dir, target_dir):
    # Define the main folders (train, test, valid) and subfolders (images, labels)
    main_folders = ['train', 'test', 'valid']
    subfolders = ['images', 'labels']

    # Ensure target subfolders exist in the target directory
    for main_folder in main_folders:
        for subfolder in subfolders:
            target_subfolder = os.path.join(target_dir, main_folder, subfolder)
            if not os.path.exists(target_subfolder):
                os.makedirs(target_subfolder)

    # Loop through each folder inside the root input directory
    for folder_name in os.listdir(root_input_dir):
        current_folder_path = os.path.join(root_input_dir, folder_name)

        if os.path.isdir(current_folder_path):  # Ensure it's a directory
            for main_folder in main_folders:
                source_main_folder = os.path.join(current_folder_path, main_folder)

                # Check if "train", "test", or "valid" exists in the current folder
                if os.path.exists(source_main_folder):
                    for subfolder in subfolders:
                        source_subfolder = os.path.join(source_main_folder, subfolder)
                        target_subfolder = os.path.join(target_dir, main_folder, subfolder)

                        # Check if the subfolder (images/labels) exists in the source main folder
                        if os.path.exists(source_subfolder):
                            for file_name in os.listdir(source_subfolder):
                                source_file = os.path.join(source_subfolder, file_name)
                                target_file = os.path.join(target_subfolder, file_name)
                                
                                # Check if the file already exists in the target location
                                if os.path.exists(target_file):
                                    print(f"Skipping {file_name} as it already exists in {target_subfolder}")
                                else:
                                    # Copy files from the source to the target directory
                                    if os.path.isfile(source_file):
                                        shutil.copy2(source_file, target_file)

if __name__ == "__main__":
   # Input directory containing folders with "train", "test", and "valid" subfolders
    input_directory = 'D:/objectDetection/objectDetection/'

    # Target directory where the merged folders will be created
    target_directory = 'D:/objectDetection/objectDetection/FinalObjectDetection/'

    merge_folders(input_directory, target_directory)
    print("Contents merged successfully!")
