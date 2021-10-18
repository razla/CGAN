import os
import config
import random
import numpy as np
import pandas as pd
import tifffile as tiff
from patchify import patchify
from PIL import Image
import cv2


def load_data(organelle_name='Mitochondria', download=True):

    base_dir = '/storage/users/assafzar'
    fovs_dir = os.path.join(base_dir, 'fovs')
    image_save_dir = os.path.join(base_dir, 'Raz', 'data/')
    patches_train_dir = os.path.join(image_save_dir, organelle_name, 'train/')
    patches_val_dir = os.path.join(image_save_dir, organelle_name, 'val/')

    data_save_path_train = '{}/image_list_train.csv'.format(image_save_dir)
    data_save_path_tets = '{}/image_list_test.csv'.format(image_save_dir)

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    if not os.path.exists(patches_train_dir):
        os.makedirs(patches_train_dir)

    if not os.path.exists(patches_val_dir):
        os.makedirs(patches_val_dir)

    # Read csv
    data_manifest = pd.read_csv(os.path.join(fovs_dir, 'metadata.csv'))

    # Indices of the unique FOVid
    unique_fov_indices = np.unique(data_manifest['FOVId'], return_index=True)[1]

    # Extract the data using the unique FOVid
    data_manifest = data_manifest.iloc[unique_fov_indices]

    # Only organelle_name data
    data_manifest = data_manifest.loc[
        data_manifest['StructureShortName'] == organelle_name]

    # Extracts the fovs path
    images_source_paths = data_manifest["SourceReadPath"]

    # Removes 'fovs/' prefix
    images_source_paths = [path.replace('fovs/', '') for path in images_source_paths]

    # Concatenates the right path
    images_source_paths = [os.path.join(fovs_dir, organelle_name, file) for file in images_source_paths]

    ###############################################
    # Make a new manifest of all the files in csv #
    ###############################################

    df = pd.DataFrame(columns=["path_tiff", "bf_channel", "org_channel"])

    df["path_tiff"] = images_source_paths
    df["bf_channel"] = data_manifest["ChannelNumberBrightfield"].values
    df["org_channel"] = data_manifest["ChannelNumberStruct"].values
    images = []

    # Iterate over the rows in the data frame
    for index, row in df.iterrows():
        try:
            if random.uniform(0, 1) < config.TRAIN_FRACTION:
                is_train = True
            else:
                is_train = False

            print(f'Image #{index}: training - {is_train}')

            # reading the tiff
            image_stack = tiff.imread(row['path_tiff'])

            # extracting the brightfield channel num
            bf_channel = row['bf_channel']

            # extracting the fluorescent channel num
            fluor_channel = row['org_channel']

            # taking the mid slice - best visualization
            mid_slice = np.int(0.5 * image_stack.shape[0])

            # taking one image of brightfield from the tiff
            bf_image = image_stack[mid_slice, bf_channel, :, :]

            # taking one image of fluorescent from the tiff
            fluor_image = image_stack[mid_slice, fluor_channel, :, :]

            # Patchify -> creating 180x180 patches
            if not download:
                images.append((bf_image, fluor_image))
            else:
                bf_patches = patchify(bf_image,
                                      (config.PATCH_SIZE, config.PATCH_SIZE),
                                      step=config.STEP_SIZE)
                fluor_patches = patchify(fluor_image,
                                         (config.PATCH_SIZE, config.PATCH_SIZE),
                                         step=config.STEP_SIZE)
                for i in range(bf_patches.shape[0]):
                    for j in range(bf_patches.shape[1]):
                        single_patch_bf_img = bf_patches[i, j, :, :]
                        single_patch_fluor_img = fluor_patches[i, j, :, :]

                        # Normalise to range 0..255
                        single_bf_norm_img = (single_patch_bf_img.astype(
                            np.float) - single_patch_bf_img.min()) * 255.0 / (
                                                     single_patch_bf_img.max() - single_patch_bf_img.min())

                        single_fluor_norm_img = (single_patch_fluor_img.astype(
                            np.float) - single_patch_fluor_img.min()) * 255.0 / (
                                                        single_patch_fluor_img.max() - single_patch_fluor_img.min())

                        concatenated_img = cv2.hconcat([single_bf_norm_img, single_fluor_norm_img])

                        if is_train:
                            # if not os.path.exists(patches_train_dir + '/' + str(index)):
                            #     os.makedirs(patches_train_dir + '/' + str(index))
                            path = patches_train_dir + '/' + str(index) + '_' + str(i) + '_' + str(j) + '.png'
                        else:
                            # if not os.path.exists(patches_val_dir + '/' + str(index)):
                            #     os.makedirs(patches_val_dir + '/' + str(index))
                            path = patches_val_dir + '/' + str(index) + '_' + str(i) + '_' + str(j) + '.png'

                        # Save as 8-bit PNG
                        Image.fromarray(concatenated_img.astype(np.uint8)).save(path)

        except Exception:
            pass
    return images


if __name__ == '__main__':
    images = load_data(organelle_name='Mitochondria')
