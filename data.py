import numpy as np
import pandas as pd
from patchify import patchify
import tifffile as tiff
import time
import os

TRAIN_FRACTION = 0.75

def load_data(organelle_name='Mitochondria'):

    base_dir = '/storage/users/assafzar'
    fovs_dir = os.path.join(base_dir, 'fovs')
    image_save_dir = os.path.join(base_dir, 'Raz')
    patches_save_dir = os.path.join(image_save_dir, 'data/', organelle_name)

    data_save_path_train = '{}/image_list_train.csv'.format(image_save_dir)
    data_save_path_tets = '{}/image_list_test.csv'.format(image_save_dir)

    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

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
    brightfield_images = []
    fluorescent_images = []
    # Iterate over the rows in the data frame
    for index, row in df.iterrows():
        image_stack = tiff.imread(row['path_tiff'])
        bf_channel = row['bf_channel']
        fluor_channel = row['org_channel']
        mid_slice = np.int(0.5 * image_stack.shape[0])
        bf_image = image_stack[mid_slice, bf_channel, :, :]
        fluor_image = image_stack[mid_slice, fluor_channel, :, :]
        brightfield_images.append(bf_image)
        fluorescent_images.append(fluor_image)

        # print(bf_image.shape)
        # bf_patches = patchify(bf_image, (52, 77), step=52)
        # print(bf_patches.shape)
        # for i in range(bf_patches.shape[0]):
        #     for j in range(bf_patches.shape[1]):
        #         print(f'Current: {i}, {j}, total: {bf_patches.shape[0]}, {bf_patches.shape[1]}')
        #         single_patch_bf_img = bf_patches[i, j, :, :]
        #         tiff.imwrite(patches_save_dir + '/bf/image_' + str('first') + '_' + str(i) + str(j) + '.tiff',
        #                      single_patch_bf_img)
        #         # single_patch_org_img = org_image[i, j, :, :]
        #         # tiff.imwrite(patches_save_dir + '/org/image_' + str('first') + '_' + str(i) + str(j) + '.tiff',
        #         #              single_patch_org_img)

    return brightfield_images, fluorescent_images


if __name__ == '__main__':
    brightfield_images, fluorescent_images = load_data()
