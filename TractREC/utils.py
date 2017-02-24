# -*- coding: utf-8 -*-
"""
Created on Thu Feb  23, 2017
utility functions
@author: Christopher Steele
"""

def mask2voxelList(mask_img, out_file = None, coordinate_space = 'scanner', mask_threshold = 0, decimals = 2):
    """
    Calculate coordinates for all voxels greater than mask threshold
    :param bin_mask:
    :param out_file:
    :param coordinate_space:    {'scanner', 'voxel'} - default is 'scanner', where each voxel center is returned after applying the affine matrix
    :param mask_threshold:      values lower or equal to this are set to 0, greater set to 1
    :param decimals:            decimals for rounding coordinates

    :return: voxel_coordinates
    """

    from nibabel.loadsave import load as imgLoad
    from nibabel.affines import apply_affine
    import os
    import numpy as np

    if out_file is None:
        out_dir = os.path.dirname(mask_img)
        out_name = os.path.basename(mask_img).split(".")[0] + "_" + coordinate_space + "_coords.csv" #take up to the first "."
        out_file = os.path.join(out_dir,out_name)
    #import the data
    img = imgLoad(mask_img)
    d = img.get_data()
    aff = img.affine

    #binarise the data
    d[d<=mask_threshold] = 0
    d[d>mask_threshold] = 1

    vox_coord = np.array(np.where(d == 1)).T # nx3 array of values of three dimensions

    if coordinate_space is "voxel":
        np.savetxt(out_file, vox_coord, delimiter=",",fmt="%d")
        #return vox_coord
    elif coordinate_space is "scanner":
        scanner_coord = np.round(apply_affine(aff, vox_coord),decimals=decimals)
        np.savetxt(out_file, scanner_coord, delimiter=",",fmt="%." + str(decimals) +"f")
        #return scanner_coord
    return out_file

def mask2labels(mask_img, out_file = None, output_lut_file = False, decimals = 2):
    """
    Convert simple binary mask to voxels that are labeled from 1..n.
    Outputs as uint32 in the hopes that you don't have over the max (4294967295)
    (i don't check, that is a crazy number of voxels!)
    :param mask_img:        any 3d image format that nibabel can read
    :param out_file:        nift1 format
    :param output_lut_file  ouptut a lut csv file for all voxels True/False (this could be large!)
    :param decimals:        number of decimals for output lut file
    :return:
    """
    import nibabel as nb
    import numpy as np

    if out_file is None:
        import os
        out_file = os.path.join(os.path.dirname(mask_img),os.path.basename(mask_img).split(".")[0]+"_index_label.nii.gz")

    img = nb.loadsave.load(mask_img)
    d = img.get_data().astype(np.uint32)
    aff = img.affine
    header = img.header


    vox_locs = np.array(np.where(d==1)).T
    idx = 1
    for vox in vox_locs:
        d[vox[0], vox[1], vox[2]] = idx
        idx += 1

    if output_lut_file:
        lut_file = os.path.join(os.path.dirname(mask_img),
                                os.path.basename(mask_img).split(".")[0] + "_index_label_lut.csv")

        lut = np.zeros((np.shape(vox_locs)[0], np.shape(vox_locs)[1] + 1))
        lut[:,1:] = nb.affines.apply_affine(aff,vox_locs)
        lut[:, 0] = np.arange(1, np.shape(vox_locs)[0] + 1)
        np.savetxt(lut_file, lut, header = "index,x_coord,y_coord,z_coord",delimiter=",",fmt="%." + str(decimals) +"f")

    img_out = nb.Nifti1Image(d, aff, header=header)
    img_out.set_data_dtype("uint32")
    print("Max label value/num voxels: {}".format(str(idx)))
    nb.loadsave.save(img_out,out_file)
    return out_file