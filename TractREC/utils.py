# -*- coding: utf-8 -*-
"""
Created on Thu Feb  23, 2017
utility functions
@author: Christopher Steele
"""

from __future__ import division  # to allow floating point calcs of number of voxels

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

def cube_mask_test(mask_img, cubed_subset_dim, max_num_labels_per_mask = None, start_idx = 1, out_file_base = None):
    import nibabel as nb
    import numpy as np

    import os
    if out_file_base is None:
        import os
        out_file_base = os.path.join(os.path.dirname(mask_img),os.path.basename(mask_img).split(".")[0]+"_index_label")

    img = nb.loadsave.load(mask_img)
    d = img.get_data()
    aff = img.affine
    header = img.header

    cubed_3d = get_cubed_array_labels_3d(np.shape(d), cubed_subset_dim).astype(np.uint32)
    d = np.multiply(d, cubed_3d) # apply the cube to the data

    #extremely fast way to replace values, suggested here: http://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
    palette = np.unique(d) #INCLUDES 0
    key = np.arange(0,len(palette))
    index = np.digitize(d.ravel(), palette, right=True)
    d = key[index].reshape(d.shape)

    unique = np.unique(d)
    non_zero_labels = unique[np.nonzero(unique)]

    print(str(np.max(d))+ " unique labels")

    if (max_num_labels_per_mask is not None) and (max_num_labels_per_mask < len(non_zero_labels)): #we cut things up
        import itertools

        all_out_files = []
        num_sub_arrays = int(np.ceil(len(non_zero_labels) / (max_num_labels_per_mask / 2)))
        cube_labels_split = np.array_split(non_zero_labels, num_sub_arrays)

        all_sets = list(itertools.combinations(np.arange(0, num_sub_arrays), 2))
        print("There are {} mask combinations to be created.".format(len(all_sets)))
        #return all_sets, num_sub_arrays, cube_labels_split
        for set in all_sets:
            print(set)
            superset = np.concatenate((cube_labels_split[set[0]], cube_labels_split[set[1]]), axis=0) #contains labels
            d_temp = np.zeros(d.shape)
            new_idx = start_idx

            for label_idx in superset: #this is extremely slow :-(
                d_temp[d == label_idx] = new_idx
                new_idx +=1

            tail = "_subset_" + str(set[0]) + "_" + str(set[1])
            out_file = out_file_base + tail + ".nii.gz"
            out_file_lut = out_file_base + tail + "_coords.csv"
            print(out_file)
            print(out_file_lut)

            img_out = nb.Nifti1Image(d_temp.astype(np.uint64), aff, header=header)
            img_out.set_data_dtype("uint64")
            # print("Max label value/num voxels: {}".format(str(start_idx)))
            nb.loadsave.save(img_out, out_file)
            np.savetxt(out_file_lut, superset, delimiter=",", fmt="%d") #not the voxel locations, just the LUT TODO:change!
            all_out_files.append(out_file)
    else:
        all_out_files = out_file_base+"_all.nii.gz"
    img = nb.Nifti1Image(d,aff,header)
    img.set_data_dtype("uint64")
    nb.save(img,out_file_base+"_all.nii.gz")
    print(out_file_base+"_all.nii.gz")
    return all_out_files

def mask2labels_multifile(mask_img, out_file_base = None, max_num_labels_per_mask = 1000, output_lut_file = False,
                          decimals = 2, start_idx = 1, coordinate_space = "scanner", cubed_subset_dim = None):
    """
    Convert simple binary mask to voxels that are labeled from 1..n.
    Outputs as uint32 in the hopes that you don't have over the max (4294967295)
    (i don't check, that is a crazy number of voxels...)
    :param mask_img:        any 3d image format that nibabel can read
    :param out_file_base:   nift1 format, base file name will be appended with voxel parcels _?x?.nii.gz
    :param output_lut_file  ouptut a lut csv file for all voxels True/False (this could be large!)
    :param decimals:        number of decimals for output lut file
    :param start_idx:       value to start index at, normally =1, unless you are combining masks...?
    :return:
    """

    import nibabel as nb
    import numpy as np
    import itertools

    if out_file_base is None:
        import os
        out_file_base = os.path.join(os.path.dirname(mask_img),os.path.basename(mask_img).split(".")[0]+"_index_label")

    img = nb.loadsave.load(mask_img)
    d = img.get_data().astype(np.uint64)
    aff = img.affine
    header = img.header

    all_vox_locs = np.array(np.where(d==1)).T
    num_vox = np.shape(all_vox_locs)[0]

    if cubed_subset_dim is not None and cubed_subset_dim > 1:
        print("Generating cubed subsets of your binary input mask")
        cubed_3d = get_cubed_array_labels_3d(np.shape(d),cubed_subset_dim)
        d = np.multiply(d, cubed_3d).astype(np.uint32) #apply the cube to the data
        # #print(np.unique(d))
        # for idx, val in enumerate(np.unique(d)):
        #     d[d==val] = idx + start_idx #move back to values based on start_idx (usually 1)

        #extremely fast way of re-assigning values
        palette = np.unique(d)
        key = np.arange(0, len(palette)) + start_idx - 1 #offset as required
        key[0] = 0 #retain 0 as the first index, since this is background
        key[0] = 0 #retain 0 as the first index, since this is background
        index = np.digitize(d.ravel(), palette, right=True)
        d = key[index].reshape(d.shape)

        num_sub_arrays = int(np.ceil(max_num_labels_per_mask / 2)) #just use this value, since we will use the sub-arrays not individual voxels
        cube_label_idxs = np.array_split(np.unique(d)[np.nonzero(np.unique(d))],num_sub_arrays)
        d_orig = np.copy(d)

    else: #we are doing this voxel-wise, go for whole hog!
        num_sub_arrays = int(np.ceil(num_vox / (max_num_labels_per_mask / 2)))
        sub_vox_locs = np.array_split(all_vox_locs, num_sub_arrays)

    out_file_names = []
    out_file_lut_names = []
    all_sets = list(itertools.combinations(np.arange(0,num_sub_arrays),2))
    print("Total number of combinations: {}".format(len(all_sets)))

    for subset in all_sets: #TODO: fix for cubed subsets, since it does not work :-(
        fir = subset[0]
        sec = subset[1]
        tail = "_label_subset_" + str(fir) + "_" + str(sec)
        out_file = out_file_base + tail + ".nii.gz"
        out_file_lut = out_file_base + tail + "_coords.csv"
        print(out_file)
        print(out_file_lut)

        d[d > 0] = 0  # don't need this data array anymore, so zero it and re-use
        label_idx = start_idx
        #return d_orig, subset, all_sets, fir, sec, cube_label_idxs
        if cubed_subset_dim is not None and cubed_subset_dim > 1:
            #we asked for cubes, so use them but have to refer to the volumetric cube data rather than the voxel locations
            superset = np.concatenate((cube_label_idxs[fir], cube_label_idxs[sec]), axis = 0)
            for superset_idx in superset:
                d[d_orig == superset_idx] = label_idx
                label_idx += 1
            d = d.astype(np.uint64)
        else:
            superset = np.concatenate((sub_vox_locs[fir], sub_vox_locs[sec]), axis = 0)
            for vox in superset:
                d[vox[0], vox[1], vox[2]] = label_idx
                label_idx += 1

        img_out = nb.Nifti1Image(d, aff, header=header)
        img_out.set_data_dtype("uint64")
        #print("Max label value/num voxels: {}".format(str(start_idx)))
        nb.loadsave.save(img_out, out_file)

        if coordinate_space is "voxel":
            np.savetxt(out_file_lut, superset, delimiter=",", fmt="%d")
            # return vox_coord
        elif coordinate_space is "scanner":
            scanner_coord = np.round(nb.affines.apply_affine(aff, superset), decimals=decimals)
            np.savetxt(out_file_lut, scanner_coord, delimiter=",", fmt="%." + str(decimals) + "f")

        out_file_names.append(out_file)
        out_file_lut_names.append(out_file_lut)

    return out_file_names, out_file_lut_names, sub_vox_locs

def get_cubed_array_labels_3d(shape, cube_subset_dim = 10):
    """
    Break a 3d array into cubes of cube_dim. Throw away extras if array is not a perfect cube
    :param shape:              - 3d matrix shape
    :param cube_subset_dim:    - size, in voxels, of one dimension of cube
    :return:                   - matrix of labeled cubes (or appx) of size cube_dim*cube_dim*cube_dim
    """
    import numpy as np


    #we make the matrix cubic to make the calculations easier, toss out the extras at the end
    max_dim = np.max(shape)
    num_cubes_per_dim = np.ceil(max_dim / cube_subset_dim).astype(int)
    d = np.zeros((max_dim,max_dim,max_dim))

    #determine the size of each cube based on the number of cubes that we will cut the supercube into (yes, this is basically the reverse of above)
    x_span = np.ceil(max_dim / num_cubes_per_dim).astype(int)
    y_span = x_span
    z_span = x_span
    print("Voxel span for single cube dimension: {}".format(x_span))

    cube_idx = 0
    for ix in np.arange(0, num_cubes_per_dim):
        for iy in np.arange(0, num_cubes_per_dim):
            for iz in np.arange(0, num_cubes_per_dim):
                cube_idx += 1
                x0 = ix*x_span
                y0 = iy*y_span
                z0 = iz*z_span
                d[x0 : x0 + x_span, y0 : y0 + y_span, z0 : z0 + z_span] = cube_idx
    return (d[0:shape[0],0:shape[1],0:shape[2]]).astype(np.uint32) #return only the dims that we requested, discard the extras at the edges

def gmwmvox2mesh(mask_img, mesh_format = "obj"):
    from skimage import measure
    import nibabel as nb
    img = nb.load(mask_img)
    d = img.get_data()
    aff = img.affine
    header = img.header

    verts, faces = measure.marching_cubes(d,0)
    return verts, faces

def mask2labels(mask_img, out_file = None, output_lut_file = False, decimals = 2, start_idx = 1):
    """
    Convert simple binary mask to voxels that are labeled from 1..n.
    Outputs as uint32 in the hopes that you don't have over the max (4294967295)
    (i don't check, that is a crazy number of voxels!)
    :param mask_img:        any 3d image format that nibabel can read
    :param out_file:        nift1 format
    :param output_lut_file  ouptut a lut csv file for all voxels True/False (this could be large!)
    :param decimals:        number of decimals for output lut file
    :param start_idx:       value to start index at, normally =1, unless you are combining masks...?
    :return:
    """
    import nibabel as nb
    import numpy as np

    if out_file is None:
        import os
        out_file = os.path.join(os.path.dirname(mask_img),os.path.basename(mask_img).split(".")[0]+"_index_label.nii.gz")

    img = nb.loadsave.load(mask_img)
    d = img.get_data().astype(np.uint64)
    aff = img.affine
    header = img.header

    vox_locs = np.array(np.where(d==1)).T
    for vox in vox_locs:
        d[vox[0], vox[1], vox[2]] = start_idx
        start_idx += 1

    if output_lut_file:
        lut_file = os.path.join(os.path.dirname(mask_img),
                                os.path.basename(mask_img).split(".")[0] + "_index_label_lut.csv")

        lut = np.zeros((np.shape(vox_locs)[0], np.shape(vox_locs)[1] + 1))
        lut[:,1:] = nb.affines.apply_affine(aff,vox_locs)
        lut[:, 0] = np.arange(1, np.shape(vox_locs)[0] + 1)
        np.savetxt(lut_file, lut, header = "index,x_coord,y_coord,z_coord",delimiter=",",fmt="%." + str(decimals) +"f")

    img_out = nb.Nifti1Image(d, aff, header=header)
    img_out.set_data_dtype("uint64")
    print("Max label value/num voxels: {}".format(str(start_idx)))
    nb.loadsave.save(img_out,out_file)
    return out_file, start_idx

def combine_and_label_2masks(mask1,mask2, out_file1 = None, out_file2 = None, output_lut_files = False, decimals = 2, start_idx = 1):
    import os
    import nibabel as nb
    out_file1, end_idx = mask2labels(mask1, out_file = out_file1 , output_lut_file = output_lut_files , decimals = decimals, start_idx = start_idx)
    out_file2, end_idx = mask2labels(mask2, out_file = out_file2 , output_lut_file = output_lut_files , decimals = decimals, start_idx = end_idx)
    f1 = nb.load(out_file1)
    f2 = nb.load(out_file2)
    d_f1 = f1.get_data()
    d_f2 = f2.get_data()
    d_f2[d_f1>0] = d_f1[d_f1>0]
    out_img = nb.Nifti1Image(d_f2,f2.affine,header=f2.header)
    out_file = os.path.join(os.path.dirname(mask1),os.path.basename(mask1).split(".")[0]+"_joined_index_label.nii.gz")
    nb.save(out_img,out_file)

## experimental ##
def get_distance(f):
    """Return the signed distance to the 0.5 levelset of a function.
    https://github.com/pmneila/morphsnakes/issues/5
    """
    import numpy as np
    import scipy.ndimage as ndimage
    # Prepare the embedding function.
    f = f > 0

    # Signed distance transform
    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, dist_func(f) - 0.5, -(dist_func(1-f) - 0.5))

    return distance

def exclude_from_outside_boundary(skeleton_img, mask_img, distance = 0):
    from scipy import ndimage
    import nibabel as nb
    import numpy as np

    img_m = nb.loadsave.load(mask_img)
    d_m = img_m.get_data()
    aff_m = img_m.affine
    header_m = img_m.header

    img_s = nb.loadsave.load(skeleton_img)
    d_s = img_s.get_data().astype('float32')
    aff_s = img_s.affine
    header_s = img_s.header

    #d_out = ndimage.distance_transform_edt(1-d_s).astype('float32') #use 1-d_s to flip 0s and 1s
    d_out = get_distance(1-d_s)
    out_img = nb.Nifti1Image(d_out,aff_s,header=header_s)
    out_name = "XXX_temp.nii.gz"
    nb.save(out_img,out_name)
    return out_name