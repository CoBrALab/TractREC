# -*- coding: utf-8 -*-
"""
Created on Thu Feb  23, 2017
Compute large connectomes using mrtrix and sparse matrices
@author: Christopher Steele
"""

from __future__ import division  # to allow floating point calcs of number of voxels


def natural_sort(l):
    """
    Returns alphanumerically sorted input
    #natural sort from the interwebs (http://stackoverflow.com/questions/11150239/python-natural-sorting)
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


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

def generate_cubed_masks(mask_img, cubed_subset_dim, max_num_labels_per_mask = None, start_idx = 1, out_file_base = None):
    """
    Generate cubes of unique indices to cover the entire volume, multiply them by your binary mask_img, then split up
    into multiple mask node files of no more than max_num_labels_per_mask (for mem preservation) and re-index each to
    start at start_idx (default = 1, best to stick with this).

    Saves each combination of subsets of rois as *index_label_?_?.nii.gz, along with

    :param mask_img:
    :param cubed_subset_dim:
    :param max_num_labels_per_mask:
    :param start_idx:
    :param out_file_base:
    :return:
    """
    import nibabel as nb
    import numpy as np

    if out_file_base is None:
        out_file_base = mask_img.split(".")[0]+"_index_label"

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

    print(str(np.max(d))+ " unique labels (including 0)")

    if (max_num_labels_per_mask is not None) and (max_num_labels_per_mask < len(non_zero_labels)): #we cut things up
        import itertools
        all_out_files = []
        all_out_files_luts = []
        num_sub_arrays = int(np.ceil(len(non_zero_labels) / (max_num_labels_per_mask / 2)))
        cube_labels_split = np.array_split(non_zero_labels, num_sub_arrays)

        all_sets = list(itertools.combinations(np.arange(0, num_sub_arrays), 2))
        print("There are {} mask combinations to be created.".format(len(all_sets)))
        #return all_sets, num_sub_arrays, cube_labels_split
        for set in all_sets:
            superset = np.concatenate((cube_labels_split[set[0]], cube_labels_split[set[1]]), axis=0) #contains labels
            d_temp = np.zeros(d.shape)
            new_idx = start_idx

            # this has been checked, and returns identical indices as with a simple loop
            all_idxs = np.copy(non_zero_labels)
            all_idxs[np.logical_not(np.in1d(non_zero_labels, superset))] = 0 #where our indices are not in the superset, set to 0
            all_idxs[np.in1d(non_zero_labels, superset)] = np.arange(0,len(superset)) + start_idx #where they are in the index, reset them to increasing
            key = all_idxs #this is the vector of values that will be populated into the matrix
            palette = non_zero_labels
            index = np.digitize(d.ravel(), palette, right=True)
            d_temp = key[index].reshape(d.shape)
            d_temp[d==0] = 0 #0 ends up being set to 1 because of edge case, so set them all back to 0

            # for label_idx in superset: #this is extremely slow :-(
            #     d_temp[d == label_idx] = new_idx
            #     new_idx +=1

            tail = "_subset_" + str(set[0]) + "_" + str(set[1])
            out_file = out_file_base + tail + ".nii.gz"
            out_file_lut = out_file_base + tail + "_coords.csv"
            print("\n"+"Subset includes {} non-zero labels.".format(len(superset)))
            print(set)
            print(out_file)
            print(out_file_lut)

            img_out = nb.Nifti1Image(d_temp.astype(np.uint64), aff, header=header)
            img_out.set_data_dtype("uint64")
            # print("Max label value/num voxels: {}".format(str(start_idx)))
            nb.loadsave.save(img_out, out_file)
            np.savetxt(out_file_lut, superset, delimiter=",", fmt="%d",header="value") #not the voxel locations, just the LUT TODO:change!
            all_out_files.append(out_file)
            all_out_files_luts.append(out_file_lut)
    else:
        all_out_files = out_file_base+"_all.nii.gz"
        all_out_files_luts = None
    img = nb.Nifti1Image(d,aff,header)
    img.set_data_dtype("uint64")
    nb.save(img,out_file_base+"_all.nii.gz")
    print(out_file_base+"_all.nii.gz")
    return all_out_files, all_out_files_luts

def generate_cubed_masks_v2(mask_img, cubed_subset_dim = None, max_num_labels_per_mask = None, start_idx = 1, out_file_base = None, zfill_num = 4):
    """
    Generate cubes of unique indices to cover the entire volume, multiply them by your binary mask_img, then split up
    into multiple mask node files of no more than max_num_labels_per_mask (for mem preservation) and re-index each to
    start at start_idx (default = 1, best to stick with this).
    Appx 4900 nodes creates appx 44mb connectome file (text) and file grows as the square of node number, so 1.8x larger (8820) should be just under 1GB
        - but this will likely break pd.read_csv unless you write a line by line reader :-/
    Saves each combination of subsets of rois as *index_label_?_?.nii.gz, along with

    :param mask_img:
    :param cubed_subset_dim:
    :param max_num_labels_per_mask:
    :param start_idx:
    :param out_file_base:
    :param zfill_num:                   number of 0s to pad indices with so that filenames look nice (always use natural sort, however!)
    :return:
    """
    import nibabel as nb
    import numpy as np

    if out_file_base is None:
        out_file_base = mask_img.split(".")[0]+"_index_label"

    img = nb.loadsave.load(mask_img)
    d = img.get_data().astype(np.uint64)
    aff = img.affine
    header = img.header

    if cubed_subset_dim is not None:
        cubed_3d = get_cubed_array_labels_3d(np.shape(d), cubed_subset_dim).astype(np.uint32)
        d = np.multiply(d, cubed_3d) # apply the cube to the data

        #extremely fast way to replace values, suggested here: http://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
        palette = np.unique(d) #INCLUDES 0
        key = np.arange(0,len(palette))
        index = np.digitize(d.ravel(), palette, right=True)
        d = key[index].reshape(d.shape)
    else:
        all_vox_locs = np.array(np.where(d == 1)).T
        idx = start_idx
        for vox in all_vox_locs:
            d[vox[0], vox[1], vox[2]] = idx
            idx += 1

    unique = np.unique(d)
    non_zero_labels = unique[np.nonzero(unique)]
    print(non_zero_labels)
    print(str(np.max(d))+ " unique labels (including 0)")

    if (max_num_labels_per_mask is not None) and (max_num_labels_per_mask < len(non_zero_labels)): #we cut things up
        import itertools
        all_out_files = []
        all_out_files_luts = []
        num_sub_arrays = int(np.ceil(len(non_zero_labels) / (max_num_labels_per_mask / 2)))
        cube_labels_split = np.array_split(non_zero_labels, num_sub_arrays)

        all_sets = list(itertools.combinations(np.arange(0, num_sub_arrays), 2))
        print("There are {} mask combinations to be created.".format(len(all_sets)))
        #return all_sets, num_sub_arrays, cube_labels_split
        for set in all_sets:
            superset = np.concatenate((cube_labels_split[set[0]], cube_labels_split[set[1]]), axis=0) #contains labels
            d_temp = np.zeros(d.shape)
            new_idx = start_idx

            # this has been checked, and returns identical indices as with a simple loop
            all_idxs = np.copy(non_zero_labels)
            all_idxs[np.logical_not(np.in1d(non_zero_labels, superset))] = 0 #where our indices are not in the superset, set to 0
            all_idxs[np.in1d(non_zero_labels, superset)] = np.arange(0,len(superset)) + start_idx #where they are in the index, reset them to increasing
            key = all_idxs #this is the vector of values that will be populated into the matrix
            palette = non_zero_labels
            index = np.digitize(d.ravel(), palette, right=True)
            d_temp = key[index].reshape(d.shape)
            d_temp[d==0] = 0 #0 ends up being set to 1 because of edge case, so set them all back to 0

            # for label_idx in superset: #this is extremely slow :-(
            #     d_temp[d == label_idx] = new_idx
            #     new_idx +=1

            tail = "_subset_" + str(set[0]).zfill(zfill_num) + "_" + str(set[1]).zfill(zfill_num)
            out_file = out_file_base + tail + ".nii.gz"
            out_file_lut = out_file_base + tail + "_coords.csv"
            print("\n"+"Subset includes {} non-zero labels.".format(len(superset)))
            print(set)
            print(out_file)
            print(out_file_lut)

            img_out = nb.Nifti1Image(d_temp.astype(np.uint64), aff, header=header)
            img_out.set_data_dtype("uint64")
            # print("Max label value/num voxels: {}".format(str(start_idx)))
            nb.loadsave.save(img_out, out_file)
            np.savetxt(out_file_lut, superset, delimiter=",", fmt="%d",header="value") #not the voxel locations, just the LUT TODO:change!
            all_out_files.append(out_file)
            all_out_files_luts.append(out_file_lut)
    else:
        all_out_files = out_file_base+"_all.nii.gz"
        all_out_files_luts = None
    img = nb.Nifti1Image(d,aff,header)
    img.set_data_dtype("uint64")
    nb.save(img,out_file_base+"_all.nii.gz")
    print(out_file_base+"_all.nii.gz")
    return all_out_files, all_out_files_luts


def do_it_all(tck_file, node_file, weight_file = None, out_mat_file=None):
    # appx 5 hrs for dim=3, max labels=5k (without connectome generation
    from scipy import io
    if out_mat_file is None:
        out_mat_file = node_file.split(".")[0] + "_all_cnctm_mat_complete.mtx"
    cubed_masks, cubed_mask_luts = generate_cubed_masks_v2(node_file,cubed_subset_dim=3,max_num_labels_per_mask=5000)
    connectome_files = tck2connectome_collection(tck_file, cubed_masks, weight_file=weight_file)
    mat = combine_connectome_matrices_sparse(connectome_files,cubed_mask_luts)
    io.mmwrite(out_mat_file,mat)
    print("Full matrix stored to: {}".format(out_mat_file))
    return mat

def combine_connectome_matrices_sparse(connectome_files_list, connectome_files_index_list, label_max = None, connectome_files_index_master = None): #TODO: does not currently work
    """
    Assumes that labels start at 1 and end at label_max, if set to None, we read through each index file and calculate it
    :param connectome_files_list:
    :param connectome_files_index_list:
    :param label_max:
    :param connectome_files_index_master:
    :return:
    """

    import pandas as pd
    import numpy as np
    import scipy.sparse as sparse

    #first check the indices so you know how large things are
    if label_max is None:
        label_max = 0
        for idx, file in enumerate(connectome_files_index_list):
            label_idx = np.ndarray.flatten(pd.read_csv(file, header = 0).values) #read quickly, then break out of the array of arrays of dimension 1
            if np.max(label_idx) > label_max:
                label_max = np.max(label_idx)
    mat = sparse.lil_matrix((label_max+1,label_max+1)) #allow space for row and column IDs, and makes indexing super easy
    mat[0, :] = np.arange(0, label_max + 1)[:,np.newaxis].T # column id
    mat[:, 0] = np.arange(0, label_max + 1)[:,np.newaxis] # row id (labels)
    print("Connectome combination from {0} files in progress:".format(len(connectome_files_list)))

    #assume that the file list and the index list are in the same order, now we can build the matrix - USE NATURAL SORT!
    for idx, file in enumerate(connectome_files_list):
        print("{0}:\n\tmatrix: {1}\n\tindex : {2}".format(idx+1,file,connectome_files_index_list[idx]))
        label_idx = np.ndarray.flatten(pd.read_csv(connectome_files_index_list[idx], header = 0).values)
        lookup_col = np.in1d(mat[0,:].toarray(),label_idx)
        lookup_row = lookup_col.T
        #mask = lookup_row[:,None]*lookup_col[None,:] #broadcast to create a 2d matrix of mat.shape with true where data will go
        # mat[lookup_row,lookup_col] = pd.read_csv(file, sep = " ", header = None).values
        # mat[mask] = pd.read_csv(file, sep = " ", header = None).values #THIS DOES NOT WORK, casts to 1d
        #return mat, lookup_row,lookup_col,pd.read_csv(file, sep = " ", header = None).values
        mat[np.ix_(lookup_row,lookup_col)]  = pd.read_csv(file, sep = " ", header = None).values #this works (tested on small sub-matrices) but not sure if all cases are covered?
    return mat


def combine_connectome_matrices(connectome_files_list, connectome_files_index_list, connectome_files_index_master = None):
    import pandas as pd
    import numpy as np

    #first check the indices so you know how large things are
    label_max = 0
    for idx, file in enumerate(connectome_files_index_list):
        label_idx = np.ndarray.flatten(pd.read_csv(file, header = 0).values) #read quickly, then break out of the array of arrays of dimension 1
        if np.max(label_idx) > label_max:
            label_max = np.max(label_idx)

    mat = np.zeros((label_max+1,label_max+1),dtype=np.uint64) #allow space for row and column IDs, and makes indexing super easy
    mat[0, :] = np.arange(0, label_max + 1).T # column id
    mat[:, 0] = np.arange(0, label_max + 1) # row id (labels)

    #assume that the file list and the index list are in the same order, now we can build the matrix
    for idx, file in enumerate(connectome_files_list):
        label_idx = np.ndarray.flatten(pd.read_csv(connectome_files_index_list[idx], header = 0).values)
        lookup_col = np.in1d(mat[0,:],label_idx)
        lookup_row = lookup_col.T
        #mask = lookup_row[:,None]*lookup_col[None,:] #broadcast to create a 2d matrix of mat.shape with true where data will go
        #return mat, mask, lookup_col,lookup_row, pd.read_csv(file, sep = " ", header = None).values
        #mat = np.where(mask,pd.read_csv(file, sep = " ", header = None).values,mat)
        mat[np.ix_(lookup_row,lookup_col)] = pd.read_csv(file, sep = " ", header = None).values
    return mat

def tck2connectome_collection(tck_file, node_files, weight_file = None, nthreads = 8):
    import subprocess

    out_files = []
    if not isinstance(node_files,list):
        node_files = [node_files] #make iterable
    for idx, node_file in enumerate(node_files):
        out_file = node_file.split(".")[0] + "_cnctm_mat.txt"
        if weight_file is None:
            cmd = ["tck2connectome", tck_file, node_file, out_file, "-assignment_end_voxels", "-nthreads", str(nthreads), "-force"]
        else: #TODO: modify with final parameters to get WM voxel crossings
            cmd = ["/home/chris/Documents/code/mrtrix3_devel/bin/tck2connectome", tck_file, node_file, out_file, "-tck_weights_in", weight_file, "-assignment_end_voxels", "-nthreads", str(nthreads), "-force"]
        print("Generating: {}".format(out_file))
        print(" ".join(cmd))
        subprocess.call(cmd)
        out_files.append(out_file)
    return out_files

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
    return (d[0:shape[0],0:shape[1],0:shape[2]]).astype(np.uint64) #return only the dims that we requested, discard the extras at the edges

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
    Outputs as uint64 in the hopes that you don't have over the max (4294967295)
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

def plot_coo_matrix(m):
    """ Taken from: http://stackoverflow.com/questions/22961541/python-matplotlib-plot-sparse-matrix-pattern"""
    import matplotlib.pyplot as plt
    from scipy.sparse import coo_matrix

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(m.col, m.row, 's', color='white', ms=1)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig