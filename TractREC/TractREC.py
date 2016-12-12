# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:07:32 2015

@author: Christopher J Steele (except for one that I took from stackoverflow ;-))
"""

#import preprocessing here?


def imgLoad(full_fileName, RETURN_RES=False, RETURN_HEADER=False):
    """
    Load img file with nibabel, returns data and affine by default
    returns data, affine, and dimension resolution (if RETURN_RES=True)
    """
    import nibabel as nb
    img = nb.load(full_fileName)
    if RETURN_RES and not RETURN_HEADER:
        return img.get_data(), img.affine, img.header.get_zooms()
    elif RETURN_HEADER and not RETURN_RES:
        return img.get_data(), img.affine, img.get_header()
    elif RETURN_RES and RETURN_HEADER:
        return img.get_data(), img.affine, img.header.get_zooms(), img.get_header()
    else:
        return img.get_data(), img.affine


# for backwards compatability with previous scripts
niiLoad = imgLoad


# XXX add mnc saving
# def imgSave(full_fileName, data, aff, data_type='float32', CLOBBER=True):

def niiSave(full_fileName, data, aff, header=None, data_type='float32', CLOBBER=True, VERBOSE=False):
    """
    Convenience function to write nii data to file
    Input:
        - full_fileName:    you can figure that out
        - data:             numpy array
        - aff:              affine matrix
        - header:			header data to write to file (use img.header to get the header of root file)
        - data_type:        numpy data type ('uint32', 'float32' etc)
        - CLOBBER:          overwrite existing file
    """
    import os
    import nibabel as nb

    img = nb.Nifti1Image(data, aff, header=header)
    if data_type is not None:  # if there is a particular data_type chosen, set it
        # data=data.astype(data_type)
        img.set_data_dtype(data_type)
    if not (os.path.isfile(full_fileName)) or CLOBBER:
        img.to_filename(full_fileName)
    else:
        print("This file exists and CLOBBER was set to false, file not saved.")
    if VERBOSE:
        print(full_fileName)


def create_dir(some_directory):
    """
    Create directory recursively if it does not exist
       uses os.makedirs
    """
    import os
    if not os.path.exists(some_directory):
        os.makedirs(some_directory)


def natural_sort(l):
    """
    Returns alphanumerically sorted input
    #natural sort from the interwebs (http://stackoverflow.com/questions/11150239/python-natural-sorting)
    """
    import re

    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_com(img_data):
    """
    Return the center of mass of image data (numpy format)
    """
    import scipy.ndimage.measurements as meas
    return meas.center_of_mass(img_data)


def get_img_bounds(img_data):
    """
    Gets the min and max in the three dimensions of 3d image data and returns
    a 3,2 matrix of values of format dim*{min,max}
    ONLY ignores values == 0
    """
    import numpy as np
    bounds = np.zeros((3, 2))
    # x
    for x in np.arange(img_data.shape[0]):
        if img_data[x, :, :].any():  # if there are any non-zero elements in this slice
            bounds[0, 0] = x
            break
    for x in np.arange(img_data.shape[0])[::-1]:
        if img_data[x, :, :].any():
            bounds[0, 1] = x
            break
    # y
    for y in np.arange(img_data.shape[1]):
        if img_data[:, y, :].any():
            bounds[1, 0] = y
            break
    for y in np.arange(img_data.shape[1])[::-1]:
        if img_data[:, y, :].any():
            bounds[1, 1] = y
            break
    # z
    for z in np.arange(img_data.shape[2]):
        if img_data[:, :, z].any():
            bounds[2, 0] = z
            break
    for z in np.arange(img_data.shape[2])[::-1]:
        if img_data[:, :, z].any():
            bounds[2, 1] = z
            break

    return bounds


def crop_to_roi(img_data, roi_buffer=3, roi_coords=None, data_4d = False):
    """
    Crop image to region of interest based on non-zero voxels, coordinates, or roi_file (not implemented)
    0-based indexing, of course
    :param img_data:
    :param roi_buffer:
    :param roi_coords:
    :param roi_file:
    :return: img_data_crop, roi_coords
    """

    import numpy as np

    if roi_buffer < 0:
        roi_buffer = 0
    roi_buffer = np.tile([-roi_buffer, roi_buffer], (3, 1))  # create for x,y,z

    if roi_coords is None:
        roi_coords = get_img_bounds(img_data) + roi_buffer
    else:
        # TODO check if the coords are out of range (see nilearn.image.crop_img)
        roi_coords = roi_coords + roi_buffer

    r_c = np.copy(roi_coords)
    r_c[:, 1] = r_c[:, 1] + 1  # now r_c has a start and stop for indexing
    img_data_crop = img_data[r_c[0, 0]:r_c[0, 1],
                    r_c[1, 0]:r_c[1, 1],
                    r_c[2, 0]:r_c[2, 1]]

    if data_4d:
        img_data_crop = img_data[r_c[0, 0]:r_c[0, 1],
                        r_c[1, 0]:r_c[1, 1],
                        r_c[2, 0]:r_c[2, 1],:]

    return img_data_crop, roi_coords


def uncrop_from_roi(img_data_crop, uncrop_shape, roi_coords, fill_value=0):
    """
    :param img_data_crop:
    :param uncrop_shape:
    :param roi_coords:
    :param fill_value:
    :return:
    """
    import numpy as np
    uncrop_shape = np.array(uncrop_shape)
    r_c = roi_coords
    if fill_value != 0:
        img_data = np.ones(uncrop_shape).astype(np.dtype(img_data_crop)) * fill_value
    else:
        img_data = np.zeros(uncrop_shape).astype(np.dtype(img_data_crop))
    img_data[r_c[0, 0]:r_c[0, 1] + 1, r_c[1, 0]:r_c[1, 1] + 1, r_c[2, 0]:r_c[2, 1] + 1] = img_data_crop
    return img_data


def erode_mask(img_data, iterations=1, mask=None, structure=None, LIMIT_EROSION=False, min_vox_count=10):
    """
    Binary erosion of 3D image data using scipy.ndimage package
    If LIMIT_EROSION=True, will always return the smallest element mask with count>=min_vox_count
    INPUT:
             - img_data (np image array)
             - iterations = number of iterations for erosion
             - mask = mask img (np array) for restricting erosion
            - structure = as defined by ndimage (will be 3,1 (no diags) if None)
            - LIMIT_EROSION = limits erosion to the step before the mask ended up with no voxels
            - min_vox_count = minimum number of voxels to have in the img_data and still return this version, otherwise returns previous iteration

    Returns mask data in same format as input
    """
    import numpy as np
    import scipy.ndimage as ndimage

    if iterations < 1:
        print("Why are you trying to erode by less than one iteration?")
        print("No erosion performed, returning your data as is.")
        return img_data
    if structure is None:
        structure = ndimage.morphology.generate_binary_structure(3, 1)  # neighbourhood

    # img_data=ndimage.morphology.binary_opening(img_data,iterations=1,structure=structure).astype(img_data.dtype) #binary opening
    if not LIMIT_EROSION:
        img_data = ndimage.morphology.binary_erosion(img_data, iterations=iterations, mask=mask,
                                                     structure=structure).astype(
            img_data.dtype)  # now erode once with the given structure
    else:
        for idx in range(0, iterations):
            img_data_temp = ndimage.morphology.binary_erosion(img_data, iterations=1, mask=mask,
                                                              structure=structure).astype(
                img_data.dtype)  # now erode once with the given structure
            if np.sum(img_data_temp) >= min_vox_count:
                img_data = img_data_temp
            else:
                break
    return img_data


def generate_overlap_mask(mask1, mask2, structure=None):
    """
    Create an overlap mask where a dilated version of mask1 overlaps mask2 (logical AND operation)
    Uses ALL elements >0 for both masks, masks must be in same space
    Dilates and then closes with full connectivity (3,3) by default
    """
    import scipy.ndimage as ndi

    if structure is None:
        structure = ndi.morphology.generate_binary_structure(3, 3)

    overlap_mask = ndi.morphology.binary_dilation(mask1, iterations=1, structure=structure).astype(mask1.dtype) * mask2
    overlap_mask[overlap_mask > 0] = 1
    return ndi.binary_closing(overlap_mask, structure=structure).astype(mask1.dtype)


def select_mask_idxs(mask_img_data, mask_subset_idx):
    """
    Returns a reduced mask_img_data that includes only those indices in mask_subset_idx
    Useful for creating boundary/exclusion masks for cortical regions that are next to the mask of interest
    """
    import numpy as np
    # stupid and probably not fast, but it works
    reduced_mask_data = np.zeros_like(mask_img_data)
    for idx in mask_subset_idx:
        reduced_mask_data[mask_img_data == idx] = idx
    return reduced_mask_data


def affine1_to_affine2(aff1, aff2):
    """
    Create transformation matrix for translating one affine to another, assuming scanner space is the same
    (i.e., images acquired in the same session)
    :param aff1:
    :param aff2:
    :return: aff
    """
    # TODO test
    import numpy as np
    aff1_inv = np.linalg.inv(aff1)
    return np.matmul(aff1_inv, aff2)

def map_values_to_label_file(values_label_lut_csv_fname, label_img_fname,
                             out_mapped_label_fname=None,
                             value_colName="Value",
                             label_idx_colName="Index",
                             SKIP_ZERO_IDX=True,
                             MATCH_VALUE_TO_LABEL_VIA_MATRIX=False):
    """
    Map from values/index dataframe to labels in label_fname (for visualising results in label space)

    #TODO: for some reason this doesn't always work -- you will need to look into it to make sure that it works when the .nii file has MORE indices than you expect given the matrix

    :param values_label_lut_csv_fname: csv file mapping values to index in label_img_fname
    :param label_img_fname: label file (nii or other)
    :param out_mapped_label_fname: ouptut file name (nii/nii.gz only)
    :param value_colName: name of column with values (default: Value)
    :param label_idx_colName:name of column with index numbers (default: Index)
    :param SKIP_ZERO_IDX: skips 0 (usually background) {True, False}
    :param MATCH_VALUE_TO_LABEL_VIA_MATRIX: if true, values_label_lut_csv_fname is a matrix with first column = labels, 2nd = values
    :return: out_mapped_label_fname
    """
    import numpy as np
    import pandas as pd
    import os
    
    if out_mapped_label_fname is None:
        out_mapped_label_fname = os.path.splitext(os.path.splitext(label_img_fname)[0])[0] + "_value_mapped.nii.gz" #takes care of two . extensions if necessary
    
    if not MATCH_VALUE_TO_LABEL_VIA_MATRIX: #we expect a csv file
        df=pd.read_csv(values_label_lut_csv_fname)
        values=df[value_colName].values
        indices=df[label_idx_colName].values
    else: #otherwise just a matrix of values
        indices = values_label_lut_csv_fname[:,0]
        values = values_label_lut_csv_fname[:,1]
        
    if SKIP_ZERO_IDX and 0 in indices:
        indices.remove(0)

    d,a,h = imgLoad(label_img_fname,RETURN_HEADER=True)
    d_out = np.zeros_like(d).astype(np.float32)

    for idx,index in enumerate(indices):
        print index, values[idx]
        d_out[d==index] = values[idx]

    niiSave(out_mapped_label_fname,d_out,a,header=h)
    return out_mapped_label_fname

def map_values_to_coordinates(values, coordinates, reference_fname, out_mapped_fname=None, return_mapped_data=True):
    """
    Maps values to coordinate locations. Coordinate space provided by reference_fname. Values in a single vector, coordinates in list/matrix of coord locations
    return: mapped data array (when return_mapped_data=True)
    """
    import numpy as np
    d,a,h = imgLoad(reference_fname,RETURN_HEADER=True)
    d_out = np.zeros_like(d).astype(np.float32)
    for idx, coord in enumerate(coordinates):
        d_out[tuple(coord)] = values(idx)
    if out_mapped_fname is None:
        niiSave(out_mapped_fname,d_out,a,header=h)
    if return_mapped_data or out_mapped_fname is None:
        return d_out
    
def extract_stats_from_masked_image(img_fname, mask_fname, thresh_mask_fname=None, combined_mask_output_fname=None,
                                    ROI_mask_fname=None, thresh_val=None,
                                    thresh_type=None, result='all', label_subset=None, SKIP_ZERO_LABEL=True,
                                    nonzero_stats=True,
                                    erode_vox=None, min_val=None, max_val=None, VERBOSE=False, USE_LABEL_RES=False,
                                    volume_idx=0):
    #TODO - THIS SHOULD BE CHECKED TO MAKE SURE THAT IT WORKS WITH ALL INPUTS - ASSUMPTIONS ABOUT TRANSFORMS WERE MADE XXX
    #TODO - works for NII and MNC, but NOT tested for combining the two of them XXX
    #TODO - Add an additional flag to remove 0s that are present in the metric file from analysis

    """
    Extract values from img at mask location
    Images do not need to be the same resolution, though this is highly preferred
        - resampling taken care of with nilearn tools
        - set nonzero_stats to false to include 0s in the calculations
        - clipped to >max_val
        - volume output based on whichever resolution you chose with USE_LABEL_RES
       Input:
         - img_fname:                   3D or 4D image (if 4D, set volume_idx to select volume)
         - mask_fname:                  3D mask in same space, single or multiple labels (though not necessarily same res)
         - thresh_mask_fname:           3D mask for thresholding, can be binary or not
         - combined_mask_output_fname:  output final binary mask to this file and a _metric file - will split on periods (used for confirmation of region overlap)
         - ROI_mask_fname               3D binary mask for selecting only this region for extraction (where mask=1)
         - thresh_val:                  upper value for thresholding thresh_mask_fname, values above/below this are set to 0
         - thresh_type:                 {'upper' = > thresh_val = 0,'lower' < thresh_val = 0}
         - result:                      specification of what output you require {'all','data','mean','median','std','min','max'}
         - label_subset:                list of label values to report stats on
         - SKIP_ZERO_LABEL:             skip where label_val==0 {True,False} (usually the background label)  - XXX probably does not work properly when False :-/
         - nonzero_stats:               calculate without 0s in img_fname, or with {True,False}
         - erode_vox                    number of voxels to erode mask by (simple dilation-erosion, then erosion, None for no erosion)
         - min_val:                     set min val for clipping of metric (eg., for FA maps, set to 0)
         - max_val:                     set max val for clipping of metric (eg., for FA maps, set to 1.0)
         - VERBOSE                      verbose reporting or not (default: False)
         - USE_LABEL_RES                otherwise uses the res of the img_fname (default: False)
         - volume_idx                   select volume of 4D img_fname that is selected (default=0, skipped if 3D file)

       Output: (in data structure composed of numpy array(s))
         - data, volume, mean, median, std, minn, maxx
         - or all in data structure if result='all'
         - note: len(data)= num vox that the values were extracted from (i.e., [len(a_idx) for a_idx in res.data])

       e.g.,
         - res=extract_stats_from_masked_image(img_fname,mask_fname)
         :rtype: object
    """
    import os
    import numpy as np
    from nilearn.image import resample_img

    class return_results(object):
        # output results as an object with these values
        def __init__(self, label_val, data, vox_coord, volume, mean, median, std, minn, maxx, settings):
            self.label_val = np.array(label_val)
            self.data = np.array(data)
            self.vox_coord = np.array(vox_coord)
            self.volume = np.array(volume)
            self.mean = np.array(mean)
            self.median = np.array(median)
            self.std = np.array(std)
            self.minn = np.array(minn)
            self.maxx = np.array(maxx)
            self.settings = settings

        def __str__(self):
            # defines what is returned when print is called on this class
            template_txt = """
            label_val: {label_val}
            len(data): {data_len}
            vox_coord: voxel coordinates of data
            volume   : {volume}
            mean     : {mean}
            median   : {median}
            std      : {std}
            maxx     : {maxx}
            minn     : {minn}
            settings : file and parameter settings (dictionary)
            """
            return template_txt.format(label_val=self.label_val, data_len=len(self.data), volume=self.volume,
                                       mean=self.mean, median=self.median, std=self.std, maxx=self.maxx, minn=self.minn)

    d_label_val = []
    d_data = []
    d_vox_coord = []
    d_volume = []
    d_mean = []
    d_median = []
    d_std = []
    d_min = []
    d_max = []
    d_settings = {'metric_fname': img_fname,
                  'label_fname': mask_fname,
                  'thresh_mask_fname': thresh_mask_fname,
                  'combined_mask_output_fname': combined_mask_output_fname,
                  'ROI_mask_fname': ROI_mask_fname,
                  'thresh_val': thresh_val,
                  'thresh_type': thresh_type,
                  'SKIP_ZERO_LABEL': SKIP_ZERO_LABEL,
                  'nonzero_stats': nonzero_stats,
                  'erode_vox': erode_vox,
                  'min_val': min_val,
                  'max_val': max_val,
                  'USE_LABEL_RES': USE_LABEL_RES}

    d, daff, dr, dh = imgLoad(img_fname, RETURN_RES=True, RETURN_HEADER=True)
    if len(np.shape(d))>3:
        #we sent 4d data!
        if VERBOSE:
            print("You are trying to extract metrics from a single volume of a 4d file, it should work (but takes longer, sorry)... ")
        print(" Extracting from volume index: " + str(volume_idx))
        print("    - data shape: " + str(np.shape(d)))
        d = d[:,:,:,volume_idx] #select the volume that was requested
        
    mask, maff, mr, mh = imgLoad(mask_fname, RETURN_RES=True, RETURN_HEADER=True)

    if os.path.splitext(mask_fname)[
        -1] == ".mnc":  # test if the extension is mnc, and make sure we have integers in this case...
        if VERBOSE:
            print(" Looks like you are using mnc files.")
            print(
                " Make sure that ALL of your input data is in the same space and mnc format (i.e., don't mix mnc and nii.gz)")
            print(
                " I will also force all your label values to be integers as a hack to fix non-integer values stored in the file. np.rint(labels).astype(int)")
        mask = np.rint(mask).astype(int)  # round with rint and the convert to int

    # dumb way to do this,but too much coffee today
    if USE_LABEL_RES:
        chosen_aff = maff
        chosen_header = mh
        chosen_shape = np.shape(mask)
        vox_vol = vox_vol = np.prod(mr)  # and mask
        if VERBOSE:
            print(" Any calculation of volume will be based on label_file resolution: "),
            print(mr),
        # see if we need to resample the img to the mask
        if not np.array_equal(np.diagonal(maff), np.diagonal(daff)):
            d = resample_img(img_fname, maff, np.shape(mask), interpolation='nearest').get_data()
            if len(np.shape(d))>3:
                d = d[:,:,:,volume_idx]
    else:  # default way, use img_fname resolution
        chosen_aff = daff
        chosen_header = dh
        chosen_shape = np.shape(d)
        vox_vol = vox_vol = np.prod(dr)  # volume of single voxel for data
        if VERBOSE:
            print(" Any calculation of volume will be based on metric_file resolution: "),
            print(dr)
        # see if we need to resample the mask to the img
        if not np.array_equal(np.diagonal(maff), np.diagonal(daff)):
            mask = resample_img(mask_fname, daff, np.shape(d), interpolation='nearest').get_data()

        else:  # they are the same and we already loaded the data
            pass

    # if we have passed an additional thresholding mask, move to the same space,
    # thresh at the given thresh_val, and remove from our mask
    if thresh_mask_fname is not None:
        thresh_mask, thresh_maff = imgLoad(thresh_mask_fname)
        if not np.array_equal(np.diagonal(thresh_maff), np.diagonal(chosen_aff)):
            thresh_mask = resample_img(thresh_mask_fname, chosen_aff, chosen_shape, interpolation='nearest').get_data()
        else:
            pass  # we already have the correct data

        if thresh_type is 'upper':
            mask[thresh_mask > thresh_val] = 0  # remove from the mask
        elif thresh_type is 'lower':
            mask[thresh_mask < thresh_val] = 0  # remove from the mask
        else:
            print("set a valid thresh_type: 'upper' or 'lower'")
            return

    if ROI_mask_fname is not None:
        ROI_mask, ROI_maff = imgLoad(ROI_mask_fname)
        if not np.array_equal(np.diagonal(ROI_maff), np.diagonal(chosen_aff)):
            ROI_mask = resample_img(ROI_mask_fname, chosen_aff, chosen_shape, interpolation='nearest').get_data()
        else:  # we already have the correct data
            pass

        mask[ROI_mask < 1] = 0  # remove from the final mask

    if label_subset is None:
        mask_ids = np.unique(mask)
        # print(mask)
        if SKIP_ZERO_LABEL:
            mask_ids = mask_ids[mask_ids != 0]
    else:  # if we selected some label subsets then we should use them here
        mask_ids = label_subset

    if len(mask_ids) == 1:  # if we only have one, we need to make it iterable
        mask_ids = [mask_ids]
    if erode_vox is not None:  # we can also erode each individual mask to get rid of some partial voluming issues (does no erosion if mask vox count falls to 0)
        single_mask = np.zeros_like(mask)
        for mask_id in mask_ids:
            single_mask[mask == mask_id] = 1
            temp_mask = np.copy(single_mask)
            single_mask = erode_mask(single_mask, erode_vox)
            temp_mask[
                np.logical_and(mask == mask_id, single_mask == 0)] = 0  # to check how many vox's we will have left over
            if np.sum(
                    temp_mask) > 0:  # if we know that there is still at least one mask voxel leftover... we use the erosion
                mask[np.logical_and(mask == mask_id, single_mask == 0)] = 0
            else:
                print("Label id: " + str(
                    mask_id) + ': Not enough voxels to erode!')  # This intelligence has also been added to erode_mask, but leaving it explicit here
            single_mask = single_mask * 0  # clear the single mask
        del single_mask

    if combined_mask_output_fname is not None:
        if VERBOSE:
            print(" Debug files:")
            print("  " + combined_mask_output_fname)
            print("  " + combined_mask_output_fname.split('.')[0] + "_metric.nii.gz")
        mask_t = np.zeros_like(mask)
        for mask_id in mask_ids:
            mask_t[mask==mask_id] = mask_id
        niiSave(combined_mask_output_fname, mask_t, chosen_aff, data_type='uint16', header=chosen_header)
        niiSave(combined_mask_output_fname.split('.')[0] + "_metric.nii.gz", d, chosen_aff, header=chosen_header)
        del mask_t

    if VERBOSE:
        print("Mask index extraction: "),

    for mask_id in mask_ids:
        if VERBOSE:
            print(mask_id),
        dx = np.ma.masked_array(d, np.ma.make_mask(np.logical_not(mask == mask_id))).compressed()
        if nonzero_stats:
            dx = dx[dx > 0]
            mask[d == 0] = 0 #this is necessary because we need the full 3d information to calculate the voxel coordinates
        if not max_val is None:
            dx[dx > max_val] = max_val
        if not min_val is None:
            dx[dx < min_val] = min_val
        if len(dx) == 0:  # NO DATA WAS RECOVERED FROM THIS MASK, report as zeros?
            dx = np.array([0])
            d_volume.append(0)  # volume is a special case, need to set explicitly
        else:
            d_volume.append(len(dx) * vox_vol)
        # keep track of these as we loop, convert to structure later on
        d_label_val.append(mask_id)
        d_data.append(dx)
        #print(np.where(dx==mask_id))
        d_vox_coord.append(np.column_stack(np.where(mask==mask_id))) #x,y,z coordinates of this voxel, not sure if works
        d_mean.append(np.mean(dx))  # XXX could put a check here to set the values to NaN or None if there is no data
        d_median.append(np.median(dx))
        d_std.append(np.std(dx))
        d_min.append(np.min(dx))
        d_max.append(np.max(dx))
    if VERBOSE:
        print("")
    results = return_results(d_label_val, d_data, d_vox_coord, d_volume, d_mean, d_median, d_std, d_min, d_max, d_settings)

    if result == 'all':
        return results
    elif result == 'data':
        return results.data
    elif result == 'volume':
        return results.volume
    elif result == 'mean':
        return results.mean
    elif result == 'median':
        return results.median
    elif result == 'std':
        return results.std
    elif result == 'min':
        return results.minn
    elif result == 'max':
        return results.maxx

def extract_label_volume(label_files,IDs=None, label_df=None,
                         label_subset_idx=None, label_tag="label_",
                         thresh_mask_files=None, ROI_mask_files=None,
                         thresh_val=None, max_val=None,thresh_type=None,
                         zfill_num=3, VERBOSE=False, volume_idx=0):
    """
    wrapper for extract_quantitative metric to calculate volume from label files,
    assumes: ALL_FILES_ORDERED= True
             USE_LABEL_RES    = True

    :param label_files:
    :param IDs:
    :param label_df:
    :param label_subset_idx:
    :param label_tag:
    :param thresh_mask_files:
    :param ROI_mask_files:
    :param thresh_val:
    :param max_val:
    :param thresh_type:
    :param zfill_num:
    :param VERBOSE:
    :param volume_idx:
    :return:
    """
    df = extract_quantitative_metric(label_files, label_files, 
                                     IDs=IDs, 
                                     label_df=label_df, 
                                     label_subset_idx=label_subset_idx,
                                     label_tag=label_tag, metric='volume',
                                     thresh_mask_files=thresh_mask_files, 
                                     ROI_mask_files=ROI_mask_files, 
                                     thresh_val=thresh_val, 
                                     max_val=max_val,
                                     thresh_type=thresh_type, 
                                     erode_vox=None, zfill_num=3,
                                     DEBUG_DIR=None, VERBOSE=False,
                                     USE_LABEL_RES=True, ALL_FILES_ORDERED=True,
                                     volume_idx=0)
    return df

def extract_quantitative_metric(metric_files, label_files, IDs=None, label_df=None, label_subset_idx=None,
                                label_tag="label_", metric='all',
                                thresh_mask_files=None, ROI_mask_files=None, thresh_val=None, max_val=None,
                                thresh_type=None, erode_vox=None, zfill_num=3,
                                DEBUG_DIR=None, VERBOSE=False,
                                USE_LABEL_RES=False, ALL_FILES_ORDERED=False,
                                n_jobs=1,volume_idx=0):

    """
    Extracts voxel-wise data for given set of matched label_files and metric files. Returns pandas dataframe of results
    CAREFUL: IDs are currently defined as the last directory of the input metric_files element
    INPUT:
        - metric_files      - list of files for the metric that you are extracting
        - label_files       - list of label files matched to each file in metric_files (currently restricted to ID at the beginning of file name ==> ID_*)
        - IDs               - list of IDs for matching files - no easy way to get around this :-/
        - label_df          - pandas dataframe of label index (index) and description (label_id)
        - label_subset_idx  - list of label indices that you want to extract data from [10, 200, 30]
        - label_tag         - string that will precede the label description in the column header
        - metric            - metric to extract {'all','mean','median','std','volume','vox_count','data'}
                            - if you select 'data', an additional list of lists will be returned
                            - such that list[0]=voxel values list[1]=voxel coordinates
        - thresh_mask_files - list of files for additional thresholding (again, same restrictions as label_files)
        - ROI_mask_files    - binary mask file(s) denoting ROI for extraction =1
        - thresh_val        - value for thresholding
        - max_val           - maximum value for clipping the metric (i.e., if FA, set to 1, 3 for MK)
        - thresh_type       - {'upper' = > thresh_val = 0,'lower' < thresh_val = 0}
        - erode_vox         - number of voxels to erode mask by (simple binary erosion, None for no erosion)
        - zfill_num         - number of zeros to fill to make label index numbers line up properly
        - DEBUG_DIR         - directory to dump new thresholded and interpolated label files to
        - VERBOSE           - verbose reporting or not (default: False)
        - USE_LABEL_RES     - otherwise uses the res of the img_fname (default: False)
        - ALL_FILES_ORDERED - set to True if you know that all of your input lists of files are matched correctly
        - volume_idx        - select volume of 4D img_fname that is selected (default=0, skipped if 3D file)

    OUTPUT:
        - df_4d             - pandas dataframe of results
    """

    import os
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed

    USE_SINGLE_LABEL_FILE=False

    if n_jobs<1:
        n_jobs=1

    if metric is 'data': #only used if we have requested "data", in which case we get the volumes in the df and the raw data in a list of results objects from extract_stats_from_masked_image
        all_res_data = []
        
    if ALL_FILES_ORDERED:
        print("You have set ALL_FILES_ORDERED=True, I will not check your input lists for ordering.")

    cols = ['ID', 'metric_file', 'label_file', 'thresh_file', 'thresh_val', 'thresh_type',
            'ROI_mask']  # used to link it to the other measures and to confirm that the masks were used in the correct order so that the values are correct

    # if we only pass a single subject, make it a list so that we can loop over it without crashing
    if isinstance(metric_files, basestring):
        metric_files = [metric_files]
    if isinstance(label_files, basestring):
        label_files = [label_files]
    if isinstance(thresh_mask_files, basestring):
        thresh_mask_files = [thresh_mask_files]
    if len(label_files) == 1:
        USE_SINGLE_LABEL_FILE = True #if there is only one, then we assume that all files are registered and we just need the single label file
    if isinstance(IDs, basestring):
        IDs = [IDs]
    if label_subset_idx is None:  # you didn't define your label indices, so we go get them for you from the 1st label file
        print("label_subset_idx was not defined")
        print("Label numbers were extracted from the first label file")
        print("label_id = 0 was removed")

        label_subset_idx = np.unique(imgLoad(label_files[0])[0]).astype(int)
        if os.path.splitext(label_files[0])[-1] == ".mnc":
            print("Looks like you are using mnc files.")
            print(
                "Make sure that ALL of your input data is in the same space and mnc format (i.e., don't mix mnc and nii.gz)")
            print("I will be converting ALL of your labels to rounded integers to be safe. np.rint(labels).astype(int)")
            label_subset_idx = np.rint(label_subset_idx).astype(int)

        label_subset_idx = label_subset_idx[label_subset_idx != 0]
    elif isinstance(label_subset_idx,int):
        label_subset_idx = [label_subset_idx] #change to a list if it was only a single integer
    if metric is not 'all':
        if metric is not 'data':
            metric_txt = metric
        else:
            metric_txt = 'volume'
        if label_df is None:  # WHAT? you didn't provide a label to idx matching dataframe??
            print("label_df dataframe (label index to name mapping) was not defined")
            print("Generic label names will be calculated from the unique values in the first label file")
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + metric_txt
                cols.append(col_name)
            df_4d = pd.DataFrame(columns=cols)
        else:
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + metric_txt
                cols.append(col_name)
            df_4d = pd.DataFrame(columns=cols)
    else: #we want all the metrics, so we need to create the columns for all of them
        if label_df is None:  # WHAT? you didn't provide a label to idx matching dataframe??
            print("label_df dataframe (label index to name mapping) was not defined")
            print("Generic label names will be calculated from the unique values in the first label file")
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + "mean"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + "median"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + "std"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + "volume"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + "vox_count"
                cols.append(col_name)
            df_4d = pd.DataFrame(columns=cols)
        else:
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + "mean"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + "median"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + "std"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + "volume"
                cols.append(col_name)
            for idx, label_id in enumerate(label_subset_idx):
                col_name = label_tag + str(label_id).zfill(zfill_num) + "_" + label_df.loc[label_id].Label + "_" + "vox_count"
                cols.append(col_name)
            df_4d = pd.DataFrame(columns=cols)

    if DEBUG_DIR is not None:
        create_dir(DEBUG_DIR)  # this is where the combined_mask_output is 
        #going to go so that we can check to see what we actually did to our masks

    if IDs is None and not(ALL_FILES_ORDERED): #if this was set to True, then we just grab the correct index
        IDs = [os.path.basename(os.path.dirname(metric_file)) for metric_file in metric_files]  # if ID was not set,
        # we assume that we can generate it here as the last directory of the path to the metric_file
        print(
            "No IDs were specified, attempting to reconstruct them as the last subdirectory of the input metric files")
        print(" e.g., " + os.path.basename(os.path.dirname(metric_files[0])))
    elif IDs is None and ALL_FILES_ORDERED: # the user knows what they are doing, we will not use IDs to lookup the correct corresponding files
        IDs = [os.path.basename(metric_file) for metric_file in metric_files]

    for idx, ID in enumerate(IDs):
        DATA_EXISTS = True
        # grab the correct label and metric files to go with the ID

        if VERBOSE:
            print(ID)
        else:
            print(ID),
        if not(ALL_FILES_ORDERED):
            metric_file = [s for s in metric_files if ID in s]  # make sure our label file is in the list that was passed
            label_file = [s for s in label_files if ID in s]  # make sure our label file is in the list that was passed
            if len(metric_file) > 1:
                print("")
                print "OH SHIT, too many metric files. This should not happen!"
            elif len(metric_file) == 0:
                print("")
                print "OH SHIT, no matching metric file for: " + ID
                print("This subject has not been processed")
                DATA_EXISTS = False

            if len(label_file) > 1:
                print("")
                print "OH SHIT, too many label files. This should not happen!"
            elif len(label_file) == 0:
                print("")
                print "OH SHIT, no matching label file for: " + ID
                print("This subject has not been processed")
                DATA_EXISTS = False
        else: #files should already be ordered
            metric_file = metric_files[idx]
            if not(USE_SINGLE_LABEL_FILE):
                label_file = label_files[idx]
            else:
                label_file = label_files[0]
        if thresh_mask_files is not None:
            if len(thresh_mask_files) == 1:  # if we only provide one mask, we use this for everyone
                thresh_mask_fname = thresh_mask_files[0]
            elif len(thresh_mask_files) > 1 and not(ALL_FILES_ORDERED):
                thresh_mask_fname = [s for s in thresh_mask_files if ID in s]  # make sure our label file
                                                                                # is in the list that was passed
                if len(thresh_mask_fname) > 1:
                    print("")
                    print "OH SHIT, too many threshold mask files. This should not happen!"

                elif len(thresh_mask_fname) == 0:
                    print("")
                    print "OH SHIT, no matching threshold mask file for: " + ID
                    DATA_EXISTS = False
            else:
                thresh_mask_fname = thresh_mask_files[idx]
        else:
            thresh_mask_fname = None

        if ROI_mask_files is not None:
            if len(ROI_mask_files) == 1:  # if we only provide one mask, we use this for everyone
                ROI_mask_fname = ROI_mask_files
            elif len(ROI_mask_files) > 1 and not(ALL_FILES_ORDERED):
                ROI_mask_fname = [s for s in ROI_mask_files if
                                  ID in s]  # make sure our label file is in the list that was passed
                if len(ROI_mask_fname) > 1:
                    print "OH SHIT, too many threshold mask files. This should not happen!"

                elif len(ROI_mask_fname) == 0:
                    print "OH SHIT, no matching ROI mask file for: " + ID
                    DATA_EXISTS = False
            else:
                ROI_mask_fname = ROI_mask_files[idx]
        else:
            ROI_mask_fname = None

        #
        ## STOP THESE CHECKS COULD BE REMOVED

        if DATA_EXISTS:
            try:
                if DEBUG_DIR is not None:
                    combined_mask_output_fname = os.path.join(DEBUG_DIR, ID + "_corrected_labels.nii.gz")
                else:
                    combined_mask_output_fname = None
                if not(ALL_FILES_ORDERED):
                    metric_file = metric_file[0]  # break them out of the list they were stored as
                    label_file = label_file[0]
                
                    if thresh_mask_fname is not None:
                        thresh_mask_fname = thresh_mask_fname[0]
                    if ROI_mask_fname is not None:
                        ROI_mask_fname = ROI_mask_fname[0]

                if VERBOSE:
                    print(" metric    : " + metric_file)
                    print(" label     : " + label_file)
                    print(" thresh    : " + str(thresh_mask_fname))
                    print(" thresh_val: " + str(thresh_val))
                    print(""),
                res = extract_stats_from_masked_image(metric_file, label_file, thresh_mask_fname=thresh_mask_fname,
                                                      combined_mask_output_fname=combined_mask_output_fname,
                                                      ROI_mask_fname=ROI_mask_fname, thresh_val=thresh_val,
                                                      thresh_type=thresh_type,
                                                      label_subset=label_subset_idx, erode_vox=erode_vox, result='all',
                                                      max_val=max_val, VERBOSE=VERBOSE, USE_LABEL_RES=USE_LABEL_RES,
                                                      volume_idx=volume_idx)

                #remove any None values, so that pandas treats it properly when writing to csv
                if thresh_mask_fname is None:
                    thresh_mask_fname = "None"
                if ROI_mask_fname is None:
                    ROI_mask_fname = "None"

                # now put the data into the rows:
                df_4d.loc[idx, 'ID'] = str(ID)  # XXX there should be a more comprehensive solution to this
                df_4d.loc[idx, 'metric_file'] = metric_file
                df_4d.loc[idx, 'label_file'] = label_file
                df_4d.loc[idx, 'thresh_file'] = thresh_mask_fname
                df_4d.loc[idx, 'thresh_val'] = thresh_val  # this is overkill, since it should always be the same
                df_4d.loc[idx, 'thresh_type'] = thresh_type  # this is overkill, since it should always be the same
                df_4d.loc[idx, 'ROI_mask'] = ROI_mask_fname
                if (metric is 'all'):
                    df_4d.loc[idx, 7:7+1*len(label_subset_idx)] = res.mean
                    df_4d.loc[idx, 7+1*len(label_subset_idx):7+2*len(label_subset_idx)] = res.median
                    df_4d.loc[idx, 7+2*len(label_subset_idx):7+3*len(label_subset_idx)] = res.std
                    df_4d.loc[idx, 7+3*len(label_subset_idx):7+4*len(label_subset_idx)] = res.volume
                    df_4d.loc[idx, 7+4*len(label_subset_idx):7+5*len(label_subset_idx)] = [len(a_idx) for a_idx in res.data]  # gives num vox
#                elif metric is 'data':
#                    data_string_list=[None]*len(res.data)
#                    for string_list_idx,res_data_single_sub in enumerate(res.data):
#                        data_string=""
#                        for val in res_data_single_sub:
#                            data_string=data_string+" "+"{0:.4f}".format(val)
#                        data_string_list[string_list_idx]=data_string
#                    df_4d.loc[idx,7::] = data_string_list
                elif metric is 'data':
                    df_4d.loc[idx, 7::] = res.volume
                    all_res_data.append(res)
                elif metric is 'mean':
                    df_4d.loc[idx, 7::] = res.mean
                elif metric is 'median':
                    df_4d.loc[idx, 7::] = res.median
                elif metric is 'std':
                    df_4d.loc[idx, 7::] = res.std
                elif metric is 'volume':
                    df_4d.loc[idx, 7::] = res.volume
                elif metric is 'vox_count':
                    df_4d.loc[idx, 7::] = [len(a_idx) for a_idx in res.data]  # gives num vox
                else:
                    print("Incorrect metric selected.")
                    return
            except:
                print("")
                print("##=====================================================================##")
                print("Darn! There is something wrong with: " + ID)
                print("##=====================================================================##")
    print ""
    if metric is not 'data':
        return df_4d
    else:
        return df_4d, all_res_data

def calc_3D_flux(data, structure=None, distance_method='edt'):
    """
    Calculate the flux of 3d image data, returns flux and distance transform
    - flux calculated as average normal flux per voxel on a sphere
    - algorithm inspired by Bouix, Siddiqi, Tannenbaum (2005)
    Input:
        - data              - numpy data matrix (binary, 1=foreground)
        - structure         - connectivity structure (generate with ndimage.morphology.generate_binary_structure, default=(3,3))
        - distance_method   - method for distance computation {'edt','fmm'}
    Output:
        - norm_struc_flux   - normalised flux for each voxel
        - data_dist         - distance map
    """
    from scipy import ndimage
    import numpy as np

    # distance metric
    if distance_method is 'edt':
        data_dist = ndimage.distance_transform_edt(data).astype('float32')
    elif distance_method is 'fmm':
        import skfmm  # scikit-fmm
        data_dist = skfmm.distance(data).astype('float32')

    data_grad = np.array(np.gradient(data_dist)).astype('float32')
    data_flux = data_dist * data_grad

    norm_flux = np.sqrt(data_flux[0] ** 2 + data_flux[1] ** 2 + data_flux[
        2] ** 2)  # calculate the flux (at normal) at each voxel, by its definition in cartesian space

    # flux for each given voxel is represented by looking to its neighbours
    if structure is None:
        structure = ndimage.morphology.generate_binary_structure(3, 3)
        structure[1, 1, 1] = 0

    norm_struc_flux = np.zeros_like(norm_flux)
    norm_struc_flux = ndimage.convolve(norm_flux,
                                       structure)  # this is the mean flux in the neighbourhood at the normal for each voxel

    return norm_struc_flux, data_dist


def skeletonise_volume(vol_fname, threshold_type='percentage', threshold_val=0.2, method='edt', CLEANUP=True):
    """
    Take an ROI, threshold it, and create 2d tract skeleton
    requires: fsl {tbss_skeleton,fslmaths}
    output:
        - _skel.nii.gz skeleton file to same directory as input
        - optional _smth intermediate file
    return:
        - full name of skeletonised file

    """

    import nibabel as nb
    import os
    import numpy as np
    import scipy.ndimage as ndimage
    import subprocess

    smth_tail = '_smth.nii.gz'
    skel_tail = '_skel.nii.gz'
    data_dist_smth_fname = os.path.join(os.path.dirname(vol_fname),
                                        os.path.basename(vol_fname).split(".")[0] + smth_tail)
    data_dist_smth_skel_fname = os.path.join(os.path.dirname(vol_fname),
                                             os.path.basename(vol_fname).split(".")[0] + skel_tail)

    img = nb.load(vol_fname)
    data = img.get_data()
    aff = img.affine

    # thresh
    if threshold_type is 'percentage':
        thresh = np.max(data) * threshold_val
        data[data < thresh] = 0
        # binarise
        data[data >= thresh] = 1
    elif threshold_type is 'value':
        thresh = threshold_val
        data[data < thresh] = 0
        # binarise
        data[data >= thresh] = 1

    # inversion is not necessary, this distance metric provides +ve vals inside the region id'd with 1s
    # data=1-data #(or data^1)

    # distance metric
    if method is 'edt':
        data_dist = ndimage.distance_transform_edt(data).astype('float32')
    elif method is 'fmm':
        import skfmm  # scikit-fmm
        data_dist = skfmm.distance(data).astype('float32')
    # smooth
    # filter may need to change depending on input resolution
    data_dist_smth = ndimage.filters.gaussian_filter(data_dist, sigma=1)
    niiSave(data_dist_smth_fname, data_dist_smth, aff)

    # skeletonise
    # tbss_skeleton seems to be the most straightforward way to do this...
    # XXX no 3d skeletonisation in python?
    cmd_input = ['tbss_skeleton', '-i', data_dist_smth_fname, '-o', data_dist_smth_skel_fname]
    subprocess.call(cmd_input)
    # now binarise in place
    cmd_input = ['fslmaths', data_dist_smth_skel_fname, '-thr', str(0), '-bin', data_dist_smth_skel_fname, '-odt',
                 'char']
    subprocess.call(cmd_input)

    if CLEANUP:
        cmd_input = ['rm', '-f', data_dist_smth_fname]
        subprocess.call(cmd_input)

    return data_dist_smth_skel_fname

def get_distance_shell(data, direction = 'outer', distance_method='edt',start_distance=0, stop_distance=1, return_as_distance=False, reset_zero_distance = False):
    """
    Calculates a distance metric on the provided binary data, limits it within start_distance and stop_distance to produce a shell.
    Calculated in voxel units.
    If stop_distance - start_distance < 1, then there will likely be holes in the shell

    :param data:                numpy.array of binary data {0,1}
    :param direction:           direction for distance function 'outer' increases from region boundary to limits of volume, 'inner' from region boundary to center
    :param distance_method:     desired distance method {'edt',fmm'}
    :param start_distance:      defines the start position of the shell, in distance units
    :param stop_distance:       defines the stop position of the shell, in distance units (None does max distance)
    :param return_as_distance:  do not binarise the distance map before returning
    :param reset_zero_distance: subtract the minimum distance from the distance map, does nothing when return_as_distance=False (note, sets all boundary voxels at start_distance to 0!)
    :return: data_dist          binary shell defined as 1s within the start and stop distances (np.array)
    """
    from scipy import ndimage
    import numpy as np

    if np.sum(np.unique(data)[:]) > 1:
        print('Please use a binary image')
        return

    #if we wanted the outer distance, need to flip our ones and zeros
    if direction is "outer":
        data=np.logical_not(data).astype(int)
    elif direction is "inner":
        pass #don't need to do anything
    else:
        print("Please select a valid direction for the distance function: {/'inner/', /'outer/'}")
        print("Exiting")
        return

    # distance metric
    if distance_method is 'edt':
        data_dist = ndimage.distance_transform_edt(data).astype('float32')
    elif distance_method is 'fmm':
        import skfmm  # scikit-fmm
        data_dist = skfmm.distance(data).astype('float32')
    else:
        print('You have not selected a valid distance metric.')
        print('Exiting.')
        return

    print("Distance range = %.2f - %.2f") %(np.min(np.unique(data_dist)),np.max(np.unique(data_dist)))
    if stop_distance > np.max(np.unique(data_dist)):
        print('You have set your stop_distance greater than the possible distance')
    if start_distance > np.max(np.unique(data_dist)):
        print("You have set your start_distance greater than the maximum distance, where distance range = %.2f - %.2f") %(np.min(np.unique(data_dist)),np.max(np.unique(data_dist)))
        print("This results in a volume filled with 0s. Have fun with that.")

    data_dist[data_dist<start_distance] = 0
    if stop_distance is not None:
        data_dist[data_dist>stop_distance] = 0
    if not return_as_distance:
        data_dist[data_dist!=0] = 1
    if return_as_distance and reset_zero_distance:
        data_dist[data_dist != 0] -= np.min(data_dist[np.nonzero(data_dist)])

    return data_dist

def submit_via_qsub(template_text=None, code="# NO CODE HAS BEEN ENTERED #", \
                    name='CJS_job', nthreads=8, mem=1.75, outdir='/scratch', \
                    description="Lobule-specific tractography", SUBMIT=True):
    """
    Christopher J Steele
    Convenience function for job submission through qsub
    Creates and then submits (if SUBMIT=True) .sub files to local SGE
    Input:
        - template_text:    correctly formatted qsub template for .format replacement. None=default (str)
        - code:             code that will be executed by the SGE (str)
        - name:             job name
        - nthreads:         number of threads to request
        - mem:              RAM per thread
        - outdir:           output (and working) directory for .o and .e files
        - description:      description that will be included in header of .sub file
        - SUBMIT:           actually submit the .sub files

        default template_text:
        template_text=\\\"""#!/bin/bash
        ## ====================================================================== ##
        ## 2015_09 Chris Steele
        ## {DESCRIPTION}
        ## ====================================================================== ##
        ##
        #$ -N {NAME}	    #set job name
        #$ -pe smp {NTHREADS}	#set number of threads to use
        #$ -l h_vmem={MEM}G	    #this is a per-thread amount of virtual memory, I think...
        #$ -l h_stack=8M 	    #required to allow multiple threads to work correctly
        #$ -V 			        #inherit user env from submitting shell
        #$ -wd {OUTDIR} 	    #set working directory so that .o files end up here (maybe superseded?)
        #$ -o {OUTDIR} 	        #set output directory so that .o files end up here
        #$ -j yes		        #merge .e and .o files into one
        
        export MKL_NUM_THREADS=1 #to make some python threaded code play well, all =1
        export NUMEXPR_NUM_THREADS=1
        export OMP_NUM_THREADS=1

        {CODE}
        \\\"""
    """
    import os
    import stat
    import subprocess

    if template_text is None:
        ## define the template and script to create, save, and run qsub files
        ## yes, this is the one that I used...
        template_text = """#!/bin/bash
## ====================================================================== ##
## 2015_09 Chris Steele
## {DESCRIPTION}
## ====================================================================== ##
##
#$ -N {NAME}	    #set job name
#$ -pe smp {NTHREADS}	#set number of threads to use
#$ -l h_vmem={MEM}G	    #this is a per-thread amount of virtual memory, I think...
#$ -l h_stack=8M 	    #required to allow multiple threads to work correctly
#$ -V 			        #inherit user env from submitting shell
#$ -wd {OUTDIR} 	    #set working directory so that .o files end up here (maybe superseded?)
#$ -o {OUTDIR} 	        #set output directory so that .o files end up here
#$ -j yes		        #merge .e and .o files into one

export MKL_NUM_THREADS=1 #to make some python threaded code play well, all =1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

{CODE}
"""

    subFullName = os.path.join(outdir, 'XXX_' + name + '.sub')
    open(subFullName, 'wb').write(template_text.format(NAME=name, NTHREADS=nthreads, MEM=mem, OUTDIR=outdir, \
                                                       DESCRIPTION=description, CODE=code))
    st = os.stat(subFullName)
    os.chmod(subFullName, st.st_mode | stat.S_IEXEC)  # make executable
    if SUBMIT:
        subprocess.call(['qsub', subFullName])


def qcheck(user='stechr', delay=5 * 60):
    """
    Check if que is clear for user at delay intervals (s)
    """
    import time
    import subprocess

    print(time.strftime("%Y_%m_%d %H:%M:%S"))
    print "=== start time ===",
    start = time.time()
    print(start)
    try:
        while len(subprocess.check_output(['qstat', '-u', user, '|', 'grep', user], shell=True)) > 0:
            print ". ",
            # print(len(subprocess.check_output(['qstat', '-u', 'tachr'],shell=True)))
            time.sleep(delay)
    except:
        pass

    print "=== end time ===",
    print(time.time())
    print(time.strftime("%Y_%m_%d %H:%M:%S"))
    duration = time.time() - start
    print("Duration: " + str(duration) + " (s)")


def print_file_array(in_file_array):
    """
    Convenience function to print file names from array to stdout
    """
    import os
    print(os.path.dirname(in_file_array[0]))
    for line in in_file_array:
        print(os.path.basename(line))


def tract_seg3(files, out_basename='', segmentation_index=None, CLOBBER=False, BY_SLICE=False):
    """
    2015_09
    Christopher J Steele
    Winner takes all segmentation of tract density images (.nii/.nii.gz)

    Input:
        - files:                list of tract density files for segmentation (with full pathname)
        - out_basename:         basename for output
        - segmentation_index:   option to map default 1-based indexing (where the first input file is label 1)
                                to custom index. Input must be a numpy array of len(files), and map to their order in files
        - CLOBBER:              over-write or not {True,False}
        - BY_SLICE:             perform segmentation slice by slice (in 3rd dimension) to reduce memory requirements
                                (note that this unzips each .nii.gz file once to reduce overhead, and zips when finished)
    """
    # improved version, processes by slice quickly after unzipping the input .gz files
    # will also work on raw .nii files, but will zip them at the end :)

    import os
    import numpy as np
    import nibabel as nb
    import subprocess

    print('You have input {num} files for segmentation'.format(num=len(files)))
    print('Your segmentation index is: {seg}'.format(seg=segmentation_index))
    print_file_array(files)
    print("Output basename: " + out_basename)

    if os.path.dirname(out_basename) == '':  # if they didn't bother to set a path, same as input
        out_dir = os.path.dirname(files[0])
    else:
        out_dir = os.path.dirname(out_basename)

    seg_idx_fname = os.path.join(out_dir, out_basename) + '_seg_idx.nii.gz'
    seg_tot_fname = os.path.join(out_dir, out_basename) + '_seg_tot.nii.gz'
    seg_prt_fname = os.path.join(out_dir, out_basename) + '_seg_prt.nii.gz'
    seg_pct_fname = os.path.join(out_dir, out_basename) + '_seg_pct.nii.gz'

    if not (os.path.isfile(seg_idx_fname)) or CLOBBER:  # if the idx file exists, don't bother doing this again
        if not BY_SLICE:
            data_list = [nb.load(fn).get_data()[..., np.newaxis] for fn in files]  # load all of the files
            combined = np.concatenate(data_list, axis=-1)  # concatenate all of the input data

            combined = np.concatenate((np.zeros_like(data_list[0]), combined),
                                      axis=-1)  # add a volume of zeros to padd axis and make calculations work correctly
            print("Data shape (all combined): " + str(np.shape(combined)))

            del data_list  # remove from memory, hopefully...

            ##%% hard segmentation (tract w/ largest number of streamlines in each voxel wins)
            # uses argmax to return the index of the volume that has the largest value (adds 1 to be 1-based)
            hard_seg = combined.argmax(axis=-1)  # now we have a 1-based segmentation (largest number in each voxel)
            hard_seg[combined.std(
                axis=-1) == 0] = 0  # where there is no difference between volumes, this should be the mask, set to 0

            ##%% create soft segmentation to show strength of the dominant tract in each voxel
            seg_part = np.zeros_like(hard_seg)
            seg_temp = np.zeros_like(hard_seg)
            seg_total = combined.sum(axis=-1)

            idx = 1
            for seg in files:
                seg_temp = combined[:, :, :,
                           idx]  # get value at this voxel for this tract seg (-1 for 0-based index of volumes)
                seg_part[hard_seg == idx] = seg_temp[hard_seg == idx]  # 1-based index of segmentation
                idx += 1

            # recode simple 1-based index into user-defined index
            if segmentation_index is not None:
                # check that we have the correct number of index values
                hard_seg_indexed = np.zeros_like(hard_seg)
                if len(files) == len(segmentation_index):
                    idx = 1
                    for seg_val in segmentation_index:
                        hard_seg_indexed[hard_seg == idx] = seg_val
                        idx += 1
                else:
                    print ""
                    print("====== YOU DID NOT ENTER THE CORRECT NUMBER OF VALUES FOR segmentation_index ======")
                    return

                np.copyto(hard_seg, hard_seg_indexed)
                del hard_seg_indexed  # be free, my memory!

            # seg_pct = seg_part/seg_total
            seg_pct = np.where(seg_total > 0, seg_part.astype(np.float32) / seg_total.astype(np.float32),
                               0)  # where there is no std (regions with no tracts) return 0, otherwise do the division
            # seg_pct[seg_pct==float('-Inf')] = 999

            # convert so that each segmentation goes from above its segmented to number to just below +1
            # .001 added to make sure that segmentations where tracts are 100% do not push into the next segmentation (not necessary depending on how the images are displayed)
            # 1st is 1-1.999, 2nd is 2-3.... (though the values should always be above the integer b/c of the segmentation
            # seg_pct=np.add(seg_pct,hard_seg) #add them and subtract a value, now the values are percentages of the segmentations for each number

            """
            # XXX This no longer works because we are assigning different index values to our segmentation
            # new way: double them to provide more space,
            #-1 sets the zero point at one below double the idx
            # add the pct to modulate accordingly
            # now idx 1 goes from 1-2 (0-100%) and 2 from 3-4... 5-6,7-8,9-10
            """
            # seg_pct2=(hard_seg.astype(np.float32)*2-1)+seg_pct
            # seg_pct2[seg_pct2==-1]=0 #remove those -1s in the regions that used to be 0

            ##%%save
            aff = nb.load(files[0]).affine
            header = nb.load(files[0]).header

            new_nii = nb.Nifti1Image(hard_seg.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_idx_fname)

            new_nii = nb.Nifti1Image(seg_total.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_tot_fname)

            new_nii = nb.Nifti1Image(seg_part.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_prt_fname)

            """
            # this should give us a combined segmentation and % of seg that is from the one that won, but
            # it does not currently work for all cases, so now just reports the percentage winner in each voxel
            # without any indication of who won the segmentation
            # XXX change to pct2 when it works :)
            """
            new_nii = nb.Nifti1Image(seg_pct, aff, header)
            new_nii.set_data_dtype(
                'float32')  # since our base file is where we get the datatype, set explicitly to float here
            new_nii.to_filename(seg_pct_fname)

            print("All segmentation files have been written")

        else:  # we are going to process this for each slice separately to see what our mem usage looks like
            print("Processing images slice by slice to conserve memory")

            # first we uncompress all of the data
            for gz_file in files:
                cmd = ['gunzip', gz_file]
                subprocess.call(cmd)

            files_nii = [fn.strip('.gz') for fn in files]
            files = files_nii

            data_shape = nb.load(files[0]).shape

            hard_seg_full = np.zeros(data_shape)
            seg_part_full = np.zeros(data_shape)
            seg_total_full = np.zeros(data_shape)
            seg_pct_full = np.zeros_like(hard_seg_full)

            print("Data shape (single image): " + str(data_shape))
            print("Slice: "),

            # loop over the last axis
            for slice_idx in np.arange(0, data_shape[-1]):
                print(slice_idx),

                data_list = [nb.load(fn).get_data()[:, :, slice_idx, np.newaxis] for fn in
                             files]  # load all of the files
                combined = np.concatenate(data_list, axis=-1)  # concatenate all of the input data
                combined = np.concatenate((np.zeros_like(data_list[0]), combined),
                                          axis=-1)  # add a volume of zeros to padd axis and make calculations work correctly
                if np.any(combined):  # if all voxels ==0, skip this slice entirely
                    ##%% hard segmentation (tract w/ largest number of streamlines in each voxel wins)
                    # uses argmax to return the index of the volume that has the largest value (adds 1 to be 1-based)
                    hard_seg = combined.argmax(axis=-1)
                    # now we have a 1-based segmentation (largest number in each voxel), where number corresponds to input file order
                    hard_seg[combined.std(
                        axis=-1) == 0] = 0  # where there is no difference between volumes, this should be the mask, set to 0

                    hard_seg_full[:, :, slice_idx] = hard_seg

                    ##%% create soft segmentation to show strength of the dominant tract in each voxel
                    seg_total_full[:, :, slice_idx] = combined.sum(axis=-1)

                    # declare empty matrices for this loop for partial and temp for calculating the partial (num of winning seg) file
                    seg_part = np.zeros_like(hard_seg)
                    seg_temp = np.zeros_like(hard_seg)

                    idx = 1
                    for seg in files:
                        seg_temp = combined[:, :,
                                   idx]  # get value at this voxel for this tract seg (-1 for 0-based index of volumes)
                        seg_part[hard_seg == idx] = seg_temp[hard_seg == idx]  # 1-based index of segmentation
                        idx += 1

                    seg_part_full[:, :, slice_idx] = seg_part

                    # recode simple 1-based index into user-defined index for hard_seg
                    if segmentation_index is not None:
                        # check that we have the correct number of index values
                        hard_seg_indexed = np.zeros_like(hard_seg)
                        if len(files) == len(segmentation_index):
                            idx = 1
                            for seg_val in segmentation_index:
                                hard_seg_indexed[hard_seg == idx] = seg_val
                                idx += 1
                        else:
                            print ""
                            print("====== YOU DID NOT ENTER THE CORRECT NUMBER OF VALUES FOR segmentation_index ======")
                            return None

                        np.copyto(hard_seg_full[:, :, slice_idx], hard_seg_indexed)
                        del hard_seg_indexed  # be free, my memory!
                    seg_pct_full[:, :, slice_idx] = np.where(seg_total_full[:, :, slice_idx] > 0,
                                                             seg_part.astype(np.float32) / seg_total_full[:, :,
                                                                                           slice_idx].astype(
                                                                 np.float32),
                                                             0)  # where there is no std (regions with no tracts) return 0, otherwise do the division

            ##%%save
            aff = nb.load(files[0]).affine
            header = nb.load(files[0]).header

            new_nii = nb.Nifti1Image(hard_seg_full.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_idx_fname)

            new_nii = nb.Nifti1Image(seg_total_full.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_tot_fname)

            new_nii = nb.Nifti1Image(seg_part_full.astype('uint32'), aff, header)
            new_nii.set_data_dtype('uint32')
            new_nii.to_filename(seg_prt_fname)

            """
            # this should give us a combined segmentation and % of seg that is from the one that won, but
            # it does not currently work for all cases, so now just reports the percentage winner in each voxel
            # without any indication of who won the segmentation
            # XXX change to pct2 when it works :)
            """
            new_nii = nb.Nifti1Image(seg_pct_full, aff, header)
            new_nii.set_data_dtype(
                'float32')  # since our base file is where we get the datatype, set explicitly to float here
            new_nii.to_filename(seg_pct_fname)

            # lets compress those files back to what they were, so everyone is happy with how much space they take
            for nii_file in files:
                cmd = ['gzip', nii_file]
                subprocess.call(cmd)

            print("")
            print("All segmentation files have been written")
        # return hard_seg_full, seg_part_full, seg_total_full, seg_pct_full, combined
        print("")
    else:
        print(
            "The index file already exists and I am not going to overwrite it because you didn't tell me to CLOBBER it! (" + seg_idx_fname + ")")


def sanitize_bvals(bvals, target_bvals=[0, 1000, 2000, 3000]):
    """
    Remove small variation in bvals and bring them to their closest target bvals
    """
    for idx, bval in enumerate(bvals):
        bvals[idx] = min(target_bvals, key=lambda x: abs(x - bval))
    return bvals


###OLD  
# def dki_prep_data_bvals_bvecs(data_fname,bvals_file,bvecs_file,bval_max_cutoff=2500,CLOBBER=False):
#    """
#    Selects only the data and bvals/bvecs that are below the bval_max_cutoff, writes to files in input dir
#    Useful for the dipy version
#    """
#    import os
#    import numpy as np
#    import subprocess
#    
#    bvals=np.loadtxt(bvals_file)
#    bvecs=np.loadtxt(bvecs_file)
#    vol_list=str([i for i,v in enumerate(bvals) if v < bval_max_cutoff]).strip('[]').replace(" ","") #strip the []s and remove spaces
#    out_fname=data_fname.split(".nii")[0] + "_bvals_under" +str(bval_max_cutoff) + ".nii.gz"
#    bvals_fname=bvals_file.split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
#    bvecs_fname=bvecs_file.split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
#    
#    if not(os.path.isfile(out_fname)) or CLOBBER:
#        cmd_input=['fslselectvols','-i',data_fname,'-o',out_fname,'--vols='+vol_list]
#        np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
#        np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
#        #print(cmd_input)
#        subprocess.call(cmd_input)
#    else:
#        print("File exists, not overwriting.")
#    return out_fname, bvals[bvals<bval_max_cutoff], bvecs[:,bvals<bval_max_cutoff]

def dki_dke_prep_data_bvals_bvecs(data_fname, bvals_file, bvecs_file, out_dir=None, bval_max_cutoff=2500,
                                  target_bvals=[0, 1000, 2000, 3000], ROTATE_OUTPUT=True, CLOBBER=False,
                                  RUN_LOCALLY=False):
    """
    Selects only the data and bvals/bvecs that are below the bval_max_cutoff, writes to files in input dir
    Automatically sanitizes your bvals for you, you don't get a choice here
    """
    import os
    import numpy as np
    import subprocess

    if out_dir is None:
        out_dir = os.path.dirname(data_fname)

    bvals = np.loadtxt(bvals_file)
    bvals = sanitize_bvals(bvals, target_bvals=target_bvals)
    bvecs = np.loadtxt(bvecs_file)
    vol_list = str([i for i, v in enumerate(bvals) if v < bval_max_cutoff]).strip('[]').replace(" ",
                                                                                                "")  # strip the []s and remove spaces so that we can have correct format for command line
    bvals_fname = os.path.basename(bvals_file).split(".")[0] + "_bvals_under" + str(bval_max_cutoff)
    bvals_fname = os.path.join(out_dir, bvals_fname)

    fname_list = []  # keeps track of the bval files that we have written, so we can merge them
    bvecs_fnames = []
    bvals_used = []

    bvals_orig = bvals
    bvecs_orig = bvecs
    cmd_txt = []
    for bval in target_bvals:  # split the file into its bvals, saves, merges, uses .nii
        if bval <= bval_max_cutoff:
            out_fname = os.path.join(out_dir,
                                     os.path.basename(data_fname).split(".nii")[0] + "_bval" + str(bval) + ".nii.gz")
            vol_list = str([i for i, v in enumerate(bvals) if v == bval]).strip('[]').replace(" ", "")
            cmd_input = ['fslselectvols', '-i', data_fname, '-o', out_fname, '--vols=' + vol_list]
            print ""
            print " ".join(cmd_input)
            cmd_txt.append(cmd_input)
            if not os.path.isfile(out_fname) or CLOBBER:
                if RUN_LOCALLY:
                    subprocess.call(cmd_input)
            if bval == 0:  # we mean this value if we are working with b=0 file
                cmd_input = ['fslmaths', out_fname, '-Tmean', out_fname]
                print " ".join(cmd_input)
                cmd_txt.append(cmd_input)
                if RUN_LOCALLY:
                    subprocess.call(cmd_input)  # no CLOBBER check here, since we actually want to overwrite this file
            else:  # non-b0 images should have their own bvecs files
                bvecs_fname = os.path.basename(bvecs_file).split(".")[0] + "_bval" + str(bval)
                bvecs_fname = os.path.join(out_dir, bvecs_fname)
                bvecs = bvecs_orig[:, bvals_orig == bval]
                if ROTATE_OUTPUT:
                    bvecs = bvecs.T
                np.savetxt(bvecs_fname, bvecs, fmt="%5.10f")

                bvecs_fnames.append(bvecs_fname)
            bvals_used.append(str(bval))
            fname_list.append(out_fname)
    out_fname = os.path.join(out_dir, os.path.basename(data_fname).split(".nii")[0] + "_dke_bvals_to_" + str(
        bval_max_cutoff) + ".nii")  # fsl only outputs GZ, so the name here is more for the input to the DKE, which only accepts .nii :-(
    cmd_input = ['fslmerge', '-t', out_fname]
    for fname in fname_list:
        cmd_input = cmd_input + [fname]
    print ""
    print " ".join(cmd_input)
    cmd_txt.append(cmd_input)
    if not os.path.isfile(out_fname) or CLOBBER:
        if RUN_LOCALLY:
            subprocess.call(cmd_input)
    cmd_input = ['gunzip', out_fname + '.gz']
    cmd_txt.append(cmd_input)
    if not os.path.isfile(out_fname) or CLOBBER:
        if RUN_LOCALLY:
            subprocess.call(cmd_input)
    return [out_fname, bvals_used, bvecs_fnames,
            cmd_txt]  # all returned as strings XXX COULD ALSO ADD numdirs (per b-value) and vox_dim


def run_diffusion_kurtosis_estimator(sub_root_dir, ID, data_fname, bvals_file, bvecs_file, out_dir=None,
                                     bval_max_cutoff=2500, template_file='HCP_dke_commandLine_parameters_TEMPLATE.dat',
                                     SUBMIT=True, CLOBBER=False):
    """
    Run the command-line diffusion kurtosis estimator
    Input:
        - sub_root_dir  - subject root directory
        - ID            - subject ID (off of root dir) (string)
        - data_fname    - 4d diffusion data (raw)
        - bvals_file    - b-values file
        - bvecs_file    - b-vectors file
        - out_dir       - directory where you want the output to go (full)
        - TEMPLATE      - template file for dke, provided by the group
    dki_dke_prep_data_bvals_bvecs(data_fname='/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/data.nii.gz',bvals_file='/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvals',bvecs_file='/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvecs',out_dir='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/DKI/100307')
    """
    import os
    import numpy as np
    import nibabel as nb

    GAUSS_SMTH_MULTIPLIER = 1.25  # taken from the DKI papers
    if out_dir is None:
        out_dir = os.path.join(sub_root_dir, ID)

    TEMPLATE = open(template_file).read()
    full_fname = os.path.join(sub_root_dir, ID, data_fname)

    # this next part takes some time, since it divides up the diffusion shells writes them to disk (with bvecs)
    fnames = dki_dke_prep_data_bvals_bvecs(data_fname=full_fname, bvals_file=bvals_file, bvecs_file=bvecs_file,
                                           out_dir=out_dir, bval_max_cutoff=bval_max_cutoff, CLOBBER=CLOBBER,
                                           RUN_LOCALLY=False)

    num_diff_dirs = 90  # this is also generated below and used to compare it? divide the dirs? in the bvec files?
    sample_bvecs = np.loadtxt(fnames[2][0])
    num_diff_dirs_2 = max(np.shape(sample_bvecs))

    if not num_diff_dirs == num_diff_dirs_2:
        print("##=========================================================================##")
        print("Oh damn, things are not going well!")
        print("The number of diffusion directions do not appear to be correct for the HCP")
        print("Be sad. :-( ")
        print("##=========================================================================##")
        return
    dke_data_fname = os.path.basename(fnames[0])
    v = nb.load(full_fname).get_header()['pixdim'][1:4] * GAUSS_SMTH_MULTIPLIER
    vox_dims = " ".join(map(str, v))  # map to string, then convert to the format that we need
    print(dke_data_fname)
    bvals_used = " ".join(fnames[1])  # list of bvals used
    bvecs_fnames = ", ".join(
        ["'{0}'".format(os.path.basename(fname)) for fname in fnames[2]])  # list of filenames of bvecs

    sub_root_out_dir = out_dir.strip(ID)  # because this script is annoying...
    dke_params_dat_fullname = os.path.join(out_dir, "XXX_" + ID + '_DKE_parameters.dat')
    TEMPLATE = TEMPLATE.format(SUB_ROOT_DIR=sub_root_out_dir, ID=ID, DKE_DATA_FNAME=dke_data_fname,
                               BVALS_USED=bvals_used, BVECS_FNAMES=bvecs_fnames, NUM_DIFF_DIRS=num_diff_dirs,
                               VOX_DIMS=vox_dims)
    open(dke_params_dat_fullname, 'wb').write(TEMPLATE)

    # now start the module for what we need or assume that it is running, and run the script
    jname = "DKE_" + ID + "_CJS"
    code = """module load DKE/2015.10.28\nrun_dke.sh /opt/quarantine/DKE/2015.10.28/build/v717 {PARAMS}
	""".format(PARAMS=dke_params_dat_fullname)
    cmd_txt = fnames[3]
    cmd_txt = [" ".join(cmd) for cmd in cmd_txt]  # to create a list of strings instead of list of lists
    code = "\n\n".join(cmd_txt) + "\n\n" + code
    print(os.path.join(sub_root_dir, ID))
    # this job requires over 18GB for the HCP data
    submit_via_qsub(code=code, description="Diffusion kurtosis estimation", name=jname, outdir=out_dir, nthreads=6,
                    mem=4.0, SUBMIT=SUBMIT)
