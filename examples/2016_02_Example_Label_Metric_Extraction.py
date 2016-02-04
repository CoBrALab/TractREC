#/data/chamal/projects/steele/working/HCP_CB_DWI/scripts

import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns

import sys
sys.path.append('/home/cic/stechr/Documents/code/TractREC/TractREC')
#from TractREC import * #my custom stuff
import TractREC as tr
%pylab inline


# ================================================================================ #
# Setup the directories and label csv file
# TODO: convert to use correct files for each subject
# ================================================================================ #

#INPUT:
#working_dir="/data/chamal/projects/steele/working/HCP_CB_DWI/processing/lobule_specific_tractography"
T1divT2_root_dir="/data/chamal/projects/Data_external/Data_Public_restricted/HCP/Images_other"
labels_dir='/data/chamal/projects/steele/working/HCP_CB_DWI/source/t1w/MAGeT_t1w_labels'

HCP_restricted_bx='/data/chamal/projects/steele/working/HCP_CB_DWI/source/bx/RESTRICTED_steelec_10_30_2015_11_14_9.csv'
HCP_unrestricted_bx='/data/chamal/projects/steele/working/HCP_CB_DWI/source/bx/HCP_S500_Bx_unrestricted_steelec_9_17_2015_9_46_45.csv'

#OUTPUT:
bx_output_dir='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/bx/'
HCP_t1divt2_fname=os.path.join(bx_output_dir,'2016_02_HCP_t1divt2_allSubjects.csv')

# csv file with segmentation for CB and label
all_label_seg_file='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/lobule_specific_tractography/cerebellum_labels.csv'
#all_label_seg_file='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/lobule_specific_tractography/2015_09_labels_CB_Hipp_Subcort_63.csv'
TEMP_OUT_DIR="/data/chamal/projects/steele/working/HCP_CB_DWI/processing/lobule_specific_tractography/masks/FA_corrected_labels"
zfill_num=3 #for padding of zeros before the lobule numbers in output file names

#create the columns for the dataframe and grab the files that we will use
label_search_string=os.path.join(labels_dir,"*labels.*")
label_seg_search_string=os.path.join(lobule_seg_input_dir,"*_idx.nii.gz")
t1divt2_search_string=os.path.join(T1divT2_root_dir,"*.mnc")
                                   
label_files=tr.natural_sort(glob.glob(label_search_string))
label_seg_files=tr.natural_sort(glob.glob(label_seg_search_string))
t1divt2_files=tr.natural_sort(glob.glob(t1divt2_search_string))


#get the label segmentation id file, reorder it so that we can use the indices correctly
all_label_seg=pd.read_csv(all_label_seg_file)
all_label_seg=all_label_seg.set_index(['Index']) #make the index column the actual index
all_label_seg #show us what it looks like!

#we are only interested in a subset of the lobules, so lets choose them here
all_lobule_subset_idx=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112] #R and L
#lobule_subset_idx=[2, 3, 4, 5, 6, 7, 8, 9, 10] #L only
#all_lobule_subset_idx=[133]

# test to get voxel counts from 20 Ss that have had MCP tract-based classification\
# THIS DOES NOT PRODUCE SENSIBLE RESULTS SINCE THE t1divt2 files are NOT IN THE SAME ORIENTATION! #

tr=reload(tr)
# create a list of the IDs so that extract_quantitative_metric can link up each metric file to each segmentation file.
# these need to be unique - files are chosen based on a simple lookup --> if thisID is in thisFileName
# this may not protect against cases where one ID is embedded within another (I haven't checked :-/) --> e.g., IDs 101 and 1101, if the filenames are 101_seg.mnc and 1101_seg.mnc
# i should test this...

t1divt2_IDs=[name.split("_")[-3] for name in t1divt2_files] #XXX NEED THIS FOR A HACK, CURRENTLY. Need to change extract_q_m

df_seg=tr.extract_quantitative_metric(t1divt2_files,label_files,IDs=t1divt2_IDs[0:20],ROI_mask_files=None,label_df=all_label_seg,\
                                      label_subset_idx=all_lobule_subset_idx,thresh_mask_files=fa_files,thresh_val=.35,\
                                      max_val=1,thresh_type='upper',erode_vox=1,metric='mean',DEBUG_DIR=TEMP_OUT_DIR,\
                                      VERBOSE=True)

df_seg.to_csv(HCP_t1divt2_fname)

