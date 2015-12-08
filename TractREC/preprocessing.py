# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:18:20 2015
Reconstructions and data preprocessing, including XXX
@author: stechr
"""
from TractREC import niiLoad
from TractREC import niiSave

def sanitize_bvals(bvals,target_bvals=[0,1000,2000,3000]):
    """
    Remove small variation in bvals and bring them to their closest target bvals
    Returns bvals equal to the set provided in target_bvals 
    """
    for idx,bval in enumerate(bvals):
        bvals[idx]=min(target_bvals, key=lambda x:abs(x-bval))
    return bvals
    
def select_and_write_data_bvals_bvecs(data_fname,bvals_file,bvecs_file,bval_max_cutoff=2500,CLOBBER=False):    
    """
    Create subset of data with the bvals that you are interested in (uses fslselectvols instead of loading into memory)
    Selects only the data and bvals/bvecs that are below the bval_max_cutoff, writes to files in input dir
    Returns output_filename, bvals, bvecs
    """
    import os
    import numpy as np
    import subprocess
    
    bvals=np.loadtxt(bvals_file)
    bvecs=np.loadtxt(bvecs_file)
    
    #alterative would be to load this into memory, but difficult when working with larger datasets like HCP so we use fsl here
    #XXX add option for doing this within python (would be faster!)    
    vol_list=str([i for i,v in enumerate(bvals) if v < bval_max_cutoff]).strip('[]').replace(" ","") #strip the []s and remove spaces to format as expected
    out_fname=data_fname.split(".nii")[0] + "_bvals_under" +str(bval_max_cutoff) + ".nii.gz"
    bvals_fname=bvals_file.split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    bvecs_fname=bvecs_file.split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    
    cmd_input=['fslselectvols','-i',data_fname,'-o',out_fname,'--vols='+vol_list]
    print(cmd_input)
    if not(os.path.isfile(out_fname)) or CLOBBER:
        np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
        np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
        subprocess.call(cmd_input)
    else:
        print("File exists, not overwriting.")
    return out_fname, bvals[bvals<bval_max_cutoff], bvecs[:,bvals<bval_max_cutoff]

def DKI_run_DKE(data_fname,gtab,slices='all'):
    import dipy.reconst.dki as dki    
    import nibabel as nb
    import numpy as np
    
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    n_contrasts=3 #number of contrasts that we are going to have output from the dki model
    img=nb.load(data_fname)
    #lets loop across the z dimension - index 2
    out_data=np.zeros(img.shape[0:3]+n_contrasts,) #replace the diff dir axis with our own for the results
    if slices is 'all':    
        slices=np.arange(0,img.shape[2])
    for zslice in slices:
        slice_d=img.get_data[:,:,zslice,:]
        
        dkifit=dkimodel.fit(slice_d)
        MK = dkifit.mk(0, 3)
        AK = dkifit.ak(0, 3)
        RK = dkifit.rk(0, 3)
        
        #assign to our out_data
        out_data[:,:,zslice,:,0]=MK
        out_data[:,:,zslice,:,1]=AK
        out_data[:,:,zslice,:,2]=RK
    return out_data
    
