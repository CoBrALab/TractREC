# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:18:20 2015
Reconstructions and data preprocessing, including XXX
@author: stechr
"""
from TractREC import imgLoad
from TractREC import niiSave
from TractREC import create_dir

def sanitize_bvals(bvals,target_bvals=[0,1000,2000,3000]):
    """
    Remove small variation in bvals and bring them to their closest target bvals
    Returns bvals equal to the set provided in target_bvals 
    """
    for idx,bval in enumerate(bvals):
        bvals[idx]=min(target_bvals, key=lambda x:abs(x-bval))
    return bvals
    
def select_and_write_data_bvals_bvecs(data_fname,bvals_file,bvecs_file,out_dir=None,bval_max_cutoff=2500,CLOBBER=False):    
    """
    Create subset of data with the bvals that you are interested in (uses fslselectvols instead of loading into memory)
    Selects only the data and bvals/bvecs that are below the bval_max_cutoff, writes to files in input dir
    Returns output_filename, bvals, bvecs
    """
    import os
    import numpy as np
    import subprocess

    if out_dir is None:
        out_dir=os.path.dirname(data_fname)
    create_dir(out_dir)
    
    bvals=np.loadtxt(bvals_file)
    bvecs=np.loadtxt(bvecs_file)
    
    #alterative would be to load this into memory, but difficult when working with larger datasets like HCP so we use fsl here
    #XXX add option for doing this within python (would be faster!)    
    vol_list=str([i for i,v in enumerate(bvals) if v < bval_max_cutoff]).strip('[]').replace(" ","") #strip the []s and remove spaces to format as expected
    
    #rename and point to the correct directory
    out_fname=os.path.basename(data_fname).split(".nii")[0] + "_bvals_under" +str(bval_max_cutoff) + ".nii.gz"
    bvals_fname=os.path.basename(bvals_file).split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    bvecs_fname=os.path.basename(bvecs_file).split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    out_fname=os.path.join(out_dir,out_fname)
    bvals_fname=os.path.join(out_dir,bvals_fname)
    bvecs_fname=os.path.join(out_dir,bvecs_fname)
    
    cmd_input=['fslselectvols','-i',data_fname,'-o',out_fname,'--vols='+vol_list]
    print(cmd_input)
    if not(os.path.isfile(out_fname)) or CLOBBER:
        np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
        np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
        subprocess.call(cmd_input)
    else:
        print("File exists, not overwriting.")
    return out_fname, bvals[bvals<bval_max_cutoff], bvecs[:,bvals<bval_max_cutoff]

def DKE_by_slice(data,gtab,slices='all'):
    """
    Fits the DKE model by slice to decrease memory requirements
    Do all slices, or array subset thereof
    """
    import dipy.reconst.dki as dki    
    import numpy as np
    
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    n_contrasts=3 #number of contrasts that we are going to have output from the dki model

    #lets loop across the z dimension - index 2
    out_data=np.zeros(np.shape(data)[0:3]+n_contrasts,) #replace the diff dir axis with our own for the results
    if slices is 'all':    
        slices=np.arange(0,np.shape(data)[2])
        
    for zslice in slices:
        slice_d=data[:,:,zslice,:]
        
        dkifit=dkimodel.fit(slice_d)
        MK = dkifit.mk(0, 3)
        AK = dkifit.ak(0, 3)
        RK = dkifit.rk(0, 3)
        
        #assign to our out_data
        out_data[:,:,zslice,:,0]=MK
        out_data[:,:,zslice,:,1]=AK
        out_data[:,:,zslice,:,2]=RK
    return out_data
    
def DKE(data_fname,bvals_fname,bvecs_fname,bval_max_cutoff=3200,out_dir=None,slices='all',NLMEANS_DENOISE=False):
    """
    DKE with dipy (dipy.__version__>=0.10.0), outputs MK, AK, and RK without and (potentially) with denoising
    """
    from dipy.core.gradients import gradient_table
    from dipy.segment.mask import median_otsu
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.denoise.nlmeans import nlmeans
    import os

    if out_dir is None:
        out_dir=os.path.dirname(data_fname)
    create_dir(out_dir)
    
    out_fname_base=os.path.join(out_dir,"DKE_")
    selected_data_fname,bvals,bvecs = select_and_write_data_bvals_bvecs(data_fname,bvals_fname,bvecs_fname,bval_max_cutoff=3200)
    data,aff=imgLoad(selected_data_fname)
    bvals=sanitize_bvals(bvals)
    gtab = gradient_table(bvals, bvecs)
    
    #XXX vol_idx could be set according to b0 if you like?
    maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)
    
    #denoising could be necessary because DKE is sensitive to outliers, look to be able to skip this for HCP data
    if NLMEANS_DENOISE:
        sigma = estimate_sigma(data, N=4)
        den = nlmeans(data, sigma=sigma, mask=mask.astype('bool'))
    
    #initiate and run the DKE model
    DK_stats=DKE_by_slice(data,gtab,slices=slices)
    del data #clear this from mem, just in case it is huuuuge!
    
    out_fname=out_fname_base+"MK.nii.gz"
    niiSave(out_fname,DK_stats[...,0],aff)
    out_fname=out_fname_base+"AK.nii.gz"
    niiSave(out_fname,DK_stats[...,1],aff)
    out_fname=out_fname_base+"RK.nii.gz"
    niiSave(out_fname,DK_stats[...,2],aff)
    del DK_stats #remove from mem
    
    if NLMEANS_DENOISE:
        DK_stats_den=DKE_by_slice(den,gtab,slices=slices)
        out_fname=out_fname_base+"MK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,0],aff)
        out_fname=out_fname_base+"AK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,1],aff)
        out_fname=out_fname_base+"RK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,2],aff)
        del DK_stats_den
        
        