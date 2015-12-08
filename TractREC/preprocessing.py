# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:18:20 2015
Reconstructions and data preprocessing, including XXX
@author: stechr
"""
from TractREC import imgLoad
from TractREC import niiSave
from TractREC import create_dir
from TractREC import submit_via_qsub

def sanitize_bvals(bvals,target_bvals=[0,1000,2000,3000]):
    """
    Remove small variation in bvals and bring them to their closest target bvals
    Returns bvals equal to the set provided in target_bvals 
    """
    for idx,bval in enumerate(bvals):
        bvals[idx]=min(target_bvals, key=lambda x:abs(x-bval))
    return bvals
    
def select_and_write_data_bvals_bvecs(data_fname,bvals_file,bvecs_file,out_dir=None,bval_max_cutoff=3500,CLOBBER=False,IN_MEM=False):    
    """
    Create subset of data with the bvals that you are interested in (uses fslselectvols instead of loading into memory)
    Selects only the data and bvals/bvecs that are below the bval_max_cutoff, writes to files in input dir
    Returns output_filename, bvals, bvecs and selects vols in memory when IN_MEM=True
    """
    import os
    import numpy as np
    import subprocess

    if out_dir is None:
        out_dir=os.path.dirname(data_fname)
    create_dir(out_dir)
    
    bvals=np.loadtxt(bvals_file)
    bvecs=np.loadtxt(bvecs_file)
    
    vol_list=[i for i,v in enumerate(bvals) if v < bval_max_cutoff]
    
    #rename and point to the correct directory
    out_fname=os.path.basename(data_fname).split(".nii")[0] + "_bvals_under" +str(bval_max_cutoff) + ".nii.gz"
    bvals_fname=os.path.basename(bvals_file).split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    bvecs_fname=os.path.basename(bvecs_file).split(".")[0]+ "_bvals_under"+str(bval_max_cutoff)
    out_fname=os.path.join(out_dir,out_fname)
    bvals_fname=os.path.join(out_dir,bvals_fname)
    bvecs_fname=os.path.join(out_dir,bvecs_fname)
    
    print('Selecting appropriate volumes and bvals/bvecs for DKE.')
    print('Output to file: ' + out_fname)
    
    if not IN_MEM: #if we think that it is going to be too big for memory, we use the fsl command-line tool
        vol_list=str(vol_list).strip('[]').replace(" ","") #strip the []s and remove spaces to format as expected by fslselectcols
        cmd_input=['fslselectvols','-i',data_fname,'-o',out_fname,'--vols='+vol_list]
        print(cmd_input)
        if not(os.path.isfile(out_fname)) or CLOBBER:
            np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
            np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
            subprocess.call(cmd_input)
        else:
            print("File exists, not overwriting.")
    else:
        if not(os.path.isfile(out_fname)) or CLOBBER:
            data,aff=imgLoad(data_fname)
            niiSave(out_fname,data[...,vol_list],aff,CLOBBER=CLOBBER)
            np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
            np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
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
    
    print('Creating diffusion kurtosis model')
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    n_contrasts=3 #number of contrasts that we are going to have output from the dki model

    
    out_data=np.zeros(list(np.shape(data)[0:3])+[n_contrasts]) #replace the diff dir axis with our own for the results
    if slices is 'all':    
        slices=np.arange(0,np.shape(data)[2])
    print("Performing diffusion kurtosis estimation by slice: "),    
    #lets loop across the z dimension - index 2
    for zslice in slices:
        print(zslice),
        slice_d=data[:,:,zslice,:]
        
        dkifit=dkimodel.fit(slice_d)
        MK = dkifit.mk(0, 3)
        AK = dkifit.ak(0, 3)
        RK = dkifit.rk(0, 3)
        
        #assign to our out_data
        out_data[:,:,zslice,0]=MK
        out_data[:,:,zslice,1]=AK
        out_data[:,:,zslice,2]=RK
    print("")
    return out_data
    
def DKE(data_fname,bvals_fname,bvecs_fname,bval_max_cutoff=3200,out_dir=None,slices='all',NLMEANS_DENOISE=False,IN_MEM=False):
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
    print("Selecting appropriate data and writing to disk")
    selected_data_fname,bvals,bvecs = select_and_write_data_bvals_bvecs(data_fname,bvals_fname,bvecs_fname,out_dir=out_dir,bval_max_cutoff=3500,IN_MEM=IN_MEM)
    data,aff=imgLoad(selected_data_fname)
    bvals=sanitize_bvals(bvals)
    gtab=gradient_table(bvals, bvecs)
    
    #XXX vol_idx could be set according to b0 if you like?
    print("Creating brain mask")    
    maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)
    
    if slices is not 'all':
        maskdata[:,:,slices,:]
    #denoising could be necessary because DKE is sensitive to outliers, look to be able to skip this for HCP data, aslo runs out of memory with this data...
    if NLMEANS_DENOISE:
        sigma = estimate_sigma(data, N=4)
        den = nlmeans(data, sigma=sigma, mask=mask.astype('bool'))
    
    del data
    #initiate and run the DKE model
    DK_stats=DKE_by_slice(maskdata,gtab,slices=slices)
    del maskdata #clear this from mem, just in case it is huuuuge!
    
    out_fname=out_fname_base+"MK.nii.gz"
    niiSave(out_fname,DK_stats[...,0],aff)
    out_fname=out_fname_base+"AK.nii.gz"
    niiSave(out_fname,DK_stats[...,1],aff)
    out_fname=out_fname_base+"RK.nii.gz"
    niiSave(out_fname,DK_stats[...,2],aff)
    del DK_stats #remove from mem
    
    if NLMEANS_DENOISE:
        print("Now do the same thing with the denoised data!")
        DK_stats_den=DKE_by_slice(den,gtab,slices=slices)
        out_fname=out_fname_base+"MK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,0],aff)
        out_fname=out_fname_base+"AK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,1],aff)
        out_fname=out_fname_base+"RK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,2],aff)
        del DK_stats_den

def create_python_exec(out_dir,code=["#!/usr/bin/python",""],name="CJS_py"):
    import os
    import stat

    code="\n".join(code) #create a single string for saving to file, separated by carriage returns
    
    subFullName=os.path.join(out_dir,'XXX_'+name+'.py')
    open(subFullName,'wb').write(code)
    st = os.stat(subFullName)
    os.chmod(subFullName,st.st_mode | stat.S_IEXEC) #make executable
    return subFullName
    
def run_diffusion_kurtosis_estimatory_dipy(data_fnames,bvals_fnames,bvecs_fnames,out_root_dir,IDs=None,TractREC_path='/home/cic/stechr/Documents/code/TractREC/TractREC',bval_max_cutoff=3200,slices='all',NLMEANS_DENOISE=False,IN_MEM=True):
    """
    Pass matched lists of data filenames, bval filenames, and bvec filenames, along with a root directory for the output
    """
    import os
    
    if not(len(data_fnames) == len(bvals_fnames)) or not(len(data_fnames) == len(bvecs_fnames)) or not(len(bvecs_fnames) == len(bvals_fnames)):
        print("Inconsistent number of files were input.")
        return
    
    for idx,fname in enumerate(data_fnames):
        bvals=bvals_fnames[idx]
        bvecs=bvecs_fnames[idx]
        if IDs is not None:
            ID=str(IDs[idx])
        else:#we try to get the ID
            print("Trying to pull the ID of this subject from the filename... hope it works :-/ ")
            ID=os.path.basename(fname).split("_")[0]
        print(ID)
        out_dir=os.path.join(out_root_dir,ID)
        create_dir(out_dir)
        code=["#!/usr/bin/python","","import sys","sys.path.append('{0}')".format(TractREC_path),"import preprocessing as pr"]
        code.append("pr.DKE('{data_fname}','{bvals_fname}','{bvecs_fname}',bval_max_cutoff={bval_max_cutoff},out_dir='{out_dir}',slices={slices},NLMEANS_DENOISE={NLMEANS_DENOISE},IN_MEM={IN_MEM})""".format(data_fname=fname,bvals_fname=bvals,bvecs_fname=bvecs,\
            bval_max_cutoff=bval_max_cutoff,out_dir=out_dir,slices=slices,NLMEANS_DENOISE=NLMEANS_DENOISE,IN_MEM=IN_MEM))
        py_sub_full_fname=create_python_exec(out_dir=out_dir,code=code,name=ID+'_DKE_dipy')
        
        submit_via_qsub(template_text=None,code="python " + py_sub_full_fname,name=ID+'_DKE_dipy',nthreads=4,mem=1.75,outdir=out_dir,\
                        description="Diffusion kurtosis estimation with dipy",SUBMIT=False)

#DKE('/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/data.nii.gz','/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvals',\
#    '/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvecs',\
#    out_dir='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/DKI/100307_dipy_3K_new',slices='all',NLMEANS_DENOISE=False,IN_MEM=True)
