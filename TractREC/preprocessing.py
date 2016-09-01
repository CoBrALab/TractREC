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
from TractREC import get_img_bounds
from TractREC import crop_to_roi

def crop_image(img_fname,mask_fname=None,crop_out_fname=None, roi_coords=None, roi_buffer=3):
    """
    Crop an input image (3d or 4d) by image mask (1 = region to include in cropped file)
    :param img_fname:
    :param mask_fname:
    :param roi_coords:
    :param roi_buffer:
    :return:
    """
    import nibabel as nb
    d,a,zooms,h = imgLoad(img_fname,RETURN_RES=True,RETURN_HEADER=True)
    if mask_fname is not None:
        md, ma = imgLoad(mask_fname)
        roi_coords = get_img_bounds(md) + roi_buffer #this makes assumptions, only 0/1s in file TODO: checking?

    crop_d,roi_coords = crop_to_roi(d, roi_buffer=roi_buffer, roi_coords=roi_coords, data_4d=True)

    if crop_out_fname is not None:
        img=nb.Nifti1Image(crop_d,a,header=h)
        return img, crop_d, roi_coords
    else:
        niiSave(crop_out_fname,crop_d,a,header=h)
    return crop_d, roi_coords


#adapted code from nilearn for smoothing a dataset, rather than an img
def smooth_data_array(arr, affine, fwhm=None, ensure_finite=True, copy=True):    
    """
    TAKEN FROM NILEARN (the power behind nilearn.image.smooth_image)
    
    Smooth images by applying a Gaussian filter.
    Apply a Gaussian filter along the three first dimensions of arr.
    Parameters
    ==========
    arr: numpy.ndarray
        4D array, with image number as last dimension. 3D arrays are also
        accepted.
    affine: numpy.ndarray
        (4, 4) matrix, giving affine transformation for image. (3, 3) matrices
        are also accepted (only these coefficients are used).
        If fwhm='fast', the affine is not used and can be None
    fwhm: scalar, numpy.ndarray, 'fast' or None
        Smoothing strength, as a full-width at half maximum, in millimeters.
        If a scalar is given, width is identical on all three directions.
        A numpy.ndarray must have 3 elements, giving the FWHM along each axis.
        REMOVED: If fwhm == 'fast', a fast smoothing will be performed with
        a filter [0.2, 1, 0.2] in each direction and a normalisation
        to preserve the local average value.
        If fwhm is None, no filtering is performed (useful when just removal
        of non-finite values is needed).
    ensure_finite: bool
        if True, replace every non-finite values (like NaNs) by zero before
        filtering.
    copy: bool
        if True, input array is not modified. False by default: the filtering
        is performed in-place.
    Returns
    =======
    filtered_arr: numpy.ndarray
        arr, filtered.
    Notes
    =====
    This function is most efficient with arr in C order.
    """
    import numpy as np
    import scipy.ndimage as ndimage
    
    if arr.dtype.kind == 'i':
        if arr.dtype == np.int64:
            arr = arr.astype(np.float64)
        else:
            # We don't need crazy precision
            arr = arr.astype(np.float32)
    if copy:
        arr = arr.copy()

    if ensure_finite:
        # SPM tends to put NaNs in the data outside the brain
        arr[np.logical_not(np.isfinite(arr))] = 0

    if fwhm is not None:
        # Keep only the scale part.
        affine = affine[:3, :3]

        # Convert from a FWHM to a sigma:
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
        vox_size = np.sqrt(np.sum(affine ** 2, axis=0))
        sigma = fwhm / (fwhm_over_sigma_ratio * vox_size)
        for n, s in enumerate(sigma):
            ndimage.gaussian_filter1d(arr, s, output=arr, axis=n)

    return arr

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
    import subprocess
    import numpy as np
    import nibabel as nb    

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
    
    if len(vol_list) == nb.load(data_fname).shape[3]: #if we are going to select all of the volumes anyway, don't bother copying them!
        print("All bvals selected, using original data file as input")
        out_fname=data_fname
        np.savetxt(bvals_fname,bvals[bvals<bval_max_cutoff])
        np.savetxt(bvecs_fname,bvecs[:,bvals<bval_max_cutoff])
    else:
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
    
def DKE(data_fname,bvals_fname,bvecs_fname,bval_max_cutoff=3200,out_dir=None,slices='all',SMTH_DEN=None,IN_MEM=False):

    """
    DKE with dipy (dipy.__version__>=0.10.0), outputs MK, AK, and RK without and (potentially) with denoising
    SMTH_DEN can take multiple arguments in list format ['smth','nlmeans'] - currently always does DKE with native data as well (XXX could add this as 'natv')
    """
    from dipy.core.gradients import gradient_table
    from dipy.segment.mask import median_otsu
    from dipy.denoise.noise_estimate import estimate_sigma
    from dipy.denoise.nlmeans import nlmeans
    import os

    GAUSS_SMTH_MULTIPLIER=1.25 #taken from the DKI papers
    
    if out_dir is None:
        out_dir=os.path.dirname(data_fname)
    create_dir(out_dir)
    
    out_fname_base=os.path.join(out_dir,"DKE_")
    print("Selecting appropriate data and writing to disk")
    selected_data_fname,bvals,bvecs = select_and_write_data_bvals_bvecs(data_fname,bvals_fname,bvecs_fname,out_dir=out_dir,bval_max_cutoff=bval_max_cutoff,IN_MEM=IN_MEM)
    data,aff=imgLoad(selected_data_fname)
    bvals=sanitize_bvals(bvals)
    gtab=gradient_table(bvals, bvecs)
    
    #XXX vol_idx could be set according to b0 if you like, but this seems to work for now
    print("Creating brain mask")    
    maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)
      
    #denoising could be necessary because DKE is sensitive to outliers, look to be able to skip this for HCP data, aslo runs out of memory with this data...
    if 'nlmeans' in SMTH_DEN:
        sigma = estimate_sigma(data, N=4)
        den = nlmeans(data, sigma=sigma, mask=mask.astype('bool'))
    if 'smth' in SMTH_DEN:
        import nibabel as nb
        vox_dims=nb.load(selected_data_fname).get_header()['pixdim'][1:4]
        smth=smooth_data_array(data, aff, fwhm=vox_dims*GAUSS_SMTH_MULTIPLIER, ensure_finite=True, copy=True)
        smth[mask==0]=0
    del data
    
    #initiate and run the DKE model
    print("Running model on raw data")    
    print("=========================")    
    DK_stats=DKE_by_slice(maskdata,gtab,slices=slices)
    del maskdata #clear this from mem, just in case it is huuuuge!
    
    out_fname=out_fname_base+"MK.nii.gz"
    niiSave(out_fname,DK_stats[...,0],aff)
    out_fname=out_fname_base+"AK.nii.gz"
    niiSave(out_fname,DK_stats[...,1],aff)
    out_fname=out_fname_base+"RK.nii.gz"
    niiSave(out_fname,DK_stats[...,2],aff)
    del DK_stats #remove from mem
    
    if 'nlmeans' in SMTH_DEN:
        print("")
        print("Running the model on denoised data")
        print("==========================================")    
        DK_stats_den=DKE_by_slice(den,gtab,slices=slices)
        out_fname=out_fname_base+"MK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,0],aff)
        out_fname=out_fname_base+"AK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,1],aff)
        out_fname=out_fname_base+"RK_den.nii.gz"
        niiSave(out_fname,DK_stats_den[...,2],aff)
        del DK_stats_den
    
    if 'smth' in SMTH_DEN:
        print("")
        print("Running the model on smoothed data " + "(vox_dim*"+str(GAUSS_SMTH_MULTIPLIER)+")")
        print("=========================================================")    
        DK_stats_smth=DKE_by_slice(smth,gtab,slices=slices)
        out_fname=out_fname_base+"MK_smth.nii.gz"
        niiSave(out_fname,DK_stats_smth[...,0],aff)
        out_fname=out_fname_base+"AK_smth.nii.gz"
        niiSave(out_fname,DK_stats_smth[...,1],aff)
        out_fname=out_fname_base+"RK_smth.nii.gz"
        niiSave(out_fname,DK_stats_smth[...,2],aff)

def create_python_exec(out_dir,code=["#!/usr/bin/python",""],name="CJS_py"):
    """
    Create executable python code
    This code can then be wrapped with an SGE .sub call
    INPUT:
        - out_dir       directory where the .py file will be saved
        - code          list of code, where each element is a single line
        - name          name for creating filename of .py 
    RETURNS:
        - subFullName   full path name to .py file
    """    
    import os
    import stat

    code="\n".join(code) #create a single string for saving to file, separated by carriage returns
    
    subFullName=os.path.join(out_dir,'XXX_'+name+'.py')
    open(subFullName,'wb').write(code)
    st = os.stat(subFullName)
    os.chmod(subFullName,st.st_mode | stat.S_IEXEC) #make executable
    return subFullName
    
def run_diffusion_kurtosis_estimator_dipy(data_fnames,bvals_fnames,bvecs_fnames,out_root_dir,IDs,bval_max_cutoff=3200,slices='all',nthreads=4,mem=3.75,SMTH_DEN=None,IN_MEM=True,SUBMIT=False,CLOBBER=False):
    """
    Creates .py and .sub submission files for submission of DKE to SGE, submits if SUBMIT=True
    Pass matched lists of data filenames, bval filenames, and bvec filenames, along with a root directory for the output
    NOTE: CLOBBER and SUBMIT interact, CLOBBER and SUBMIT both need to be true to submit to SGE when the .py file already exists in the output dir
    INPUT:
        - data_fnames       list of diffusion data files
        - bvals_fnames      list of bvals files
        - bvecs_fnames      list of bvecs files
        - out_root_dir      root directory of output (see below for subdir name)
        - IDs               list of IDs that are used to create subdir names and check that the correct files are being used
                            IDs are used as the lookup to confirm full filenames of data, bvals, bvecs - so they must be unique (order does not matter, full search)
                            only IDs included in this list will be processed (even if they are only a subset of the data_fnames etc)
        - TractREC_path     path to TractREC code where the preprocessing scripts are held, needed for qsub submission
        - bval_max_cutoff   bval cutoff value for selection of vols to be included in DKE - filenames are derived from this
        - slices            list of slice indices to process, or 'all' XXX THIS ONLY WORKS FOR ALL NOW
        - SMTH_DEN          list to select smooth, denoise, or not ['smth','nlmeans',''], may run out of memory with large datasets (i.e., HCP)
        - IN_MEM            perform diffusion volume selection (based on bvals that were selected by bval_max_cutoff) in mem or with fslselectcols via command line
        - SUBMIT            submit to SGE (False=just create the .py and .sub submission files)
        - CLOBBER           force overwrite of output files (.py and .sub files are always overwritten regardless)
        
    RETURNS: 
        - nothing, but dumps all DKE calcs (MK, RK, AK) in out_dir/ID
    """
    import os
    
    caller_path=os.path.dirname(os.path.abspath(__file__)) #path to this script, so we can add it to a sys.addpath statement
    print("Running the dipy-based diffusion kurtosis estimator.")
    for idx,ID in enumerate(IDs):
        fname=[s for s in data_fnames if ID in s] #we use the IDs as our master to lookup files in the provided lists, the full filename should have the ID SOMEWHERE!
        bvals=[s for s in bvals_fnames if ID in s]
        bvecs=[s for s in bvecs_fnames if ID in s]
        out_dir=os.path.join(out_root_dir,ID)
        create_dir(out_dir)
        #check that we are pulling the correct files
        if len(fname)>1 or len(bvals)>1 or len(bvecs)>1:
            print "OH SHIT, too many possible files. This should not happen!"
            DATA_EXISTS=False
        elif len(fname)<1 or len(bvals)<1 or len(bvecs)<1:
            print "OH SHIT, no matching file for this ID: " + ID
            DATA_EXISTS=False
        else:
            fname=fname[0] #break it out of the list of one  
            bvals=bvals[0]
            bvecs=bvecs[0]
            DATA_EXISTS=True
            print(ID)
            print(" input:  "+ (fname))
            print(" input:  "+ (bvals))
            print(" input:  "+ (bvecs))
            print(" output: "+ (out_dir))
        
        if DATA_EXISTS:
            code=["#!/usr/bin/python","","import sys","sys.path.append('{0}')".format(caller_path),"import preprocessing as pr"]
            code.append("pr.DKE('{data_fname}','{bvals_fname}','{bvecs_fname}',bval_max_cutoff={bval_max_cutoff},out_dir='{out_dir}',slices='{slices}',SMTH_DEN={SMTH_DEN},IN_MEM={IN_MEM})""".format(data_fname=fname,bvals_fname=bvals,bvecs_fname=bvecs,\
                bval_max_cutoff=bval_max_cutoff,out_dir=out_dir,slices=slices,SMTH_DEN=SMTH_DEN,IN_MEM=IN_MEM))
            py_sub_full_fname=create_python_exec(out_dir=out_dir,code=code,name='DKE_'+ID)
            
            #XXX this check condition currently does nothing, because the .py file is always created. Dumb.
            if CLOBBER:
                print("Creating submission files and following your instructions for submission to que. (CLOBBER=True)"),
                print(" (SUBMIT=" + str(SUBMIT)+")")
                submit_via_qsub(template_text=None,code="python " + py_sub_full_fname,name='DKE_'+ID,nthreads=nthreads,mem=mem,outdir=out_dir,\
                                description="Diffusion kurtosis estimation with dipy",SUBMIT=SUBMIT)
            else:
                print("Creating submission files and following your instructions for submission to que. (CLOBBER=False)")
                print(" (SUBMIT=" + str(SUBMIT)+")")
                submit_via_qsub(template_text=None,code="python " + py_sub_full_fname,name='DKE_'+ID,nthreads=nthreads,mem=mem,outdir=out_dir,\
                                description="Diffusion kurtosis estimation with dipy",SUBMIT=SUBMIT)
        print("")

def run_amico_noddi_dipy(subject_root_dir,out_root_dir,subject_dirs=None,b0_thr=0, bStep=[0,1000,2000,3000],nthreads=8,mem=2.5,CLOBBER=False,SUBMIT=False):
    #No... requires closer to 36GB for the HCP data
    #when requesting cores, select 24 and take the whole memory (time it...)
    #currently requires the compiled version of spams that I have installed locally    
    import os
    import sys
    from TractREC import create_dir
    spams_path='/home/cic/stechr/Documents/code/spams-python'
    import spams #this is added here so that the requirements.txt is updated
    import amico #this is added here so that the requirements.txt is updated
    #working_amico_path='/home/cic/stechr/Documents/code/amico_cjs/AMICO/python/amico'
    caller_path=os.path.dirname(os.path.abspath(__file__)) #path to this script, so we can add it to a sys.addpath statement
    #caller_path="caller_path_test"
    #spams required by amico, along with specific numpy version (1.10 has a fortran issue that crops up here)
    sys.path.append(spams_path) #EW! TODO: make me permanent at some point...
    
    if subject_dirs is None:
        #subject_dirs=['100307'] #TODO, create a list of subjects from the subject_root_dir
        subject_dirs=os.listdir(subject_root_dir)
        if "kernels" in subject_dirs: subject_dirs.remove("kernels") #don't try to do this for the kernels directory, which AMICO hard-codes here
    
    #amico.core.setup()
    for ID in subject_dirs:    #ID is the subdirectory off of subject_root_dir that contains each subject
        
        """
        ae=amico.Evaluation(subject_root_dir,ID)
        amico.util.fsl2scheme(bvals_fname, bvecs_fname,scheme_fname,bStep)
        #ae.load_data(dwi_filename=dwi_fname,scheme_filename=scheme_fname,mask_filename=mask_fname,b0_thr=bo_thresh) #loading takes tabout 4-5 mins (HCP), and about 8gb
        ae.load_data(dwi_filename='data.nii.gz',scheme_filename='sanitised.scheme',mask_filename='nodif_brain_mask.nii.gz',b0_thr=0)
        ae.set_model("NODDI")
        ae.generate_kernels() #single core only, 2.5mins (HCP), apparently only needs to be done once if all data was acquired the same way (loading fits to the bvecs) - not impractical to do it every time, since I don't see a way to save it
        ae.load_kernels() #resamples the LUT for this subject - looks to use a lot of mem...and will use available cores - 2.5mins on 8
        ae.fit() #numpy version 1.10.0 will not work, uses all cores and about 8GB for HCP data
        ae.save_results()
        #move to proper output directory so that we keep with our processing stream
        """
        dwi_fname=os.path.join(subject_root_dir,ID,"data.nii.gz")
        bvals_fname=os.path.join(subject_root_dir,ID,"bvals")
        bvecs_fname=os.path.join(subject_root_dir,ID,"bvecs")
        scheme_fname=os.path.join(subject_root_dir,ID,"bvals_bvecs_sanitised.scheme")
        mask_fname=os.path.join(subject_root_dir,ID,"nodif_brain_mask.nii.gz")
        out_dir=os.path.join(out_root_dir,ID)
        create_dir(out_dir)
        
        #bStep=[0,1000,2000,3000]
        #b0_thr=0
        model="NODDI"

        code=["#!/usr/bin/python","","import sys","sys.path.append('{0}')".format(caller_path),"sys.path.append('{0}')".format(spams_path),"import spams","import amico"]
        code.append("import spams" )
        code.append("")
        code.append("print amico.__file__")
        code.append("amico.core.setup()")
        code.append("ae=amico.Evaluation('{subject_root_dir}','{ID}',output_path='{output_dir}')".format(subject_root_dir=subject_root_dir,ID=ID,output_dir=out_dir))
        code.append("amico.util.fsl2scheme('{bvals_fname}', '{bvecs_fname}','{scheme_fname}',{bStep})".format(bvals_fname=bvals_fname, bvecs_fname=bvecs_fname,scheme_fname=scheme_fname,bStep=bStep))
        code.append("ae.load_data(dwi_filename='{dwi_fname}',scheme_filename='{scheme_fname}',mask_filename='{mask_fname}',b0_thr={b0_thr})".format(dwi_fname=dwi_fname,scheme_fname=scheme_fname,mask_fname=mask_fname,b0_thr=b0_thr))
        code.append("ae.set_model('{model}')".format(model=model))
        code.append("ae.set_config('OUTPUT_path','{OUTPUT_path}')".format(OUTPUT_path=out_dir))        
        code.append("ae.generate_kernels()") #single core only, 2.5mins (HCP), apparently only needs to be done once if all data was acquired the same way (loading fits to the bvecs) - not impractical to do it every time, since I don't see a way to save it
        code.append("ae.load_kernels()") #resamples the LUT for this subject - looks to use a lot of mem...and will use available cores - 2.5mins on 8
        code.append("ae.fit()") #fit the model
        code.append("ae.save_results()")

        #return code
        py_sub_full_fname=create_python_exec(out_dir=out_dir,code=code,name='NOD_'+ID)
        if CLOBBER:
            print("Creating submission files and following your instructions for submission to que. (CLOBBER=True)"),
            print(" (SUBMIT=" + str(SUBMIT)+")")
            submit_via_qsub(template_text=None,code="python " + py_sub_full_fname,name='NOD_'+ID,nthreads=nthreads,mem=mem,outdir=out_dir,
                            description="NODDI estimation with AMICO",SUBMIT=SUBMIT)
        else:
            print("Creating submission files and following your instructions for submission to que. (CLOBBER=False)")
            print(" (SUBMIT=" + str(SUBMIT)+")")
            submit_via_qsub(template_text=None,code="python " + py_sub_full_fname,name='NOD_'+ID,nthreads=nthreads,mem=mem,outdir=out_dir,
                            description="NODDI estimation with AMICO",SUBMIT=SUBMIT)
        print(py_sub_full_fname)

def interp_discrete_hist_peaks(input_fname, output_fname=None, value_count_cutoff=50):
    """
    Detects peaks in histogram of input image where the number of voxels is greater than value_count_cutoff, brings in the 
    value from a 3d gaussian filtered version of the data (sigma=1), and writes the file with extension _smth_peaks.nii.gz

    :param input_fname: full path and fname to input image file
    :param output_fname: full path and fname to output image file, None for input location and _smooth_peaks.nii.gz
    :param value_count_cutoff: voxel count, where greater than this is considered a peak
    :return:
    """
    from scipy import ndimage, interpolate
    import scipy.ndimage.filters as filt
    import nibabel as nb
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    sigma=1
    
    if output_fname is None:
        output_fname = os.path.join(os.path.dirname(input_fname),os.path.basename(input_fname).split(".")[0]+
                                    "_smth_peaks.nii.gz")

    img = nb.load(input_fname)
    d3d = img.get_data()
    d3d_smth = filt.gaussian_filter(d3d,sigma) #3d gaussian filter to blur the data
    d = np.ravel(d3d)

    hist, bin_edges = np.histogram(d,bins=len(np.unique(d))) #histogram for each value
    plt.figure(), plt.plot(hist[1:])  # cut at least the first, maybe the last?, since they represent 0 and 1 and have a lot of vals

    #define the bin edges (uncessary because we should have one value per bin...
    left = bin_edges[np.where(hist > value_count_cutoff)][1:] #skip zero
    right = bin_edges[np.add(np.where(hist > value_count_cutoff), 1)][0][1:] #skip zero

    for idx, mybin in enumerate(left):
        #vox_vals = np.unique(d[np.logical_and(d >= left[idx], d <= right[idx])])
        print mybin
        # should only be one value, but loop over it just in case...
        # loop across vox_vals
        #for vox_val in vox_vals:
        d3d[d3d == mybin]=np.nan
        # TODO: could create the neighbourhood by hand and calc the fit here?

    valid_mask = ~np.isnan(d3d)
    coords = np.array(np.nonzero(valid_mask)).T
    values = d3d[valid_mask]
    it = interpolate.LinearNDInterpolator(coords,values,fill_value=np.nan)
    filled = it(list(np.ndindex(d3d.shape))).reshape(d3d.shape)

    #d3d_out = ndimage.convolve(d3d,structure)
    # calculate the mean of all non-equal neighbouring voxels to re-calculate this mean
    # what do I do when the neighbour is the same as vox_val?

    return d,hist,bin_edges,filled


# XXX stuff for testing XXX
#DKE('/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/data.nii.gz','/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvals',\
#    '/data/chamal/projects/steele/working/HCP_CB_DWI/source/dwi/100307/bvecs',\
#    out_dir='/data/chamal/projects/steele/working/HCP_CB_DWI/processing/DKI/100307_dipy_3K_new',slices='all',SMTH_DEN="smth",IN_MEM=True)

#import os
#print os.path.realpath(__file__)

"""
import sys
#spams required by amico, along with specific numpy version (1.10 has a fortran issue that crops up here)
sys.path.append('/home/cic/stechr/Documents/code/spams-python') #EW! TODO: make me permanent at some point...
import spams
import amico
ae=amico.Evaluation('/data/chamal/projects/steele/working/HCP_CB_DWI/source/HCP_S900/diffusion','100307')
amico.util.fsl2scheme("bvals", "bvecs","bvals_bvecs_sanitised.scheme",[0,1000,2000,3000])
ae.load_data(dwi_filename='data_2slices.nii.gz',scheme_filename='bvals_bvecs_sanitised.scheme',mask_filename='nodif_brain_mask.nii.gz',b0_thr=0) #may run out of mem in the 32bit conversion
ae.set_model("NODDI")
ae.set_config('OUTPUT_path','/data/chamal/projects/steele/working/NODDI_test/AMICO_test2')
ae.generate_kernels() #single core only, 2.5mins (HCP), apparently only needs to be done once if all data was acquired the same way (loading fits to the bvecs) - not impractical to do it every time, since I don't see a way to save it
ae.load_kernels() #resamples the LUT for this subject - looks to use a lot of mem...and will use available cores (though not fully) - 2.5mins
ae.fit()
ae.save_results()
check this job id: 459901
"""
