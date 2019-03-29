import numpy as np 
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask
from sklearn import preprocessing
from scipy import stats


def get_bold_for_condition(dir_input,num_runs,option_zscore=0):
    """" This function extracts the bold signal for three conditions. 
    option_zscore = 0 => no z-scoring
    option_zscore = 1 =>z-score the data
    
    Returns: bold values for all conditions for each run.
    A mean value for the entire run.
    """
    from utils import shift_timing, mask_data, scale_data
    #Initialize arrays
    stim_label=[]
    bold_A=[]
    bold_B=[]
    bold_C=[]
    bold_fix=[]
    bold_mean_all=[]
    TR_shift_size=2 # Number of TRs to shift the extraction of the BOLD signal.

    maskdir = dir_input
    masks = ['ROI_Cool']


    ### Extract the BOLD Signal for the conditions A, B, C
    ###

    print ("Processing Start ...")
    maskfile = (maskdir + "%s.nii.gz" % (masks[0]))
    mask = nib.load(maskfile)
    print ("Loaded Mask")
    print(mask.shape)

    for run in range(1,num_runs+1):
        epi_in = (dir_input +"lab1_r0%s.nii.gz" % ( run))
        stim_label=np.load(dir_input + 'labels_r0%s.npy' % (run))

        # Haemodynamic shift
        label_TR_shifted = shift_timing(stim_label, TR_shift_size)

        # Get labels for conditions for A, B, C, and baseline fixation.
        A = np.squeeze(np.argwhere(label_TR_shifted==1))
        B = np.squeeze(np.argwhere(label_TR_shifted==2))
        C = np.squeeze(np.argwhere(label_TR_shifted==3))

        fixation= np.squeeze(np.argwhere(label_TR_shifted==0))
        epi_data = nib.load(epi_in)
        epi_mask_data = mask_data(epi_data, mask)

        if option_zscore==1:
            epi_maskdata_zscore = scale_data(epi_mask_data)
            epi_mask_data = epi_maskdata_zscore

        if run==1:
                bold_A=epi_mask_data[A]
                bold_B=epi_mask_data[B]
                bold_C=epi_mask_data[C]
                bold_fix=epi_mask_data[fixation]
                bold_data_all= epi_mask_data
        else:
                bold_A=np.vstack([bold_A,epi_mask_data[A]])
                bold_B= np.vstack([ bold_B,epi_mask_data[B]])
                bold_C= np.vstack([bold_C,epi_mask_data[C]])
                bold_fix= np.vstack([bold_fix,epi_mask_data[fixation]])
                bold_data_all= np.vstack([bold_data_all,epi_mask_data])
        bold_mean_all.append(np.mean(epi_mask_data))
    print("Processing Completed")
    return bold_data_all, bold_mean_all,bold_A, bold_B, bold_C, bold_fix, label_TR_shifted

def compute_mean_diff(num_runs,bold_A, bold_B, bold_C, bold_fix):

    """" This function computes the mean effect for conditions A, B, and C for each run.
    It takes the difference between the signal for the condition and the baseline.
    Inputs: num_runs, bold_A, bold_B, bold_C, bold_fix
    
    Returns: diff """
    mean_diff_A=np.zeros(num_runs)
    mean_diff_B=np.zeros(num_runs)
    mean_diff_C=np.zeros(num_runs)
    
    num_rows_per_run = int(bold_A.shape[0]/num_runs)
    num_rows_per_run_fix = int(bold_fix.shape[0]/num_runs)
    
    for run in range(num_runs):

        row_start=0 + num_rows_per_run*(run)
        row_end=row_start+num_rows_per_run
        row_start_fix=0 + num_rows_per_run_fix *run
        row_end_fix=row_start_fix + num_rows_per_run_fix 
        mean_diff_A[run]=np.mean(bold_A[row_start:row_end,:]) - np.mean(bold_fix[row_start_fix:row_end_fix,:])
        mean_diff_B[run]=np.mean(bold_B[row_start:row_end,:]) - np.mean(bold_fix[row_start_fix:row_end_fix,:])
        mean_diff_C[run]=np.mean(bold_C[row_start:row_end,:]) - np.mean(bold_fix[row_start_fix:row_end_fix,:])

    diff = [mean_diff_A, mean_diff_B, mean_diff_C]
    return diff

def compute_stats(data,confidence=0.95):
    """" This function computes the mean, standard errror, 
    and confidence intervals  using scipy."""
    
    n = len(data) # sample size
    m = np.mean(data, axis=1) # mean
    std_err = stats.sem(data, axis=1) # standard error
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1) #width of the confidence interval
    interval = [m-h, m+h] # confidence interval
    
    return m, std_err, h, interval