from MCCA import MCCA
from scree_plot import select_all_subs
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import glob
import pandas as pd


def plot_inter_sub_pca(subjects, n_pc, path):
    """
        The function is made to fit and calculate the MCCA function on multiple subjects. After the fitting of the model,
        it calculates inter subject correlation of PCA components and CCA components, and visualize them as figures.

         Parameters:
            subjects (list):
                this is a list of all subjects, each subject is a tuple. Each tuple should contain two elements, the
                first it's the ID of the subject as a string, and the second it's the timeseries of that subject
                (samples x features) as a ndarray.
            n_pc (int):
                the number of principal components to retain.
            path (str):
                the output directory where the figures are going to be stored.
        """
    # transform our information so that it can be read by the function MCCA, we need an array
    # (subjects x samples x features), we need to unpack the tuple and concatenate
    just_timeseries = [one_sub[1] for one_sub in subjects]
    timeseries = np.stack(just_timeseries, axis=0)
    # define and fit our MCCA
    mcca = MCCA(n_pc, pca_only=True)
    mcca.obtain_mcca(timeseries)
    t_mu = mcca.mu[:, np.newaxis]
    t_sigma = mcca.sigma[:, np.newaxis]
    timeseries -= t_mu  # centered
    pca_s = np.matmul(timeseries, mcca.weights_pca)  # PCAs obtained
    pca_s /= t_sigma  # normalized
    PC = mcca.weights_pca.shape[2]
    correlations_pc = []
    # iterate over PC
    for j in range(PC):
        corr_matrix = np.corrcoef(np.squeeze(pca_s[:, :, j]))
        corr_idx = np.triu_indices(corr_matrix.shape[0], k=1)
        inter_subject_corr_pc = corr_matrix[corr_idx]
        # append inter-subject correlations
        correlations_pc.append(inter_subject_corr_pc)
    # create the path to store the figure
    fig_name_pc = (path + '/task-aomovie_acq-CS_inter_subject_corr_' + str(n_pc) + '_pc.svg')
    plt.figure(figsize=(16, 10), dpi=600)
    ax_pc = sns.boxplot(data=correlations_pc, palette='flare')
    ax_pc.set_ylim(-1.1, 1.1)
    ax_pc.set_ylabel('Pearson Correlation')
    ax_pc.set_xlabel('Principal component index')
    ax_pc.set_title(f'Inter subject correlation for {n_pc} \n principal component scores')
    plt.savefig(fig_name_pc, format='svg', dpi=400)
    plt.close()


def plot_inter_sub_cc(train, test, n_pc, *n_cc, path):
    """
        The function is made to fit and calculate the MCCA function on multiple subjects. After the fitting of the model,
        on a train set it fits the model on a test set, and then it calculates inter subject correlation of  CCA
        components, and visualize them as figures.

         Parameters:
            train (ndarray):
                an ndarray of shape (subjects, samples, features).
            test (ndarray):
                a ndarray of shape (subjects, samples, features).
            n_pc (int):
                the number of principal components to retain.
            *n_cc (int):
                the number of canonical components to retain, if not specified it will be the same as n_pc.
            path (str):
                the output directory where the figures are going to be stored.
        """
    if n_cc:
        cc = n_cc[0]
    else:
        cc = n_pc
    # use the functions created in MCCA code to calculate the correlations
    mcca = MCCA(n_pc, cc)
    mcca.obtain_mcca(train)
    correlations = mcca.test_mcca(train, test)
    # create the figure and its path
    fig_name_cc = open((path + '/task-aomovie_acq-CS_inter_subject_corr_' + str(cc) + '_cc.svg'), 'wb')
    fig_cc = plt.figure(figsize=(16, 10), dpi=400)
    ax_cc = fig_cc.add_subplot(111)
    ax_cc.plot(correlations[:, 0], color='orange')
    ax_cc.plot(correlations[:, 1], color='green')
    ax_cc.set_ylim(-0.1, 1.1)
    ax_cc.set_ylabel('Averaged inter-subject correlations')
    ax_cc.set_xlabel('Canonical component index')
    ax_cc.legend(['Train Data', 'Test Data'])
    ax_cc.set_title(f'Average inter subject correlation for {cc} \n canonical components')
    fig_cc.tight_layout()
    plt.savefig(fig_name_cc, format='svg', dpi=400)
    plt.close()


def split_tt_data(path, data, subjects):
    """
    Function to separate the data into a test and train set, based on the runs of the experiment. It will select the
    first 3 runs as the train set and the last 3 as the test set. This function is based on the information and format
    of the functions concatenate_sub and check_alignment from the script MCCA_fMRI_fit.py.

     Parameters:
        path (str):
            this is the folder directory in which the subjects data is stored, following the BIDS convention
            the path should then be followed by the type of data to be extracted. In this folder there should be
            a tsv and a pkl file, one stores the structure of the data and one the data.
        data (str):
            this is the data we are extracting, as a string.
        subjects (list):
            this is a list of all subjects, each subject is a tuple. Each tuple should contain two elements, the
            first it's the ID of the subject as a string, and the second it's the timeseries of that subject
            (samples x features) as a ndarray. All subjects should have the same values for their shape.
    Returns:
        train_data (ndarray):
            an array of the train data, has the shape (subjects x samples x features).
        test_data (ndarray):
            an array of the test data, has the shape (subjects x samples x features).
        """
    full_path = os.path.join(path, data)
    # extract the tsv file in the folder and save it as a data frame
    tsv_files = glob.glob((full_path + '/*.tsv'))
    # this data frame should have the headers: Sub, Run_Length, Voxel_Size, and Run_ID
    df = pd.read_csv(tsv_files[0])
    # because we are using subjects that all have the same length in total, and of each individual run we can use the
    # same information for all so we just select the ID of one subject
    sub_id = subjects[0][0]
    # extract the information of one subject
    sub_df = df[df['Sub'] == sub_id]
    # get the values for the number of samples in the first 3 runs and the number of samples the last 3 runs
    first_runs = sum(sub_df['Run_Length'].to_list()[0:3])
    # transform our information we need an array, we need to unpack the tuple and concatenate
    just_timeseries = [one_sub[1] for one_sub in subjects]
    timeseries = np.stack(just_timeseries, axis=0)
    # cut our data into train and test
    train_data = timeseries[:, 0:first_runs, :]
    test_data = timeseries[:, first_runs-1:-1, :]
    return train_data, test_data


if __name__ == "__main__":
    input_root = '/data2/jpanzay1/thesis_bids/derivatives/encoding_results_files/'
    output_root = '/data2/jpanzay1/thesis_bids/derivatives/mcca_results/'
    mask = 'gray_matter_mask'
    feature = 'lagging2to-10.2tag25ms'
    stimuli = 'stimuli_alwaysCS'
    subs = 'all_subs'
    acquired = 'acq-CS'
    # define our input directory, where our folder has the data
    input_dir = os.path.join(input_root, mask, feature, stimuli, subs, acquired)
    # extract only the timeseries of n subjects, that all have the same length
    subs_timeseries = select_all_subs(input_dir, 'predicted_bold')
    # split data
    train_data, test_data = split_tt_data(input_dir, 'predicted_bold', subs_timeseries)
    # define the directory where our PCA analysis will be stored
    output_dir = os.path.join(output_root, mask, feature, stimuli, subs, acquired, 'mcca')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create a figure for our inter subject correlations
    plot_inter_sub_cc(train_data, test_data, 140, 140, path=output_dir)
