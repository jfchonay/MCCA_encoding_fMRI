import joblib
import numpy as np
import os
import glob
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def select_subs(path, data, n_subs):
    """
    The function is made to extract the predicted BOLD data for n random subjects. It first will create a list of all
    subjects that have the same number of samples for the BOLD data, and then it will randomly select n subjects from
    that list and return their subject ID and timeseries. This function is based on the information and format of the
    functions concatenate_sub and check_alignment from the script MCCA_fMRI_fit.py.

     Parameters:
        path (str):
            this is the folder directory in which the subjects data is stored, following the BIDS convention
            the path should then be followed by the type of data to be extracted, in our case is the predicted bold. In
            this folder there should be a tsv and a pkl file, one stores the structure of the data and one the
            data.
        data (str):
            this is the data we are extracting, as a string.
        n_subs (int):
            amount of subjects we want to extract for further analysis, as an integer.
    Returns:
        subs_timeseries (list):
            this is a list of tuples, that contains the subject's ID and the concatenated array of the predicted BOLD
            data. This should be as long as the amount of subjects we defined and all of them should have the same
            length.
    """
    full_path = os.path.join(path, data)
    # extract the tsv file in the folder and save it as a data frame
    tsv_files = glob.glob((full_path + '/*.tsv'))
    # this data frame should have the headers: Sub, Run_Length, Voxel_Size, and Run_ID
    df = pd.read_csv(tsv_files[0])
    # group the data frame by the sum of the length of all the runs for one subject
    run_sum = df.groupby('Sub')['Run_Length'].sum().reset_index()
    # organize this new data frame to have a list of all the subjects who have the same value for the sum of the
    # length of runs
    group_length = run_sum.groupby('Run_Length')['Sub'].apply(list).reset_index()
    # sort this by the value of the sum which has more subjects
    group_length = group_length.sort_values(by='Sub', ascending=False).reset_index()
    # extrac the list of subjects that belong to the sum of all runs, which has more subjects
    list_subs = group_length['Sub'][0]
    # now we will randomly select 4 subjects to do the PCA analysis on
    rand_subs = random.sample(list_subs, n_subs)
    # open our pkl file where the full timeseries of all our subjects is
    pkl_files = glob.glob((full_path + '/*.pkl'))
    with open(pkl_files[0], 'rb') as file:
        all_subs = joblib.load(file)
    # our all_subs is a list that contains a tuple, with the subject ID and a ndarray of data, we want to extract only
    # ndarray of the subjects whose ID matches our random sample
    subs_timeseries = []
    for randi in rand_subs:
        for one_sub in all_subs:
            if randi == one_sub[0]:
                subs_timeseries.append((one_sub[0], one_sub[1]))
                break
    return subs_timeseries


def scree_plot(timeseries, output_dir, *pc):
    """
    The function is made to run a principal component analysis (PCA) on the timeseries data for one subject, calculate
    explained variance ratio for each component and create a scree plot. The number of components can be defined or it
    is going to be the highest number of meaningful components (n samples - 1).

     Parameters:
        timeseries (tuple):
            this tuple should contain two elements, the first one it's the ID of the subject as a string, and the second
            it's the timeseries of that subject (samples x features) as a ndarray.
        output_dir (str):
            the output directory where the figures are going to be stored.
        *pc (int):
            the number of principal components to be calculated, this is an optional argument.
    """
    # assign the values that are contained inside the tuple
    s_id = timeseries[0]
    x = timeseries[1]
    n_samples = x.shape[0]
    n_features = x.shape[1]
    # before calculating PCA we should standardize our data, subtracting the mean and dividing by the standard
    # deviation, this is done in the direction of each feature
    scaler = StandardScaler()
    scaler.fit(x)
    x_sc = scaler.transform(x)
    # We want to use the number of components we defined as a variable input, but if there is nothing defined
    # we want to create a scree plot we know that the maximum number of components with a meaningful solution is samples
    # - 1, so we need to determine first if we have a greater number of samples or features.
    if not pc:
        if n_features > n_samples:
            max_pc = (n_samples - 1)
        else:
            max_pc = n_features
    else:
        max_pc = pc[0]
    # log_vector = np.logspace(start=1, stop=max_pc, num=3, dtype='int').tolist()
    # define the pca model for the number of components used and fit it
    pca = PCA(n_components=max_pc, svd_solver='full')
    pca.fit(x_sc)
    # extract the explained variance ratio and calculate the cumulative sum of the ratio
    var_ratio = pca.explained_variance_ratio_
    sum_vr = np.cumsum(var_ratio)
    # plot the information and save the figure
    fig_name = open((output_dir + '/' + s_id + '_task-aomovie_acq-CS_scree-plot_' + str(max_pc) + '-pc.svg'), 'wb')
    fig = plt.figure(figsize=(16, 10), dpi=400)
    ax = fig.add_subplot(111)
    ax.bar(range(0, len(var_ratio)), var_ratio, align='center', color='pink',
           label='Individual explained variance')
    ax.step(range(0, len(sum_vr)), sum_vr, where='mid', color='seagreen',
            label='Cumulative explained variance')
    ax.axhline(0.8, 0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Explained variance ratio')
    ax.set_xlabel('Principal component index')
    ax.legend(loc='best')
    ax.set_title(f'Scree plot of the explained variance for {max_pc}'
                 f'\n principal components for subject {s_id}')
    fig.tight_layout()
    plt.savefig(fig_name, format='svg', dpi=400)
    plt.close()


def select_all_subs(path, data):
    """
    The function is made to extract the predicted BOLD data for subjects that have the same number of samples.
    It first will create a list of all subjects that have the same number of samples for the BOLD data, and return
    their subject ID and timeseries. This function is based on the information and format of the functions
    concatenate_sub and check_alignment from the script MCCA_fMRI_fit.py.

     Parameters:
        path (str):
            this is the folder directory in which the subjects data is stored, following the BIDS convention
            the path should then be followed by the type of data to be extracted. In this folder there should be
            a tsv and a pkl file, one stores the structure of the data and one the data.
        data (str):
            this is the data we are extracting, as a string.
    Returns:
        subs_timeseries (list):
            this is a list of tuples, that contains the subject's ID and the concatenated array of the predicted BOLD
            data or the data specified, all the timeseries should be the same length.
    """
    full_path = os.path.join(path, data)
    # extract the tsv file in the folder and save it as a data frame
    tsv_files = glob.glob((full_path + '/*.tsv'))
    # this data frame should have the headers: Sub, Run_Length, Voxel_Size, and Run_ID
    df = pd.read_csv(tsv_files[0])
    # group the data frame by the sum of the length of all the runs for one subject
    run_sum = df.groupby('Sub')['Run_Length'].sum().reset_index()
    # organize this new data frame to have a list of all the subjects who have the same value for the sum of the
    # length of runs
    group_length = run_sum.groupby('Run_Length')['Sub'].apply(list).reset_index()
    # sort this by the value of the sum which has more subjects
    group_length = group_length.sort_values(by='Sub', ascending=False).reset_index()
    # extrac the list of subjects that belong to the sum of all runs, which has more subjects
    list_subs = np.sort(group_length['Sub'][0]).tolist()
    # open our pkl file where the full timeseries of all our subjects is
    pkl_files = glob.glob((full_path + '/*.pkl'))
    with open(pkl_files[0], 'rb') as file:
        all_subs = joblib.load(file)
    # our all_subs is a list that contains a tuple, with the subject ID and a ndarray of data, we want to extract only
    # ndarray of the subjects who have the same number of samples
    all_timeseries = []
    for sort_sub in list_subs:
        for one_sub in all_subs:
            if sort_sub == one_sub[0]:
                all_timeseries.append((one_sub[0], one_sub[1]))
                break
    return all_timeseries


def sum_variance(subjects, n_pc, output_dir):
    """
    The function is made to run a principal component analysis (PCA) on the timeseries data for multiple subjects, it
    will calculate the amount of variance that is retained in every number of components specified. It will create a tsv
    file with the information and one figure, and store them in the output direction we specified.

     Parameters:
        subjects (list):
            this is a list, each element of the list is one a tuple that has the information of one subject. This tuple
            should contain two elements, the first one it's the ID of the subject as a string, and the second
            it's the timeseries of that subject (samples x features) as a ndarray.
        n_pc (list):
            this is a list with the number of principal components to be calculated, they should be an integer.
        output_dir (str):
            the output directory where the figures and tsv files are going to be stored.
    """
    # we want to construct a data frame, in which we will store the amount of variance explained by the number of pc
    columns = ['Subject', 'Sum_Var', 'n_PC']
    df = pd.DataFrame(columns=columns)
    # iterate over all subjects
    for timeseries in subjects:
        # assign the values that are contained inside the tuple
        s_id = timeseries[0]
        x = timeseries[1]
        # before calculating PCA we should standardize our data, subtracting the mean and dividing by the standard
        # deviation, this is done in the direction of each feature
        scaler = StandardScaler()
        scaler.fit(x)
        x_sc = scaler.transform(x)
        # define the pca model for the number of components used and fit it
        pca = PCA(svd_solver='full')
        pca.fit(x_sc)
        # extract the explained variance ratio and calculate the sum of the variance
        var_ratio = pca.explained_variance_ratio_
        for pc in n_pc:
            sum_vr = np.sum(var_ratio[0:pc-1])
            # create a new data frame and append to the global data frame
            df_pc = pd.DataFrame([[s_id, sum_vr, pc]], columns=columns)
            df = pd.concat((df, df_pc))
    # store our data frame as a csv
    df.to_csv((output_dir + '/all_subs_task-aomovie_acq-CS_pca_variance.tsv'), index=False)
    # create our bar plot
    plot_name = (output_dir + '/all_subs_task-aomovie_acqs-CS_pca_variance.svg')
    plt.figure(figsize=(16, 10), dpi=400)
    ax = sns.barplot(x='Subject', y='Sum_Var', hue='n_PC', data=df, errorbar=None, palette='husl')
    ax.set_title('Cumulative variance explained for each subject \n for each number of principal components retained')
    ax.set_xlabel('Subject ID')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_ylim(0, 1)
    plt.savefig(plot_name, format='svg', dpi=400)
    plt.close()


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
    # subs_timeseries = select_subs(input_dir, 'predicted_bold', 3)
    # define the directory where our PCA analysis will be stored
    output_dir = os.path.join(output_root, mask, feature, stimuli, subs, acquired, 'pca')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # perform PCA on our subjects and store the scree plots
    # for one_sub in subs_timeseries:
    #     scree_plot(one_sub, output_dir)
    # extract all the subjects that have the same length or number of samples
    all_subs = select_all_subs(input_dir, 'predicted_bold')
    # run a PCA analysis with a determined number of principal components, then calculate how much of the variance
    # is explained in each subject
    n_pc = [70, 100, 130]
    sum_variance(all_subs, n_pc, output_dir)
