import joblib
import numpy as np
import glob
import pandas as pd
import os


def concatenate_sub(folder, s_id):
    """
    The function is made to extract the predicted BOLD data for each run in each subject and store
     the values and the shape of those values.

     Parameters:
        folder (str):
            this is the folder directory in which the subjects data is stored, following the BIDS convention
            the path should then be followed by the task or acquired condition, and then by the type of data. The data stored
            inside  these folder should be as a pickle file, that contains an array of shape (samples, voxels) as a result of
            the encoding model.
        s_id (str):
            this is the subject's ID, should be a string.
    Returns:
        one sub (list):
            this is a list of tuples, that contains the subject's ID and the concatenated array of the predicted BOLD
            data. The array should have the length of the full run and the number of voxels.
        shape_run (list):
            this is a list of tuples, they contain the information for all the runs for one subject. They are organized
            in first the subject ID, the length or number of samples of each run, the amount of voxels for each run, and
            the ID of the run.
    """
    predicted_bold_sub = []
    shape_run = []
    # we want to iterate over all pkl files inside our folder, in the same order starting from 0 to 5
    pkl_filenames = np.sort(glob.glob((folder+'/*.pkl')))
    for idx, pkl_files in enumerate(pkl_filenames):
        run_id = ('run_' + pkl_files[-5:-4])
        with open(pkl_files, 'rb') as pkl:
            predicted_bold_run = joblib.load(pkl)
            # get the shape of the data, we can use it to check the alignment later
            run_len = predicted_bold_run.shape[0]
            voxel_num = predicted_bold_run.shape[1]
            # save the shape of the run and the subject id in a list
            shape_run.append((s_id, run_len, voxel_num, run_id))
            # save the predicted data for all the runs in one subject
            predicted_bold_sub.append(predicted_bold_run)
    one_sub = (s_id, np.concatenate(predicted_bold_sub, axis=0))
    return one_sub, shape_run


def check_alignment(shape, output_dir):
    """
     The function is made to organize the information of all the subjects into a data frame, and then save it as a
     tsv file.

      Parameters:
         shape (list):
             it should be a list, that is the length of all the subjects, inside the list each subject should contain a
             list of tuples that represent the subject ID, the length or number of samples of the run, the amount of
             voxels for each run, and the run ID.
         output_dir (str):
            the path were we want to save our tsv file.
     Returns:
         df (df):
             a data frame, that organizes the information of the list, it will contain the keys Sub, Run_Length,
             Voxel_Size, and Run_ID.
     """
    # define the header of our data frame
    columns = ['Sub', 'Run_Length', 'Voxel_Size', 'Run_ID']
    # create an empty data frame to append each new subject
    df = pd.DataFrame(columns=columns)
    # iterate over each subject
    for sub in shape:
        df_sub = pd.DataFrame(sub, columns=columns)
        df = pd.concat((df, df_sub))
    # save the data frame
    df.to_csv((output_dir+'/all_subs_task-aomovie_acq-CS_desc-boldpredicted.tsv'), index=False)
    # we want to check how is the length of the runs distributed in our subjects
    # run_sum = df.groupby('Sub')['Run_Length'].sum().reset_index()
    # run_sum.to_csv((output_dir+'all_subs_task-aomovie_acq-CS_desc-boldpredicted_bysub.tsv'), index=False)
    return df


if __name__ == "__main__":
    root_path = '/data2/jpanzay1/thesis_bids/derivatives/encoding_results_files/'
    mask = 'gray_matter_mask'
    feature = 'lagging2to-10.2melspec850ms'
    stimuli = 'stimuli_alwaysCS'
    acquired = 'acq-CS'
    data = 'predicted_bold'

    input_dir = os.path.join(root_path, mask, feature, stimuli)
    all_sub = []
    all_shape = []
    for subs in glob.iglob((input_dir + '/*')):
        # extract the subject id in each loop
        s_id = subs[-6:]
        for folder in glob.iglob((subs + '/' + acquired + '/' + data)):
            one_sub, sub_shape = concatenate_sub(folder, s_id)
            all_sub.append(one_sub)
            all_shape.append(sub_shape)
    # create the directory in which to store the results
    subject = 'all_subs'
    output_dir = os.path.join(root_path, mask, feature, stimuli, subject, acquired, data)
    os.makedirs(output_dir)
    # check the sizes and dimensions of the data before storing them
    df = check_alignment(all_shape, output_dir)
    with open((output_dir+'/all_subs_task-aomovie_acq-CS_desc-boldpredicted.pkl'), 'wb') as file:
        joblib.dump(all_sub, file)