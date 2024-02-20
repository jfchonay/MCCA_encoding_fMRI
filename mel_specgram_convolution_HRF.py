import numpy as np
import glob
import json
import pandas as pd
import gzip
import matplotlib.pyplot as plt
from nilearn.glm.first_level.hemodynamic_models import glover_hrf


def fft_hrf(tsv_file, freq_labels, hrf):
    """
    The function will apply an FFT to the convolved HRF and stimuli function of each frequency band of the MELspecgram.

     Parameters:
        tsv_file (str):
            the path to the tsv file that contains the data of the stimuli response for one run.
        freq_labels (list):
            a list containing the labels of the frequencies of the MELspecgram.
        hrf (ndarray):
            the hemodynamic response function calculated and sampled in the same scale as our stimuli, as a ndarray.
    Returns:
        fft_all (list):
            a list, of the length of the number of frequencies in each run. Inside the list there is the FFT of the
            convolved HFR and one frequency band, as a ndarray.
        freq_vector (ndarray):
            a frequency vector calculated using the fft.fftfreq function, as a ndarray.
    """
    # convolved the hrf with each frequency band
    con_freqs = convolution_hrf(tsv_file, freq_labels, hrf)
    fft_all = []
    # iterate over all convolved functions
    for c_freq in con_freqs:
        # apply the fft
        fft = np.fft.fft(c_freq)
        fft_all.append(fft)
        # calculate the length in the time domain
        length_s = c_freq.shape[-1]
    # using the length in the time domain, calculate the frequency vector corresponding to this FFT
    freq_vector = np.fft.fftfreq(length_s)
    return fft_all, freq_vector


def convolution_hrf(tsv_file, freq_labels, hrf):
    """
    The function will convolve the HRF to one frequency band of the MELspecgram.

     Parameters:
        tsv_file (str):
            the path to the tsv file that contains the data of the stimuli response for one run.
        freq_labels (list):
            a list containing the labels of the frequencies of the MELspecgram.
        hrf (ndarray):
            the hemodynamic response function calculated and sampled in the same scale as our stimuli, as a ndarray.
    Returns:
        con_freqs (list):
            a list, of the length of the number of frequencies in each run. Inside the list there is the convolved HFR
             with one frequency band, as a ndarray.
    """
    # open and unpack the tsv file as a data frame to extract the information of each frequency band
    with gzip.open(tsv_file, 'rt') as file:
        # create a data frame using as key the frequencies labels
        data_df = pd.read_csv(file, sep='\t', header=None, names=freq_labels)
    con_freqs = []
    # iterate over each of the frequency bands contained inside the data frame
    for freq in freq_labels:
        # extract only the values of the frequency band
        one_freq = data_df[freq].values
        # use the function np.convolve to convolve the frequency band with the HRF
        one_con = np.convolve(one_freq, hrf, 'full')
        con_freqs.append(one_con)
    return con_freqs


def plot_fft(all_fft, n_runs, output):
    """
        The function will plot the FFT of the convolved HRF with the frequency band, it will plot all the frequency
        bands in the same plot for one run. It will create one plot for every run, then it will save it in the
        directory that we specified.

         Parameters:
            all_fft (list):
                a list that contains all the FFT for all the runs, inside the list there is a tuple. The tuple consists
                of a list of the FFT for every frequency band, and a frequency vector.
            n_runs (int):
                the number of runs we are going to plot, as an integer.
            output (str):
                the path where we will save the figures.
        """
    # create a range based on the number of runs to plot
    num_run = range(1, n_runs+1)
    # iterate over our list that contains the FFT and the frequency vector
    for idx, run in enumerate(all_fft):
        # create a string with the name of the run
        this_run = str(num_run[idx])
        # define the x axis of our plot, in this case is our frequency vector
        x = run[1]
        # extract the list containing the values of our FFT
        all_y = run[0]
        # iterate over all the FFT and plot in the same x axis
        for i, y in enumerate(all_y):
            plt.plot(x, y)
        plt.xlabel('Frequencies [Hz]')
        plt.title('FFT of convolution between frequencies of the MELspecgram of CS \n with the HRF for the run' + this_run)
        plt.ylim(-500, 1000)
        plt.xlim(0, 0.2)
        plt.grid(True)
        plot_name = output + '0_to_0.2s_' + 'task-aomovie_acq-CS_run-0' + this_run + '.svg'
        plt.savefig(plot_name, format='svg', dpi=300)
        plt.show()


if __name__ == "__main__":
    # set the folder where our audio stimuli is stored
    root_folder = '/data2/azubaidi1/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/aligned_with_audio/presentingstimuli/acoustic_features/melspec850ms/'
    # extract the json file of our audio stimuli, so that we can get information like the sampling rate and the labels
    # of the frequencies we are using
    with open(root_folder + 'ses-01_task-aomovie_acq-CS_run-01_recording-melspec850ms_stim.json', 'r') as stim_data:
        stim_config = json.load(stim_data)
    sr = stim_config['SamplingFrequency']
    freq_labels = stim_config['Columns']
    # iterate over the folder where our audio stimuli is presented but only extract a list of all the files that are the
    # stimuli files
    stim_tsv = '/**/*melspec850ms_stim.tsv.gz'
    file_list = glob.glob(root_folder+stim_tsv, recursive=True)
    # only extract the files that match the type of stimuli we want, by defining the value of event
    event = 'acq-CS'
    only_con_files = [file for file in file_list if event in file]
    # we only want to have one file per run, as we are using the same stimuli for all subjects, so we are going to
    # iterate over our list of files and create a new list where the files are organized from run 1 to 8
    base_run = 'run-0'
    num_runs = range(1, 9)
    organized_files = []
    for num in num_runs:
        # create the key each_run, which contains the string run-0 and then a number from 1 to 8
        each_run = base_run + str(num)
        # iterate over all the files corresponding to the condition and when the first file matches our key, append to
        # the list of organized_files and exit the loop to move to the next key
        for cs_file in only_con_files:
            if each_run in cs_file:
                organized_files.append(cs_file)
                break
    # using the function glover_hrf we can calculate a hemodynamic response function to convolve with our stimuli
    # function, we input the tr that was used in the fMRI collection, because our tr and the sampling rate of our
    # stimuli are in the same scale we define an oversampling factor of 1
    hrf = glover_hrf(tr=0.850, oversampling=1)
    plt.plot(hrf)
    plt.savefig('/data2/jpanzay1/thesis/encoding_model/melspecgram850ms/can_hrf.svg', format='svg')
    # now we want to convolve each frequency band with the calculated hrf, do an FFT and plot all the frequencies
    fft_all = []
    # iterate over our list of files to extract the stimuli function of each frequency band
    for run in organized_files:
        fft_run, freq_vector = fft_hrf(run, freq_labels, hrf)
        # append to our list as a tuple, that contains all the FFT for the run and the frequency vector
        fft_all.append((fft_run, freq_vector))
    # define where we want to save our information, in this case our plots
    output_dir = '/data2/jpanzay1/thesis_bids/derivatives/encoding_model/melspecgram850ms/'
    plot_fft(fft_all, 8, output_dir)


