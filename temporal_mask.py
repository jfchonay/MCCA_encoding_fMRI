from nilearn import image, masking
import joblib
import nibabel as nib
import os
from scree_plot import select_all_subs, sum_variance, scree_plot
from inter_subject_corr import plot_inter_sub_pca, plot_inter_sub_cc, split_tt_data


def apply_temporal_mask(subjects, gray_mask, temporal_mask):
    subjects_temporal = []
    # iterate over every subject
    for one_sub in subjects:
        # define the values of the tuple
        s_id = one_sub[0]
        timeseries = one_sub[1]
        # unmask the data to get an NII file
        gray_unmask = masking.unmask(timeseries, gray_mask)
        # apply the temporal lobe mask
        temporal_lobe = masking.apply_mask(gray_unmask, temporal_mask)
        # append to a list as a tuple with the subject ID
        subjects_temporal.append((s_id, temporal_lobe))
    return subjects_temporal


if __name__ == "__main__":
    input_root = '/data2/jpanzay1/thesis_bids/derivatives/encoding_results_files/'
    output_root = '/data2/jpanzay1/thesis_bids/derivatives/mcca_results/'
    mask = 'gray_matter_mask'
    feature = 'lagging2to-10.2tag25ms'
    stimuli = 'stimuli_alwaysCS'
    subs = 'all_subs'
    acquired = 'acq-CS'
    data = 'predicted_bold'
    input_dir = os.path.join(input_root, mask, feature, stimuli, subs, acquired)
    all_subs = select_all_subs(input_dir, data)
    gray_mask = nib.load('/data2/azubaidi1/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/aligned_with_audio/ROIs/GrayMatter/template_space-MNI152NLin6Asym_res-2_label-GM_mask.nii.gz')
    temporal_mask = image.resample_img('/data2/azubaidi1/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/aligned_with_audio/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz',
                                       gray_mask.affine, gray_mask.shape, interpolation='nearest')
    temporal_lobe = apply_temporal_mask(all_subs, gray_mask, temporal_mask)
    save_dir = os.path.join(input_root, 'temporal_lobe_mask', feature, stimuli, subs, acquired, data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = (save_dir + '/all-subs_task-aomovie_acq-CS_desc-boldpredicted.pkl')
    with open(file_name, 'wb') as file_tm:
        joblib.dump(temporal_lobe, file_tm)
    output_dir = os.path.join(output_root, 'temporal_lobe_mask', feature, stimuli, subs, acquired, 'pca')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    n_pc = [70, 100, 130]
    sum_variance(temporal_lobe, n_pc, output_dir)
    plot_inter_sub_pca(temporal_lobe, 50, output_dir)
    scree_plot(temporal_lobe[15], output_dir, 130)
    new_path = os.path.join(input_root, 'temporal_lobe_mask', feature, stimuli, subs, acquired)
    train_data, test_data = split_tt_data(new_path, 'predicted_bold', temporal_lobe)
    cc_path = os.path.join(output_root, 'temporal_lobe_mask', feature, stimuli, subs, acquired, 'mcca')
    if not os.path.exists(cc_path):
        os.makedirs(cc_path)
    plot_inter_sub_cc(train_data, test_data, 100, 100, path=cc_path)