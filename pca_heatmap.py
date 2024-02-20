import seaborn as sns
import os
import joblib
from scree_plot import scree_plot
from MCCA import MCCA
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    input_root = '/data2/jpanzay1/thesis_bids/derivatives/encoding_results_files/'
    output_root = '/data2/jpanzay1/thesis_bids/derivatives/mcca_results/'
    mask = 'temporal_lobe_mask'
    feature = 'lagging2to-10.2tag25ms'
    stimuli = 'stimuli_alwaysCS'
    subs = 'all_subs'
    acquired = 'acq-CS'

    input_dir = os.path.join(input_root, mask, feature, stimuli, subs, acquired, 'predicted_bold')
    with open((input_dir + '/all-subs_task-aomovie_acq-CS_desc-boldpredicted.pkl'), 'rb') as pbold_f:
        temp_scores = joblib.load(pbold_f)

    output_dir = os.path.join(output_root, mask, feature, stimuli, subs, acquired, 'pca')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scree_plot(temp_scores[15], output_dir, 130)

    only_scores = np.stack([one_sub[1] for one_sub in temp_scores], axis=0)

    mcca = MCCA(20, pca_only=True)
    pc_scores = mcca.obtain_mcca(only_scores)

    PC = 6
    pc_corr = []
    for j in range(PC):
        corr_matrix = np.corrcoef(np.squeeze(pc_scores[:, :, j]))
        pc_corr.append(corr_matrix)

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 9), dpi=600)
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    for idx, ax in enumerate(ax.flat):
        sns.heatmap(data=pc_corr[idx], cmap='coolwarm',
                    ax=ax, vmin=-1, vmax=1,
                    cbar=idx == 0,
                    cbar_ax=None if idx else cbar_ax)
        ax.set_title(f'PC {idx+1}')
    fig.suptitle('Inter subject correlation of PC scores')
    fig.supxlabel('Subjects')
    fig.supylabel('Subjects')
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig((output_dir + '/heatmap_4.svg'), format='svg', dpi=400)
    plt.close()

