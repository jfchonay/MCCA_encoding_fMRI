import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, kruskal
import seaborn as sns
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.formula.api import ols


def descriptive_stats(df, trial_type, output_dir):
    stats_by_response(df, trial_type, output_dir)
    stats_by_value(df, trial_type, output_dir)
    stats_by_rt(df, trial_type, output_dir)
    return


def stats_by_response(df, trial_type, output_dir):
    # delete all odd conditions
    df = df[df['condition'] != 'None']
    # evaluate if the accuracy of responses is affected by the condition presented
    contingency_table_cr = pd.crosstab(df['condition'], df['respons_type'])
    # apply a chi square test, test if the condition has an effect of proportion of response types
    chi2_cr, p_cr, dof_cr, expected_freq_cr = chi2_contingency(contingency_table_cr)
    # calculate the percentage of response types
    percentages_cr = contingency_table_cr.div(contingency_table_cr.sum(axis=1), axis=0) * 100
    # create a figure and save it
    plot_name_1 = output_dir + 'all_subs_sess_acqs_runs_events_' + 'response_type_condition_trial_' + trial_type + '.svg'
    plt.figure(figsize=(24, 16), dpi=400)
    ax_1 = percentages_cr.plot(kind='bar', stacked=True, rot=0)
    ax_1.set_xlabel('Condition')
    ax_1.set_ylabel('Percentage')
    ax_1.set_title('Percentage of Response Types by Condition \n for trial: '
                 + trial_type + f'\n Chi-squared test p-value: {p_cr:.4f}', fontsize = 10)
    ax_1.legend(title='Response type', loc='upper right')
    plt.savefig(plot_name_1, format='svg', dpi=400)

    # evaluate if the accuracy of responses is affected by the condition presented
    contingency_table_rr = pd.crosstab(df['run_id'], df['respons_type'])
    # apply a chi square test, test if the condition has an effect of proportion of response types
    chi2_rr, p_rr, dof_rr, expected_freq_rr = chi2_contingency(contingency_table_rr)
    # calculate the percentage of response types
    percentages_rr = contingency_table_rr.div(contingency_table_rr.sum(axis=1), axis=0) * 100
    # create a figure and save it
    plot_name_2 = output_dir + 'all_subs_sess_acqs_runs_events_' + 'response_type_run_id_trial_' + trial_type + '.svg'
    plt.figure(figsize=(24, 16), dpi=400)
    ax_2 = percentages_rr.plot(kind='bar', stacked=True, rot=0)
    ax_2.set_xlabel('Run ID')
    ax_2.set_ylabel('Percentage')
    ax_2.set_title('Percentage of Response Types by Run \n for trial: '
                 + trial_type + f'\n Chi-squared test p-value: {p_rr:.4f}', fontsize=10)
    ax_2.legend(title='Response type', loc='upper right')
    plt.savefig(plot_name_2, format='svg', dpi=400)
    return


def stats_by_value(df, trial_type, output_dir):
    # delete all n/a values and None conditions
    df = df[df['value'] != 'n/a']
    df = df[df['condition'] != 'None']
    # run a kruskal wallis test to evaluate differences in values
    conditions = df['condition'].unique()
    df_by_c = [df[df['condition'] == condition]['value'] for condition in conditions]
    stat_krus, p_krus = kruskal(*df_by_c)
    plot_name_1 = output_dir + 'all_subs_sess_acqs_runs_events_' + 'value_condition_trial_' + trial_type + '.svg'
    plt.figure(figsize=(16, 10), dpi=400)
    ax_1 = sns.violinplot(df, x=pd.to_numeric(df['value']), y=df['condition'])
    ax_1.set_title('Value of responses by condition \n for trial: '
             + trial_type + f'\n Kruskal Wallis test p-value: {p_krus:.4f}', fontsize=18)
    ax_1.set_xlabel('Value')
    ax_1.set_ylabel('Condition')
    plt.savefig(plot_name_1, format='svg', dpi=400)
    # run post hoc test
    if p_krus < 0.001:
        dunn_test = sp.posthoc_dunn(df, val_col='value', group_col='condition', p_adjust='bonferroni')
        plot_name_2 = output_dir + 'all_subs_sess_acqs_runs_events_post_hoc' + 'value_condition_trial_' + trial_type + '.svg'
        plt.figure(figsize=(16, 10), dpi=400)
        ax_2 = sns.heatmap(dunn_test,annot=True, fmt=".4f", cmap="YlGnBu", cbar=False)
        ax_2.set_title('Pairwise group differences of values between conditions \n for trial: '
                       + trial_type, fontsize=18)
        plt.savefig(plot_name_2, format='svg', dpi=400)
    return


def stats_by_rt(df, trial_type, output_dir):
    df = df[df['value'] != 'n/a']
    df = df[df['reaction_time'] != 'n/a']
    df = df[df['condition'] != 'None']
    cols = ['value', 'reaction_time']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    # we want to check for differences in reaction time, if they are affected by the type of answer or the condition
    two_way_anova(df, trial_type, 'value', output_dir)
    two_way_anova(df, trial_type, 'respons_type', output_dir)
    return


def two_way_anova(df, trial_type, response, output_dir):
    formula = 'reaction_time ~ '+response+ ' + condition + '+response+':condition'
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    plot_name_1 = output_dir + 'all_subs_sess_acqs_runs_events_anova' + 'reaction_time_' + response + \
                  '_condition_trial_' + trial_type + '.svg'
    plt.figure(figsize=(16, 10), dpi=400)
    ax_1 = sns.heatmap(anova_table, annot=True, fmt=".4f", cmap="BuPu", cbar=False)
    ax_1.set_title('Two way ANOVA of reaction times between ' + response + ' conditions \n for trial: '
                   + trial_type, fontsize=18)
    plt.savefig(plot_name_1, format='svg', dpi=400)
    plot_name_2 = output_dir + 'all_subs_sess_acqs_runs_events_' + 'reaction_time_' + response + \
                  '_condition_trial_' + trial_type + '.svg'
    plt.figure(figsize=(16, 10), dpi=400)
    ax_2 = sns.barplot(x='condition', y='reaction_time', hue=response, data=df)
    ax_2.set_title('Reaction time by condition and ' +response+ '\n for trial: '
                   + trial_type, fontsize=18)
    ax_2.set_xlabel('Condition')
    ax_2.set_ylabel('Reaction Time')
    plt.savefig(plot_name_2, format='svg', dpi=400)
    return


if __name__ == "__main__":
    root_folder = '/data2/azubaidi1/ForrestGumpHearingLoss/BIDS_ForrGump/rawdata/'
    output_dir = '/data2/jpanzay1/thesis_bids/derivatives/encoding_model/behavioral_data/'

    with open(output_dir + 'all_subs_sess_acqs_runs_events.json', 'r') as json_file:
        json_data = json.loads(json_file.read())

    with open(output_dir + 'all_subs_sess_acqs_runs_events.pkl', 'rb') as data_file:
        events_data = joblib.load(data_file)

    filt_data = events_data[events_data['respons_type'] != 'n/a']

    speech_compr_df = filt_data[filt_data['trial_type'] == 'speech_compr']

    sub_effort_df = filt_data[filt_data['trial_type'] == 'sub_effort']

    seg_q_df = filt_data[(filt_data['trial_type'] != 'sub_effort') &
                         (filt_data['trial_type'] != 'speech_compr')]

    descriptive_stats(speech_compr_df, 'speech_comprehension', output_dir)
    descriptive_stats(sub_effort_df, 'sub_effort', output_dir)
    descriptive_stats(speech_compr_df, 'seg_q', output_dir)