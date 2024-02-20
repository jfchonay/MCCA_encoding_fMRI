import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns


if __name__ == "__main__":
    output_dir = '/data2/jpanzay1/thesis_bids/derivatives/encoding_model/behavioral_data/'
    # load our data
    with open(output_dir + 'all_subs_sess_acqs_runs_events.json', 'r') as json_file:
        json_data = json.loads(json_file.read())

    with open(output_dir + 'all_subs_sess_acqs_runs_events.pkl', 'rb') as data_file:
        events_data = joblib.load(data_file)
    # we filter everything that does not  have a response and
    filt_data = events_data[events_data['respons_type'] != 'n/a']
    df = filt_data[filt_data['condition'] != 'None']
    df = filt_data[(filt_data['trial_type'] != 'sub_effort') & (filt_data['trial_type'] != 'speech_compr')]
    q_df = df[(df['trial_type'] != 'sub_effort') & (df['trial_type'] != 'speech_compr')]
    q_df['respons_type'] = q_df['respons_type'].apply(lambda x: 'correct' if x == 'hit' else 'incorrect')
    # now we can check if there is a significant difference in the response type according to conditions
    count = q_df.groupby(['sub_id', 'condition', 'respons_type'])['respons_type'].count().unstack().fillna(0)
    proportion = count['correct'] / (count['correct'] + count['incorrect'])
    p_df = proportion.to_frame().reset_index()
    p_df = p_df.rename(columns={0: 'proportion_correct'})
    post_hoc = pg.pairwise_tests(dv='proportion_correct', within='condition', subject='sub_id', data=p_df)
    post_hoc.round(3)
    # post_hoc.to_csv('/data2/jpanzay1/thesis_bids/derivatives/encoding_model/behavioral_data/trial_seg-q_proportion_correct.tsv', index=False)
    # plt.figure(figsize=(16, 10), dpi=400)
    # ax = sns.boxplot(x='condition', y='proportion_correct', data=p_df, order=['CS', 'S2', 'N4'],
    #                  palette='pastel')
    # ax.set_title('Proportion of correct answers for segment questions \n by experimental condition')
    # ax.set_xlabel('Condition')
    # ax.set_ylabel('Proportion of correct answers')
    # ax.set_ylim(0, 1.1)
    # plt.savefig('/data2/jpanzay1/thesis_bids/derivatives/encoding_model/behavioral_data/trial_seg-q_proportion_correct.svg',
    #             format='svg', dpi=400)
    # plt.close()
