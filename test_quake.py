# from dataset import * 
from os import path
from pgmpy.estimators import TreeSearch, BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
import numpy as np
import pandas as pd
from BNs.phonemic import *
from compute_QI import *
import argparse
                             




def run_quake(fluency: str, epsilon: float):
    bn = get_expert_graph()
    data = pd.read_csv(f"features_binned_binary.csv") # set the path to the data folder
    columns = ['age_of_acquisition_mean','avg_cluster_size','num_switches','num_clusters', 'speech_rate','word_frequency_mean','word_length_mean', 'age_of_acquisition_total_words']
    features = []
    for col in columns:
        features.append(f"{fluency}_{col}")
        

    col_names = ['age_of_acquisition','avg_cluster_size','num_switches','num_clusters','speech_rate', 'word_frequency_mean','word_length_mean', 'total_words']
    filtered_data = data[features].copy()
    filtered_data.columns = col_names
    labels = pd.read_csv("labels.csv")
    filtered_data['cognitive_impairment'] = labels['Year 1/Baseline']
    cog_imp_to_int = {'Normal Aging': 0, 'MCI': 1, 'Alzheimers': 2}
    filtered_data['cognitive_impairment'] = filtered_data['cognitive_impairment'].map(cog_imp_to_int)

    bn.fit(filtered_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    N = 100
    frame = BayesianModelSampling(bn).forward_sample(size=N, show_progress = False, seed = 1234)
    frame2 = BayesianModelSampling(bn).forward_sample(size=N, show_progress = False, seed = 6789)
    names = frame.columns.tolist()
    for col in names:
        frame[col] = frame[col].astype(int)
        frame2[col] = frame2[col].astype(int)

    n = len(frame.columns)
    r = [2 for _ in range(n-1)]
    r.append(2)
    


    C = np.zeros((n, n))
    D = np.zeros((n, n))
    influences = compute_monotonic_influences_from_bn(bn, frame, r, +1, epsilon)
    for i, row in influences[influences.Degree != 0].iterrows():
        C[names.index(row.First), names.index(row.Second)] = +1
        D[names.index(row.First), names.index(row.Second)] = row.Degree


    influences = compute_monotonic_influences_from_bn(bn, frame, r, -1, epsilon)
    for i, row in influences[influences.Degree != 0].iterrows():
        C[names.index(row.First), names.index(row.Second)] = -1
        D[names.index(row.First), names.index(row.Second)] = row.Degree

    output = format_monotonic_influences(C, D ,names)
    print("Monotonicities found:")
    for influence in output:
       print(influence)
    

    

    
    
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Monotonicities")
    parser.add_argument('--fluency', type=str, required=True, choices=['semantic', 'phonemic'],
                        help="Fluency task type: 'semantic' or 'phonemic'")
    parser.add_argument('--epsilon', type=float, default=0,
                        help="Epsilon is the monotonic slack parameter")
    args = parser.parse_args()

    run_quake(fluency=args.fluency, epsilon=args.epsilon)