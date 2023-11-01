from collections import Counter
import glob
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def run_fold(X, y, i, j, OUT_DIR):
    print(f"--FOLD COUNT: {j}")
    try:
        # CREATE 2/3 1/3 SPLIT AND OVERSAMPLE
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # don't do random_state=const, otherwise it gets the same sample every time
        oversample = SMOTE(k_neighbors=1)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        # Also want to oversample testing set but separately from training:
        oversample_test = SMOTE(k_neighbors=1)
        X_test, y_test = oversample_test.fit_resample(X_test, y_test)

        # FIT & SCORE MODEL
        mod = LinearSVC(C=0.01, penalty="l2", dual=False)
        mod.fit(X_train, y_train)
        scr = mod.score(X_test, y_test)

        # SAVE DATA TO PICKLES
        with open(f'{OUT_DIR}/pickle_{i}_{j}_model.pickle', 'wb') as handle:
            pickle.dump([mod, scr], handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('== exception ==')
        print(e)
        return None

def select_features(ITER_COUNT=50, DATA_DIR="./", INPUT_FILE="", MIE="", MIE2="xxx", OUT_DIR="./", df=None, interpretation=[]):
    # load from params
    df = df
    i = 0
    feat_name = df.columns.tolist()
    while len(feat_name) >= 10: 
        print(f"=-=-=-=-=-=-=-= ITER. {i+1} =-=-=-=-=-=-=-=")

        # IF NOT THE FIRST ITERATION, USE ONLY SELECTED FEATURES
        if i > 0:
            df = df[feat_name]

        # GET ACTUAL PREDICTIONS FROM DATA FILE
        interpretation_ppara = []

        if MIE2 == "xxx":
            for index, r in df.iterrows():
                if f'{MIE}' in index:
                    interpretation_ppara.append(1)
                else:
                    interpretation_ppara.append(0)
        else: 
            for index, r in df.iterrows():
                if f'{MIE}' in index or f'{MIE2}' in index:
                    interpretation_ppara.append(1)
                else:
                    interpretation_ppara.append(0)


        X = df
        y = interpretation_ppara

        # CREATE & LAUNCH THREADS
        for j in range(0, 10, 1):
            run_fold(X, y, i, j, OUT_DIR)

        # LOAD PICKLED DATA CREATED IN THREADS
        glob_model = glob.glob(f'{OUT_DIR}/pickle_{i}_*_model.pickle')

        folds_accuracy = 0.0
        folds_feature_weights = dict()
        for j in glob_model:
            print(f"=== MODEL: {i} -- {j} ===")
            with open(j, 'rb') as handle:
                tmp = pickle.load(handle)
                mod = tmp[0] # model
                scr = tmp[1] # score
                folds_accuracy += scr

                # PULL OUT FEATURE NAMES W/ WEIGHTS
                for k in range(0, len(mod.feature_names_in_)):
                    if mod.feature_names_in_[k] in folds_feature_weights.keys():
                        folds_feature_weights[mod.feature_names_in_[k]] = folds_feature_weights[mod.feature_names_in_[k]] + mod.coef_[0][k] # Add to what was already there
                    else :
                        folds_feature_weights[mod.feature_names_in_[k]] = mod.coef_[0][k]

        # GET AVERAGE WEIGHTS FOR EACH FEATURE IN DATASET AND CONVERT TO PANDAS DATAFRAME
        for k,v in folds_feature_weights.items():
            folds_feature_weights[k] = v / len(glob_model)
        folds_feature_weights = pd.DataFrame(folds_feature_weights, index=['weight']).rename_axis('feature', axis=1).transpose().reset_index()

        # SORT FEATURES IN DESCENDING ORDER BY WEIGHTS (IMPORTANCE SCORES)
        folds_feature_weights = folds_feature_weights.sort_values(by='weight', axis=0, ascending=False)

        # SAVE SELECTED FEATURES W/ WEIGHTS TO CSV
        iter_name = i + 1
        iter_name = str(iter_name)
        folds_feature_weights.to_csv(f'{OUT_DIR}/test_avg_weights_SORTED_PREREMOVAL_ITER-{iter_name}.csv', sep='\t')

        # REMOVE BOTTOM 10% OF FEATURES AND SAVE TO CSV
        tmp_removal_threshold = len(folds_feature_weights.index) - int(len(folds_feature_weights.index) * 0.1)
        tmp_folds_feature_weights = folds_feature_weights.head(tmp_removal_threshold)
        feat_name = tmp_folds_feature_weights['feature']
        tmp_df = df[feat_name]
        tmp_df.to_csv(f'{OUT_DIR}/test_avg_weights_SORTED_POSTREMOVAL_ITER-{iter_name}.csv', sep='\t')

        # SAVE MEAN ACCURACY TO CSV
        folds_accuracy = folds_accuracy / len(glob_model)
        f_acc = open(f'{OUT_DIR}/test_avg_accuracy_ITER-{i+1}.csv', 'w')
        f_acc.write(f'{folds_accuracy}\n')
        f_acc.close()

        i = i + 1
