import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def train_model(ITER=1, INPUT_FILE="", DATA_DIR="./", MIE="", MIE2="", df=None, interpretation=[], X_holdout=None, y_holdout=[]):
    df = df

    # filter down to only selected columns from feature selection run
    cols = pd.read_csv(f'{DATA_DIR}{MIE.lower()}/test_avg_weights_SORTED_PREREMOVAL_ITER-{ITER}.csv', sep='\t')
    cols = cols['feature'].tolist()

    # then, filter out any probe IDs that don't map to a gene name
    probe_mapping_file = pd.read_csv(f"{DATA_DIR}gene_mapping_{MIE.lower()}.csv", sep='\t')
    badmap = []
    for i, r in probe_mapping_file.iterrows():
        if r['gene_rgd_mapping'] == "Not in DB" and r['gene_blast_mapping'] == "Not in DB":
            badmap.append(r['feature'])
    cols = [feat for feat in cols if feat not in badmap]
    cols_list = []
    for ii in cols:
        cols_list.append(ii)

    df = df[cols_list]

    X = df
    y = interpretation

    # Split the dataset
    # 80% training | 10% validation | 10% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Oversample using SMOTE
    oversample = SMOTE()

    steps = [('o', oversample)] #no undersampling
    pipeline = Pipeline(steps)
    X_train, y_train = pipeline.fit_resample(X_train, y_train)

    # Also SMOTE test set
    ov2 = SMOTE()
    steps2 = [('o', ov2)]
    pipe2 = Pipeline(steps2)
    X_test, y_test = pipe2.fit_resample(X_test, y_test)


    classifier = LinearSVC()

    steps = [('cls', classifier)]
    pipeline = Pipeline(steps)

    for i in range(0, 1, 1):
        # Set the parameters by cross-validation
        tuned_parameters = [
            {"cls__penalty": ['l2'], "cls__loss": ['squared_hinge'], "cls__dual": [False], "cls__tol": [0.0001], "cls__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0] }
        ]
        # just find for f1-score
        score = "f1"

        clf = GridSearchCV(pipeline, tuned_parameters, scoring="%s_macro" % score)
        clf.fit(X_train, y_train)

        X_holdout = X_holdout[cols_list]
        y_true, y_pred = y_holdout, clf.predict(X_holdout)

        print(classification_report(y_true, y_pred))

        clrep = classification_report(y_true, y_pred, output_dict=True)
        clfdf = pd.DataFrame.from_dict(data=clrep)
        clfdf.to_csv(f"{DATA_DIR}/{MIE}__class_report__" + str(i) + "__" + str(score) + "__Over__TOP50__XGB.csv", sep="\t")
        print("... Wrote CV results to file.")

        tmp_rank = dict(
            feature_name=df.columns.tolist(),
            weight=clf.best_estimator_['cls'].coef_[0]
        )        
        clf_ranking = pd.DataFrame.from_dict(data=tmp_rank)

        print(clf_ranking)
        
        clf_ranking.to_csv(f"{DATA_DIR}/{MIE}__{i}__ranking__Over__TOP50__XGB.csv", sep="\t")

