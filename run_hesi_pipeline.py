# Import other scripts and packages
import pandas as pd
import select_features_pipeline as sfp
import accuracy_check_pipeline as acp
import ml_train_pipeline as mtp
from multiprocessing import Process
from sklearn.model_selection import train_test_split

# Load configured variables from config file
vars = open("config.txt", "r").readlines()
vars_dict = dict()
for i in vars:
    spl = i.rstrip().split("=")
    vars_dict[spl[0]] = spl[1].replace("\"", "")
ITER_COUNT = vars_dict["ITER_COUNT"]
DATA_DIR = vars_dict["DATA_DIR"]
INPUT_FILE = vars_dict["INPUT_FILE"]

def run_ml(mie, mie2, ITER_COUNT, DATA_DIR, INPUT_FILE):
    MIE = mie
    MIE2 = mie2
    OUT_DIR = f"{DATA_DIR}/{MIE.lower()}/"

    # Create holdout subset from full data matrix
    df = pd.read_csv(f'{DATA_DIR}{INPUT_FILE}', sep='\t', index_col=0)
    df = df.dropna(axis=0, how='any')

    interpretation = []
    for index, r in df.iterrows():
        if f'{MIE}' in index or f'{MIE2}' in index:
            interpretation.append(1)
        else:
            interpretation.append(0)
    X = df
    y = interpretation

    X_data, X_holdout, y_data, y_holdout = train_test_split(X, y, test_size=0.2)    

    print('Created main/holdout split')
    print('=== X_data ===')
    print(X_data)
    print('=== y_data ===')
    print(y_data)

    # Run feature selection
    sfp.select_features(ITER_COUNT=ITER_COUNT, DATA_DIR=DATA_DIR, INPUT_FILE=INPUT_FILE, MIE=MIE, MIE2=MIE2, OUT_DIR=OUT_DIR, df=X_data, interpretation=y_data)

    print('=== SELECT FEATURES COMPLETE ===')

    # Find inflection point and get features there
    inflection_point = acp.find_inflection_point(IN_DIR=OUT_DIR, OUT_DIR=OUT_DIR)

    print('=== INFLECTION POINT COMPLETE ===')

    # Train model using features at the given inflection point
    mtp.train_model(ITER=inflection_point, INPUT_FILE=INPUT_FILE, DATA_DIR=DATA_DIR, MIE=MIE, MIE2=MIE2, df=X_data, interpretation=y_data, X_holdout=X_holdout, y_holdout=y_holdout)
    print(f"Finished {mie}.")

if __name__ == '__main__':
    p_ahr = Process(target=run_ml, args=('AhR', "xxx",  ITER_COUNT, DATA_DIR, INPUT_FILE))
    p_car = Process(target=run_ml, args=('CAR', "xxx",  ITER_COUNT, DATA_DIR, INPUT_FILE))
    p_cytotox = Process(target=run_ml, args=('Cytotox', 'xxx', ITER_COUNT, DATA_DIR, INPUT_FILE))
    p_er = Process(target=run_ml, args=('ER', "xxx",  ITER_COUNT, DATA_DIR, INPUT_FILE))
    p_genotoxicity = Process(target=run_ml, args=('Genotoxicity', 'xxx',  ITER_COUNT, DATA_DIR, INPUT_FILE))
    p_ppara = Process(target=run_ml, args=('PPARa', "xxx", ITER_COUNT, DATA_DIR, INPUT_FILE))

    p_ahr.start()
    p_car.start()
    p_cytotox.start()
    p_er.start()
    p_genotoxicity.start()
    p_ppara.start()

    p_ahr.join()
    p_car.join()
    p_cytotox.join()
    p_er.join()
    p_genotoxicity.join()
    p_ppara.join()
