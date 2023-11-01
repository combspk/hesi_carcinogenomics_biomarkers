import pandas as pd
import sys
import matplotlib.pyplot as plt
import glob

def find_inflection_point(IN_DIR="./", OUT_DIR="./"):
    files = glob.glob(f"{IN_DIR}/test_avg_accuracy_ITER-*.csv")
    acc_dict = dict()
    for f in files:
        fname = f.split("/")

        fname = fname[len(fname)-1].split("-")
        fname = fname[len(fname)-1].replace(".csv", "")
        fname = int(fname)
        df = pd.read_csv(f, header=None)
        acc = df[0][0]
        acc_dict[fname] = acc
    acc_iter = []
    acc_scores = []

    inflection_point = 0
    inflection_point_index = 0

    for i in sorted(acc_dict, reverse=True):
        if acc_dict[i] > inflection_point:
            inflection_point = acc_dict[i]
            inflection_point_index = i
        acc_iter.append(i)
        acc_scores.append(acc_dict[i])
        print(f"| {i} || {acc_dict[i]}")

    print(f"BEST ITERATION FOR: {IN_DIR}: {inflection_point_index}")
    return(inflection_point_index)

if __name__ == "__main__":
	find_inflection_point(sys.argv[1], sys.argv[1])
