import pandas as pd
import numpy as np
import itertools
import json, csv, os, sys
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def distance(x1, x2):
    return min(1, abs((x1 - x2) / (x1+1e-15)))


def compute_cost_matrix(a1, a2):
    cost_matrix = np.zeros((len(a1), len(a2)))
    for index1, elt1 in enumerate(a1):
        for index2, elt2 in enumerate(a2):
            cost_matrix[index1, index2] = distance(elt1, elt2)
    return cost_matrix


def compute_score(lst1, lst2):
    cost_matrix = compute_cost_matrix(lst1, lst2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()
    score = 1 - cost / max(len(lst1), len(lst2))
    return score

def remove_strings(lst):
    new_lst = []
    for elt in lst:
        elt = str(elt).replace("%", '')
        # Filter out strings.
        try:
            new_lst.append(float(elt))
        except:
            continue
    return new_lst

def main():
    ground_truth_tables_folder = sys.argv[1]
    extracted_tables_folder = sys.argv[2]
    scores = []
    for filename in tqdm(os.listdir(ground_truth_tables_folder)):
        # Load Ground Truth Data Table
        df = pd.read_csv(os.path.join(ground_truth_tables_folder, filename))
        values = df.values
        flattened = list(itertools.chain.from_iterable(values))
        flattened = remove_strings(flattened)

        # Load extracted Data Table
        df2 = pd.read_csv(os.path.join(extracted_tables_folder, filename))
        values2 = df2.values
        flattened2 = list(itertools.chain.from_iterable(values2))
        flattened2 = remove_strings(flattened2)
        # Check if there are no numerical values in the extracted data after filtering out strings.
        if len(flattened2) == 0:
            # Consider this as a failure.
            score = 0
        else:
            # Compute the Score
            score = compute_score(flattened, flattened2)
        scores.append(score)
    print("Accuracy: ", sum(scores)/len(scores))

if __name__ == "__main__":
    main()