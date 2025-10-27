# Similarity analysis for the representations obtained from different backbones
# Following the method described in: https://proceedings.mlr.press/v97/kornblith19a.html
# "Similarity of Neural Network Representations Revisited"

import numpy as np

def linear_hsic(X, Y):
    """
    Linear HSIC with the (biased) V-statistic estimator.

    Implements Eq. (3) specialized to linear kernels using Eqs. (1)–(2):
        HSIC_linear(X, Y) = ||Y^T X||_F^2 / (n - 1)^2
    when X, Y have columns centered across examples.

    Parameters
    ----------
    X : array_like, shape (n, p1)
        Representation 1 (rows = examples, columns = features).
    Y : array_like, shape (n, p2)
        Representation 2 (same n examples).

    Returns
    -------
    hsic : float
        Linear HSIC. Returns np.nan if n < 2.

    Notes
    -----
    - This is the biased HSIC (V-statistic) used in the paper’s main text (Eq. 3).
    - Centering is performed across examples before computing the statistic.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (examples).")

    n = X.shape[0]
    if n < 2:
        return np.nan

    # Center features across examples (columns)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # HSIC_linear = ||Yc^T Xc||_F^2 / (n-1)^2
    cross = Yc.T @ Xc
    hsic = np.sum(cross * cross) / ((n - 1) ** 2)
    return hsic


def linear_cka(X, Y):
    """
    Linear CKA (Centered Kernel Alignment).

    Implements the closed-form in Table 1 (Linear CKA), which is Eq. (4)
    with linear-kernel HSIC pieces from Eqs. (1)–(2):
        CKA_linear(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )
    when X, Y have columns centered across examples.

    Parameters
    ----------
    X : array_like, shape (n, p1)
        Representation 1 (rows = examples, columns = features).
    Y : array_like, shape (n, p2)
        Representation 2 (same n examples).

    Returns
    -------
    cka : float
        Linear CKA in [0, 1]. Returns np.nan if the denominator is zero.

    Notes
    -----
    - Invariant to orthogonal transforms and isotropic rescaling of features.
    - Centering is performed across examples before computing the statistic.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (examples).")

    # Center features across examples (columns)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Numerator: ||Yc^T Xc||_F^2
    cross = Yc.T @ Xc
    num = np.sum(cross * cross)

    # Denominator: ||Xc^T Xc||_F * ||Yc^T Yc||_F
    XX = Xc.T @ Xc
    YY = Yc.T @ Yc
    den = np.linalg.norm(XX, ord='fro') * np.linalg.norm(YY, ord='fro')

    if den == 0.0:
        return np.nan
    return num / den


if __name__ == "__main__":
    import json
    import os
    import pandas as pd

    # path to the json file containing the file names of the representations
    PATH_DICT_PATH = os.path.join(os.path.dirname(__file__), 'reps_paths_dict.json')

    # types of backbones used
    BACKBONE_TYPES = ['separable', 'entangled', 'entangled_trainable_rzz', 'classical']
    CLASSICAL_BACKBONE_TYPES = ['64P', '4096P']

    with open(PATH_DICT_PATH, "r") as f:
        reps_paths_dict = json.load(f)

    # flatten the path dictionary in a more orderly way
    rep_file_list = {}
    for backbone_type in BACKBONE_TYPES:
        print(f"Processing backbone type: {backbone_type}")
        if backbone_type == 'classical':
            layer_keys = CLASSICAL_BACKBONE_TYPES
        else:
            # sort the layer keys to ensure consistent ordering
            layer_keys = list(reps_paths_dict[backbone_type].keys())
            layer_keys.sort()
        print(layer_keys)
        for layer_key in layer_keys:
            reps_paths_dict[backbone_type][layer_key].sort()
            # extract the seed number
            for file_path in reps_paths_dict[backbone_type][layer_key]:
                if backbone_type == 'classical':
                    # classical has the file name format:
                    # collected_reps/Pong1PCReps_<64P or 4096P>_seed_<seed>.npy
                    seed_num = file_path.split('_seed_')[-1].split('.npy')[0]
                else:
                    # quantum has the file name format:
                    # collected_reps/Pong1PQFMXObsReps_<agent_type>_layers_<num_layers>_seed_<seed>.npy
                    seed_num = file_path.split('_seed_')[-1].split('.npy')[0]
                
                rep_file_list[f"{backbone_type}_{layer_key}_seed_{seed_num}"] = file_path


    # calculate CKA for each pair of representation files
    # and store it in dataframes to be saved as csv files
    cka_results = {}
    for rep_name1, path1 in rep_file_list.items():
        reps1 = np.load(path1)
        cka_results[rep_name1] = {}
        for rep_name2, path2 in rep_file_list.items():
            # avoid repetition
            if rep_name1 != rep_name2:
                print(f"Calculating CKA between {rep_name1} and {rep_name2}")
                reps2 = np.load(path2)
                cka_value = linear_cka(reps1, reps2)
                cka_results[rep_name1][rep_name2] = cka_value
    # convert to dataframe
    cka_df = pd.DataFrame(cka_results)
    # save to csv
    cka_df.to_csv(os.path.join(os.path.dirname(__file__), 'cka_results.csv'))
    print("CKA results saved to cka_results.csv")
    
    