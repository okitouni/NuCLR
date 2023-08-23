import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import (
    QuantileTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import torch
import argparse
from collections import namedtuple, OrderedDict
from .config import datadir
from .utils.tensor_dict import TensorDict


Data = namedtuple(
    "Data",
    [
        "tensor_dict",
        "all_fields",
        "vocab_size",
        "regression_transformers",
        "train_mask",
        "valid_mask",
    ],
)


def delta(Z, N):
    A = Z + N
    aP = 11.18
    delta = aP * A ** (-1 / 2)
    delta[(Z % 2 == 1) & (N % 2 == 1)] *= -1
    delta[(Z % 2 == 0) & (N % 2 == 1)] = 0
    delta[(Z % 2 == 1) & (N % 2 == 0)] = 0
    return delta


def semi_empirical_mass_formula(Z, N):
    A = N + Z
    aV = 15.75
    aS = 17.8
    aC = 0.711
    aA = 23.7
    Eb = (
        aV * A
        - aS * A ** (2 / 3)
        - aC * Z * (Z - 1) / (A ** (1 / 3))
        - aA * (N - Z) ** 2 / A
        + delta(Z, N)
    )
    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


def apply_to_df_col(column):
    def wrapper(fn):
        return lambda df: df[column].astype(str).apply(fn)

    return wrapper


@apply_to_df_col(column="jp")
def get_spin_from(string):
    string = (
        string.replace("(", "")
        .replace(")", "")
        .replace("+", "")
        .replace("-", "")
        .replace("]", "")
        .replace("[", "")
        .replace("GE", "")
        .replace("HIGH J", "")
        .replace(">", "")
        .replace("<", "")
        .strip()
        .split(" ")[0]
    )
    if string == "":
        return float("nan")
    else:
        return float(eval(string))  # eval for 1/2 and such


@apply_to_df_col("jp")
def get_parity_from(string):
    # find the first + or -
    found_plus = string.find("+")
    found_minus = string.find("-")

    if found_plus == -1 and found_minus == -1:
        return float("nan")
    elif found_plus == -1:
        return 0  # -
    elif found_minus == -1:
        return 1  # +
    elif found_plus < found_minus:
        return 1  # +
    elif found_plus > found_minus:
        return 0  # -
    else:
        raise ValueError("something went wrong")


def get_half_life_from(df):
    # selection excludes unknown lifetimes and ones where lifetimes are given as bounds
    series = df.half_life_sec.copy()
    series[(df.half_life_sec == " ") | (df.operator_hl != " ")] = float("nan")
    series = series.astype(float)
    series = series.apply(np.log10)
    return series


@apply_to_df_col("qa")
def get_qa_from(string):
    # ~df.qa.isna() & (df.qa != " ")
    if string == " ":
        return float("nan")
    else:
        return float(string)


@apply_to_df_col("qbm")
def get_qbm_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("qbm_n")
def get_qbm_n_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("qec")
def get_qec_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("sn")
def get_sn_from(string):
    return float(string.replace(" ", "nan"))


@apply_to_df_col("sp")
def get_sp_from(string):
    return float(string.replace(" ", "nan"))


def get_abundance_from(df):
    # abundance:
    # assumes that " " means 0
    return df.abundance.replace(" ", "0").astype(float)


@apply_to_df_col("half_life")
def get_stability_from(string):
    if string == "STABLE":
        return 1.0
    elif string == " ":
        return float("nan")
    else:
        return 0.0


@apply_to_df_col("isospin")
def get_isospin_from(string):
    return float(eval(string.replace(" ", "float('nan')")))


def get_binding_energy_from(df):
    binding = df.binding.replace(" ", "nan").astype(float)
    return binding - semi_empirical_mass_formula(df.z, df.n)


def get_radius_from(df):
    return df.radius.replace(" ", "nan").astype(float)


def get_targets(df):
    # place all targets into targets an empty copy of df
    targets = df[["z", "n"]].copy()
    # binding energy per nucleon
    targets["binding"] = get_binding_energy_from(df)
    # radius in fm
    targets["radius"] = get_radius_from(df)
    # half life in log10(sec)
    targets["half_life_sec"] = get_half_life_from(df)
    # stability in {0, 1, nan}
    targets["stability"] = get_stability_from(df)
    # spin as float
    targets["spin"] = get_spin_from(df)
    # parity as {0 (-),1 (+), nan}
    targets["parity"] = get_parity_from(df)
    # isotope abundance in %
    targets["abundance"] = get_abundance_from(df)
    # qa = alpha decay energy in keV
    targets["qa"] = get_qa_from(df)
    # qbm = beta minus decay energy in keV
    targets["qbm"] = get_qbm_from(df)
    # qbm_n = beta minus + neutron emission energy in keV
    targets["qbm_n"] = get_qbm_n_from(df)
    # qec = electron capture energy in keV
    targets["qec"] = get_qec_from(df)
    # sn = neutron separation energy in keV
    targets["sn"] = get_sn_from(df)
    # sp = proton separation energy in keV
    targets["sp"] = get_sp_from(df)
    # isospin as float
    targets["isospin"] = get_isospin_from(df)

    return targets


def get_nuclear_data(datadir, recreate=False):
    def lc_read_csv(url):
        req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    unclean_data = os.path.join(datadir, "ground_states.csv")
    if recreate or not os.path.exists("data/ground_states.csv"):
        df = lc_read_csv("fields=ground_states&nuclides=all")
        df.to_csv(unclean_data, index=False)
    else:
        # df = pd.read_csv("data/ground_states.csv")
        data_2020 = os.path.join(datadir, "ame2020.csv")
        df2 = pd.read_csv(data_2020).set_index(["z", "n"])
        df2 = df2[~df2.index.duplicated(keep="first")]
        df = pd.read_csv(unclean_data).set_index(["z", "n"])
        df["binding_unc"] = df2.binding_unc
        df["binding_sys"] = df2.binding_sys
        df.reset_index(inplace=True)

    paths = get_decay_paths_for_nuclei(df)
    df = df[(df.z > 8) & (df.n > 8)]
    return df, paths

def get_decay_paths_for_nuclei(df, verbose=False):
    df = df.set_index(["z", "n"])

    stable_sel = df.half_life == "STABLE"
    valid_decays = ["P", "N", "EC+B+", "B+", "EC", "B-", "A", "2P", "2N", "2EC", "2B+", "2B-"]
    unstable_sel = df.decay_1.isin(valid_decays) & ~stable_sel # for some reason some are stable but also decaying
    bigger_nuclei_sel = df.index > (8,8)
    stable_nuclei = df[stable_sel]
    unstable_nuclei = df[unstable_sel & bigger_nuclei_sel]

    def path_to_stable(z, n):
        path = [(z,n)]
        while (z,n) not in stable_nuclei.index:
            match df.loc[z,n].decay_1:
                case "P":
                    z -= 1
                case "N":
                    n -= 1
                case "EC+B+":
                    z -= 1
                    n += 1
                case "B+":
                    z -= 1
                    n += 1
                case "EC":
                    z -= 1
                    n += 1
                case "B-":
                    z += 1
                    n -= 1
                case "A":
                    z -= 2
                    n -= 2
                case "2P":
                    z -= 2
                case "2N":
                    n -= 2
                case "2EC":
                    z -= 2
                    n += 2
                case "2B+":
                    z -= 2
                    n += 2
                case "2B-":
                    z += 2
                    n -= 2
                case _:
                    raise ValueError(f"Unknown decay mode {z,n}: \"{df.loc[z,n].decay_1}\"")
            path.append((z,n))
        return path[::-1]

    paths = {}
    for z,n in unstable_nuclei.index:
        try:
          path = path_to_stable(z,n)
          paths[(z,n)] = path
        except Exception as e:
            if verbose:
              print(f"during processing of {z,n}")
              print(e,"\n")
    for z,n in stable_nuclei.index:
        paths[(z,n)] = [(z,n)]
    return paths



def _train_test_split_exact(X, train_frac, n_embedding_inputs, seed=1):
    """
    Take exactly train_frac of the data as training data.
    """
    # TODO shuffle data when using SGD
    torch.manual_seed(seed)
    while True:
        train_mask = torch.ones(X.shape[0], dtype=torch.bool)
        train_mask[int(train_frac * X.shape[0]) :] = False
        train_mask = train_mask[torch.randperm(X.shape[0])]
        for i in range(n_embedding_inputs):
            if len(X[train_mask][:, i].unique()) != len(X[:, i].unique()):
                print("resampling train mask")
                break
        else:
            break
    test_mask = ~train_mask
    return train_mask, test_mask


def _train_test_split_sampled(X, train_frac, n_embedding_inputs, seed=1):
    """
    Samples are assigned to train by a bernoulli distribution with probability train_frac.
    """
    torch.manual_seed(seed)
    # assert that we have each X at least once in the training set
    while True:
        train_mask = torch.rand(X.shape[0]) < train_frac
        for i in range(n_embedding_inputs):
            if len(X[train_mask][:, i].unique()) != len(X[:, i].unique()):
                print("Resampling train mask")
                break
        else:
            break
    test_mask = ~train_mask
    return train_mask, test_mask


def _train_test_split(size, train_frac, seed=1):
    torch.manual_seed(seed)
    train_mask = torch.zeros(size, dtype=torch.bool)
    train_mask[: int(train_frac * size)] = True
    train_mask = train_mask[torch.randperm(size)]
    return train_mask, ~train_mask


def prepare_nuclear_data(
    config: argparse.Namespace, recreate: bool = False, datadir=datadir
):
    # TODO simplify this function
    """Prepare data to be used for training. Transforms data to tensors, gets tokens X,targets y,
    vocab size and output map which is a dict of {target:output_shape}. Usually output_shape is 1 for regression
    and n_classes for classification.

    Args:
        columns (list, optional): List of columns to use as targets. Defaults to None.
        recreate (bool, optional): Force re-download of data and save to csv. Defaults to False.
    returns (Data): namedtuple of X, y, vocab_size, output_map, quantile_transformer
    """
    df, paths = get_nuclear_data(datadir, recreate=recreate)
    targets = get_targets(df)

    X = torch.tensor(targets[["z", "n"]].values)
    vocab_size = (
        targets.z.max() + 1,
        targets.n.max() + 1,
        len(config.TARGETS_CLASSIFICATION) + len(config.TARGETS_REGRESSION),
    )

    # classification targets increasing integers
    for col in config.TARGETS_CLASSIFICATION:
        targets[col] = targets[col].astype("category").cat.codes
        # put nans back
        targets[col] = targets[col].replace(-1, np.nan)

    output_map = OrderedDict()
    for target in config.TARGETS_CLASSIFICATION:
        output_map[target] = targets[target].nunique()

    for target in config.TARGETS_REGRESSION:
        output_map[target] = 1

    reg_columns = list(config.TARGETS_REGRESSION)
    feature_transformers = dict()
    for col in reg_columns:
        trafo = MinMaxScaler()
        targets[[col]] = trafo.fit_transform(
            targets[[col]].values
        )
        feature_transformers[col] = trafo

    # don't consider nuclei with high uncertainty in binding energy
    # BUT only for evaluation!
    # except_binding = (df.binding_unc * (df.z + df.n) > 100).values
    # targets.loc[test_mask.numpy() & except_binding, "binding_energy"] = np.nan

    pred_table = torch.tensor(targets[list(output_map.keys())].values).float()

    train_mask, test_mask = _train_test_split(
        len(pred_table), config.TRAIN_FRAC, seed=config.SEED
    )

    all_fields = OrderedDict(z=1, n=1, **output_map)

    tensor_dict = TensorDict(
        dict(
            z=X[:, [0]],
            n=X[:, [1]],
            **{k: pred_table[:, [i]] for i, k in enumerate(output_map.keys())}
        )
    )

    return Data(
        tensor_dict.to(config.DEV),
        all_fields,
        vocab_size,
        feature_transformers,
        train_mask.to(config.DEV),
        test_mask.to(config.DEV),
    )
