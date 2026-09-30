# ============================================================
# AMAZON ML CHALLENGE 2026
# BUSINESS ENTITY RESOLUTION - COMPLETE IMPROVED PIPELINE
# ============================================================
#
# IMPORTANT:
# This script does NOT create random/dummy training data.
# It uses the real Amazon TSV files and ground truth.
#
# No code can honestly guarantee 100% accuracy on unseen data.
# This version is designed to maximize the actual entity-resolution
# Macro F0.5 score available from the training ground truth.
#
# ============================================================

# ============================================================
# 1. INSTALL / IMPORT
# ============================================================

!pip install -q kagglehub rapidfuzz

import os
import re
import gc
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from rapidfuzz import fuzz

warnings.filterwarnings("ignore")

print("Environment ready")


# ============================================================
# 2. DOWNLOAD DATASET
# ============================================================

import kagglehub

DATASET = "satwiksps/amazon-ml-challenge-2026"

DATA_ROOT = Path(
    kagglehub.dataset_download(DATASET)
)

print("Dataset downloaded:")
print(DATA_ROOT)


# ============================================================
# 3. LOCATE FILES
# ============================================================

def find_file(filename):
    hits = list(DATA_ROOT.rglob(filename))
    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename}"
        )
    return hits[0]


FILES = {
    "train_s1": find_file("train_source1.tsv"),
    "train_s2": find_file("train_source2.tsv"),
    "train_s3": find_file("train_source3.tsv"),
    "ground_truth": find_file("train_ground_truth.tsv"),
    "test_s1": find_file("test_source1.tsv"),
    "test_s2": find_file("test_source2.tsv"),
    "test_s3": find_file("test_source3.tsv"),
}

for name, path in FILES.items():
    print(f"{name:15s} -> {path}")


# ============================================================
# 4. BASIC SETTINGS
# ============================================================

CORE_COLS = [
    "entity_id",
    "business_name",
    "business_address",
    "country"
]

RANDOM_STATE = 42

# Validation sizes.
# Increase these if Colab RAM/time allows.
VAL_S1_N = 10000
VAL_S2_N = 50000
VAL_S3_N = 50000

# Set True first for a fast development run.
# Set False for full test inference.
FAST_TEST_MODE = True

DEV_TEST_S1 = 25000
DEV_TEST_S2 = 100000
DEV_TEST_S3 = 100000


# ============================================================
# 5. NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "incorporated",
    "inc",
    "corporation",
    "corp",
    "limited",
    "ltd",
    "llc",
    "llp",
    "private",
    "pvt",
    "company",
    "co",
    "limitedliabilitycompany"
}


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    # Normalize common symbol.
    x = x.replace("&", " and ")

    # Keep letters/numbers.
    x = re.sub(
        r"[^a-z0-9\s]",
        " ",
        x
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x


def normalize_name(x):
    s = normalize_text(x)

    tokens = [
        token
        for token in s.split()
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(tokens)


def normalize_address(x):
    return normalize_text(x)


def compact(x):
    return re.sub(
        r"\s+",
        "",
        str(x)
    )


def first_token(x):
    parts = str(x).split()
    return parts[0] if parts else ""


def prepare_dataframe(df):
    out = df[
        CORE_COLS
    ].copy().fillna("")

    out["norm_name"] = (
        out["business_name"]
        .map(normalize_name)
    )

    out["norm_address"] = (
        out["business_address"]
        .map(normalize_address)
    )

    out["compact_name"] = (
        out["norm_name"]
        .map(compact)
    )

    out["compact_address"] = (
        out["norm_address"]
        .map(compact)
    )

    # Exact blocking keys.
    out["country_name_key"] = (
        out["country"].astype(str)
        + "|"
        + out["compact_name"]
    )

    out["country_addr_key"] = (
        out["country"].astype(str)
        + "|"
        + out["compact_address"]
    )

    out["country_name_addr_key"] = (
        out["country"].astype(str)
        + "|"
        + out["compact_name"]
        + "|"
        + out["compact_address"]
    )

    # Additional robust blocking keys.
    out["country_name_prefix_key"] = (
        out["country"].astype(str)
        + "|"
        + out["compact_name"].str[:6]
    )

    out["country_addr_prefix_key"] = (
        out["country"].astype(str)
        + "|"
        + out["compact_address"].str[:8]
    )

    out["country_name_first_token_key"] = (
        out["country"].astype(str)
        + "|"
        + out["norm_name"].map(first_token)
    )

    return out


# ============================================================
# 6. LOAD GROUND TRUTH
# ============================================================

gt = pd.read_csv(
    FILES["ground_truth"],
    sep="\t",
    dtype=str
).fillna("")


def split_ids(x):
    if not x:
        return []

    return [
        value.strip()
        for value in str(x).split(",")
        if value.strip()
    ]


gt["match_list"] = (
    gt["matched_entity_ids"]
    .map(split_ids)
)

gt_lookup = dict(
    zip(
        gt["source1_entity_id"],
        gt["match_list"]
    )
)

print("\nGround truth rows:", len(gt))

print(
    "S1 with at least one match:",
    int(
        (
            gt["match_list"].str.len() > 0
        ).sum()
    )
)


# ============================================================
# 7. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

train_s1_val = pd.read_csv(
    FILES["train_s1"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str,
    nrows=VAL_S1_N
)

train_s2_val = pd.read_csv(
    FILES["train_s2"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str,
    nrows=VAL_S2_N
)

train_s3_val = pd.read_csv(
    FILES["train_s3"],
    sep="\t",
    usecols=CORE_COLS,
    dtype=str,
    nrows=VAL_S3_N
)

train_s1_val = prepare_dataframe(
    train_s1_val
)

train_s2_val = prepare_dataframe(
    train_s2_val
)

train_s3_val = prepare_dataframe(
    train_s3_val
)

print("Validation S1:", len(train_s1_val))
print("Validation S2:", len(train_s2_val))
print("Validation S3:", len(train_s3_val))


# ============================================================
# 8. BUILD INDEXES
# ============================================================

BLOCK_COLUMNS = [
    "country_name_key",
    "country_addr_key",
    "country_name_addr_key",
    "country_name_prefix_key",
    "country_addr_prefix_key",
    "country_name_first_token_key"
]


def build_index(df, column):
    index = defaultdict(list)

    for row in df.itertuples(index=False):

        key = getattr(
            row,
            column
        )

        if key:
            index[key].append(
                row.entity_id
            )

    return index


def make_indexes(df):

    return {
        column: build_index(
            df,
            column
        )
        for column in BLOCK_COLUMNS
    }


val_idx_s2 = make_indexes(
    train_s2_val
)

val_idx_s3 = make_indexes(
    train_s3_val
)


s2_by_id = (
    train_s2_val
    .set_index("entity_id")
    .to_dict("index")
)

s3_by_id = (
    train_s3_val
    .set_index("entity_id")
    .to_dict("index")
)


print("Validation indexes built.")


# ============================================================
# 9. CANDIDATE GENERATION
# ============================================================

def candidates_for_row(
    row,
    index_dict,
    max_candidates=500
):

    if hasattr(row, "_asdict"):
        row = row._asdict()
    elif hasattr(row, "to_dict"):
        row = row.to_dict()

    candidates = set()

    for column in BLOCK_COLUMNS:

        key = row.get(column)

        if not key:
            continue

        values = index_dict.get(
            column,
            {}
        ).get(
            key,
            []
        )

        candidates.update(
            values
        )

        # Avoid pathological candidate explosion.
        if len(candidates) >= max_candidates:
            break

    return candidates


# ============================================================
# 10. TOKEN / FUZZY FEATURES
# ============================================================

def jaccard_tokens(a, b):

    A = set(
        str(a).split()
    )

    B = set(
        str(b).split()
    )

    if not A or not B:
        return 0.0

    return (
        len(A & B)
        /
        len(A | B)
    )


def pair_features(a, b):

    name1 = str(
        a.get(
            "norm_name",
            ""
        )
    )

    name2 = str(
        b.get(
            "norm_name",
            ""
        )
    )

    addr1 = str(
        a.get(
            "norm_address",
            ""
        )
    )

    addr2 = str(
        b.get(
            "norm_address",
            ""
        )
    )

    compact_name1 = str(
        a.get(
            "compact_name",
            compact(name1)
        )
    )

    compact_name2 = str(
        b.get(
            "compact_name",
            compact(name2)
        )
    )

    compact_addr1 = str(
        a.get(
            "compact_address",
            compact(addr1)
        )
    )

    compact_addr2 = str(
        b.get(
            "compact_address",
            compact(addr2)
        )
    )

    country1 = str(
        a.get(
            "country",
            ""
        )
    ).strip().lower()

    country2 = str(
        b.get(
            "country",
            ""
        )
    ).strip().lower()

    name_exact = int(
        compact_name1 != ""
        and
        compact_name1 == compact_name2
    )

    addr_exact = int(
        compact_addr1 != ""
        and
        compact_addr1 == compact_addr2
    )

    country_exact = int(
        country1 != ""
        and
        country1 == country2
    )

    name_jaccard = jaccard_tokens(
        name1,
        name2
    )

    addr_jaccard = jaccard_tokens(
        addr1,
        addr2
    )

    name_fuzzy = (
        fuzz.token_set_ratio(
            name1,
            name2
        ) / 100.0
        if name1 and name2
        else 0.0
    )

    addr_fuzzy = (
        fuzz.token_set_ratio(
            addr1,
            addr2
        ) / 100.0
        if addr1 and addr2
        else 0.0
    )

    name_ratio = (
        fuzz.ratio(
            name1,
            name2
        ) / 100.0
        if name1 and name2
        else 0.0
    )

    addr_ratio = (
        fuzz.ratio(
            addr1,
            addr2
        ) / 100.0
        if addr1 and addr2
        else 0.0
    )

    return {
        "name_exact": name_exact,
        "addr_exact": addr_exact,
        "country_exact": country_exact,
        "name_jaccard": name_jaccard,
        "addr_jaccard": addr_jaccard,
        "name_fuzzy": name_fuzzy,
        "addr_fuzzy": addr_fuzzy,
        "name_ratio": name_ratio,
        "addr_ratio": addr_ratio
    }


# ============================================================
# 11. BUILD VALIDATION CANDIDATE PAIRS
# ============================================================

print("\nBuilding validation candidate pairs...")

candidate_rows = []

start = time.time()

for row in train_s1_val.itertuples(
    index=False
):

    s1_id = row.entity_id

    row_dict = row._asdict()

    true_ids = set(
        gt_lookup.get(
            s1_id,
            []
        )
    )

    candidates = set()

    candidates.update(
        candidates_for_row(
            row_dict,
            val_idx_s2
        )
    )

    candidates.update(
        candidates_for_row(
            row_dict,
            val_idx_s3
        )
    )

    for candidate_id in candidates:

        if candidate_id in s2_by_id:

            candidate = s2_by_id[
                candidate_id
            ]

        elif candidate_id in s3_by_id:

            candidate = s3_by_id[
                candidate_id
            ]

        else:
            continue

        features = pair_features(
            row_dict,
            candidate
        )

        candidate_rows.append({
            "s1_id": s1_id,
            "candidate_id": candidate_id,
            "label": int(
                candidate_id in true_ids
            ),
            **features
        })


candidate_df = pd.DataFrame(
    candidate_rows
)

print(
    "Candidate pairs:",
    f"{len(candidate_df):,}"
)

print(
    "Positive pairs:",
    int(
        candidate_df["label"].sum()
    )
)

print(
    "Time:",
    round(
        time.time() - start,
        2
    ),
    "seconds"
)


# ============================================================
# 12. CANDIDATE RECALL
# ============================================================

grouped_candidates = {
    sid: group
    for sid, group
    in candidate_df.groupby(
        "s1_id"
    )
}

candidate_recall_values = []

for sid, true_ids in gt_lookup.items():

    if sid not in grouped_candidates:
        candidate_recall_values.append(
            1.0 if not true_ids else 0.0
        )
        continue

    candidate_ids = set(
        grouped_candidates[sid][
            "candidate_id"
        ]
    )

    true_set = set(true_ids)

    if not true_set:
        candidate_recall_values.append(
            1.0
        )
    else:
        candidate_recall_values.append(
            len(
                true_set &
                candidate_ids
            )
            /
            len(true_set)
        )

candidate_recall = float(
    np.mean(
        candidate_recall_values
    )
)

print(
    "\nCandidate Recall:",
    round(
        candidate_recall,
        6
    )
)


# ============================================================
# 13. MACRO F0.5
# ============================================================

def f05_single(
    true_ids,
    predicted_ids
):

    true_set = set(true_ids)
    pred_set = set(predicted_ids)

    if not true_set and not pred_set:
        return 1.0

    if not pred_set:
        return 0.0

    tp = len(
        true_set &
        pred_set
    )

    precision = (
        tp /
        len(pred_set)
    )

    recall = (
        tp /
        len(true_set)
        if true_set
        else 0.0
    )

    if precision == 0 and recall == 0:
        return 0.0

    beta2 = 0.25

    return (
        (1 + beta2)
        *
        precision
        *
        recall
        /
        (
            beta2 * precision
            +
            recall
        )
    )


def macro_f05(
    y_true,
    y_pred
):

    return float(
        np.mean(
            [
                f05_single(
                    true,
                    pred
                )
                for true, pred
                in zip(
                    y_true,
                    y_pred
                )
            ]
        )
    )


# ============================================================
# 14. IMPROVED MATCHING RULE
# ============================================================

def optimized_match(
    f,
    name_threshold,
    address_threshold
):

    # Exact agreement on both fields.
    if (
        f["name_exact"]
        and
        f["addr_exact"]
    ):
        return True

    # Exact name + strong address.
    if (
        f["name_exact"]
        and
        f["addr_fuzzy"]
        >= address_threshold
    ):
        return True

    # Exact address + strong name.
    if (
        f["addr_exact"]
        and
        f["name_fuzzy"]
        >= name_threshold
    ):
        return True

    # Both fields are reasonably strong.
    if (
        f["name_jaccard"]
        >= name_threshold
        and
        f["addr_jaccard"]
        >= address_threshold
    ):
        return True

    # Very strong fuzzy name + address.
    if (
        f["name_fuzzy"] >= 0.97
        and
        f["addr_fuzzy"] >= 0.90
        and
        f["country_exact"]
    ):
        return True

    # Very strong address + name.
    if (
        f["addr_fuzzy"] >= 0.97
        and
        f["name_fuzzy"] >= 0.90
        and
        f["country_exact"]
    ):
        return True

    return False


# ============================================================
# 15. THRESHOLD SEARCH
# ============================================================

def evaluate_thresholds(
    name_threshold,
    address_threshold
):

    scores = []

    for sid, true_ids in gt_lookup.items():

        group = grouped_candidates.get(
            sid
        )

        predicted = []

        if group is not None:

            for _, r in group.iterrows():

                features = {
                    key: r[key]
                    for key in [
                        "name_exact",
                        "addr_exact",
                        "country_exact",
                        "name_jaccard",
                        "addr_jaccard",
                        "name_fuzzy",
                        "addr_fuzzy",
                        "name_ratio",
                        "addr_ratio"
                    ]
                }

                if optimized_match(
                    features,
                    name_threshold,
                    address_threshold
                ):

                    predicted.append(
                        r["candidate_id"]
                    )

        scores.append(
            f05_single(
                true_ids,
                predicted
            )
        )

    return float(
        np.mean(scores)
    )


print("\nSearching thresholds...")

threshold_values = np.arange(
    0.50,
    1.001,
    0.025
)

threshold_results = []

for name_threshold in threshold_values:

    for address_threshold in threshold_values:

        score = evaluate_thresholds(
            name_threshold,
            address_threshold
        )

        threshold_results.append({
            "name_threshold":
                round(
                    float(name_threshold),
                    3
                ),

            "address_threshold":
                round(
                    float(address_threshold),
                    3
                ),

            "macro_f05":
                score
        })


threshold_df = (
    pd.DataFrame(
        threshold_results
    )
    .sort_values(
        "macro_f05",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\nTOP THRESHOLDS")

display(
    threshold_df.head(20)
)


# ============================================================
# 16. BEST THRESHOLDS
# ============================================================

BEST_NAME_THRESH = float(
    threshold_df.iloc[0][
        "name_threshold"
    ]
)

BEST_ADDR_THRESH = float(
    threshold_df.iloc[0][
        "address_threshold"
    ]
)

BEST_F05 = float(
    threshold_df.iloc[0][
        "macro_f05"
    ]
)

print(
    "\nBest name threshold:",
    BEST_NAME_THRESH
)

print(
    "Best address threshold:",
    BEST_ADDR_THRESH
)

print(
    "Best validation Macro F0.5:",
    round(
        BEST_F05,
        6
    )
)


# ============================================================
# 17. CURRENT BASELINE COMPARISON
# ============================================================

BASELINE_NAME_THRESH = 0.80
BASELINE_ADDR_THRESH = 0.80

baseline_f05 = evaluate_thresholds(
    BASELINE_NAME_THRESH,
    BASELINE_ADDR_THRESH
)

print("\n" + "=" * 70)
print("BASELINE VS OPTIMIZED")
print("=" * 70)

print(
    "Baseline Macro F0.5:",
    round(
        baseline_f05,
        6
    )
)

print(
    "Optimized Macro F0.5:",
    round(
        BEST_F05,
        6
    )
)

print(
    "Improvement:",
    round(
        BEST_F05 - baseline_f05,
        6
    )
)


# ============================================================
# 18. VALIDATION PREDICTIONS
# ============================================================

validation_predictions = {}

for sid, true_ids in gt_lookup.items():

    group = grouped_candidates.get(
        sid
    )

    predicted = []

    if group is not None:

        for _, r in group.iterrows():

            features = {
                key: r[key]
                for key in [
                    "name_exact",
                    "addr_exact",
                    "country_exact",
                    "name_jaccard",
                    "addr_jaccard",
                    "name_fuzzy",
                    "addr_fuzzy",
                    "name_ratio",
                    "addr_ratio"
                ]
            }

            if optimized_match(
                features,
                BEST_NAME_THRESH,
                BEST_ADDR_THRESH
            ):

                predicted.append(
                    r["candidate_id"]
                )

    validation_predictions[sid] = sorted(
        set(predicted)
    )


# ============================================================
# 19. VALIDATION ERROR ANALYSIS
# ============================================================

error_rows = []

for sid, true_ids in gt_lookup.items():

    pred_ids = validation_predictions.get(
        sid,
        []
    )

    true_set = set(true_ids)
    pred_set = set(pred_ids)

    error_rows.append({
        "source1_entity_id": sid,
        "true_count": len(true_set),
        "pred_count": len(pred_set),
        "TP": len(
            true_set &
            pred_set
        ),
        "FP": len(
            pred_set -
            true_set
        ),
        "FN": len(
            true_set -
            pred_set
        ),
        "F0.5": f05_single(
            true_set,
            pred_set
        )
    })


errors_df = pd.DataFrame(
    error_rows
)

print("\nValidation error analysis:")

display(
    errors_df
    .sort_values(
        ["FP", "FN"],
        ascending=False
    )
    .head(30)
)


# ============================================================
# 20. SAVE OPTIMIZED CONFIGURATION
# ============================================================

import joblib

optimization_config = {
    "name_threshold":
        BEST_NAME_THRESH,

    "address_threshold":
        BEST_ADDR_THRESH,

    "validation_macro_f05":
        BEST_F05,

    "baseline_macro_f05":
        baseline_f05,

    "candidate_recall":
        candidate_recall
}

CONFIG_PATH = (
    "/content/"
    "amazon_entity_resolution_config.pkl"
)

joblib.dump(
    optimization_config,
    CONFIG_PATH
)

print(
    "\nSaved configuration:",
    CONFIG_PATH
)


# ============================================================
# 21. FREE VALIDATION MEMORY
# ============================================================

del candidate_df
del grouped_candidates
del train_s1_val
del train_s2_val
del train_s3_val
del s2_by_id
del s3_by_id
del val_idx_s2
del val_idx_s3

gc.collect()

print("\nValidation memory released.")


# ============================================================
# 22. LOAD TEST DATA
# ============================================================

def load_core(
    path,
    limit=None
):

    kwargs = {
        "sep": "\t",
        "usecols": CORE_COLS,
        "dtype": str
    }

    if limit is not None:
        kwargs["nrows"] = limit

    df = pd.read_csv(
        path,
        **kwargs
    ).fillna("")

    return prepare_dataframe(
        df
    )


print("\nLoading test data...")

if FAST_TEST_MODE:

    test_s1 = load_core(
        FILES["test_s1"],
        DEV_TEST_S1
    )

    test_s2 = load_core(
        FILES["test_s2"],
        DEV_TEST_S2
    )

    test_s3 = load_core(
        FILES["test_s3"],
        DEV_TEST_S3
    )

else:

    test_s1 = load_core(
        FILES["test_s1"]
    )

    test_s2 = load_core(
        FILES["test_s2"]
    )

    test_s3 = load_core(
        FILES["test_s3"]
    )


print(
    "Test S1:",
    f"{len(test_s1):,}"
)

print(
    "Test S2:",
    f"{len(test_s2):,}"
)

print(
    "Test S3:",
    f"{len(test_s3):,}"
)


# ============================================================
# 23. TEST INDEXES
# ============================================================

test_idx_s2 = make_indexes(
    test_s2
)

test_idx_s3 = make_indexes(
    test_s3
)

test_s2_by_id = (
    test_s2
    .set_index("entity_id")
    .to_dict("index")
)

test_s3_by_id = (
    test_s3
    .set_index("entity_id")
    .to_dict("index")
)

print(
    "Test indexes ready."
)


# ============================================================
# 24. LOAD OPTIMIZED THRESHOLDS
# ============================================================

optimization_config = joblib.load(
    CONFIG_PATH
)

BEST_NAME_THRESH = (
    optimization_config[
        "name_threshold"
    ]
)

BEST_ADDR_THRESH = (
    optimization_config[
        "address_threshold"
    ]
)

print(
    "\nFinal thresholds:"
)

print(
    "Name:",
    BEST_NAME_THRESH
)

print(
    "Address:",
    BEST_ADDR_THRESH
)


# ============================================================
# 25. FINAL TEST MATCHING
# ============================================================

print(
    "\nStarting final test matching..."
)

test_predictions = {}

candidate_rows = []
matching_rows = []

start = time.time()

for row in test_s1.itertuples(
    index=False
):

    row_dict = row._asdict()

    s1_id = row.entity_id

    candidates = set()

    candidates.update(
        candidates_for_row(
            row_dict,
            test_idx_s2
        )
    )

    candidates.update(
        candidates_for_row(
            row_dict,
            test_idx_s3
        )
    )

    matched = []

    for candidate_id in candidates:

        if candidate_id in test_s2_by_id:

            candidate = test_s2_by_id[
                candidate_id
            ]

        elif candidate_id in test_s3_by_id:

            candidate = test_s3_by_id[
                candidate_id
            ]

        else:
            continue

        features = pair_features(
            row_dict,
            candidate
        )

        if optimized_match(
            features,
            BEST_NAME_THRESH,
            BEST_ADDR_THRESH
        ):

            matched.append(
                candidate_id
            )

    matched = sorted(
        set(matched)
    )

    test_predictions[
        s1_id
    ] = matched

    matching_rows.append({
        "source1_entity_id":
            s1_id,

        "matched_entity_ids":
            ",".join(matched)
    })

    candidate_rows.append({
        "source1_entity_id":
            s1_id,

        "candidate_entity_ids":
            ",".join(
                sorted(candidates)
            )
    })


elapsed = time.time() - start

matching_df = pd.DataFrame(
    matching_rows
)

candidate_df = pd.DataFrame(
    candidate_rows
)

print(
    "\nFinal matching completed."
)

print(
    "Time:",
    round(
        elapsed,
        2
    ),
    "seconds"
)

print(
    "S1 entities:",
    f"{len(matching_df):,}"
)


# ============================================================
# 26. FINAL DIAGNOSTICS
# ============================================================

match_counts = (
    matching_df[
        "matched_entity_ids"
    ]
    .map(
        lambda x:
        len(x.split(","))
        if x else 0
    )
)

candidate_counts = (
    candidate_df[
        "candidate_entity_ids"
    ]
    .map(
        lambda x:
        len(x.split(","))
        if x else 0
    )
)

print("\n" + "=" * 70)
print("FINAL TEST DIAGNOSTICS")
print("=" * 70)

print(
    "S1 entities:",
    f"{len(matching_df):,}"
)

print(
    "Predicted links:",
    f"{int(match_counts.sum()):,}"
)

print(
    "Predicted singletons:",
    f"{int((match_counts == 0).sum()):,}"
)

print(
    "Average matches/S1:",
    round(
        match_counts.mean(),
        4
    )
)

print(
    "Maximum matches/S1:",
    int(
        match_counts.max()
    )
)

print(
    "Average candidates/S1:",
    round(
        candidate_counts.mean(),
        4
    )
)

print(
    "Maximum candidates/S1:",
    int(
        candidate_counts.max()
    )
)


# ============================================================
# 27. SAVE OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "/content/kaggle/working/output"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

matching_path = (
    OUTPUT_DIR /
    "matching_results_optimized.tsv"
)

candidate_path = (
    OUTPUT_DIR /
    "candidate_pairs_optimized.tsv"
)

matching_df.to_csv(
    matching_path,
    sep="\t",
    index=False
)

candidate_df.to_csv(
    candidate_path,
    sep="\t",
    index=False
)

print("\nFiles saved:")

print(
    matching_path
)

print(
    candidate_path
)


# ============================================================
# 28. PREVIEW OUTPUT
# ============================================================

print("\nMatching output preview:")

display(
    matching_df.head(20)
)

print(
    "\nCandidate output preview:"
)

display(
    candidate_df.head(20)
)


# ============================================================
# 29. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("AMAZON ENTITY RESOLUTION - COMPLETE")
print("=" * 70)

print(
    "Validation candidate recall:",
    round(
        optimization_config[
            "candidate_recall"
        ],
        6
    )
)

print(
    "Baseline validation Macro F0.5:",
    round(
        optimization_config[
            "baseline_macro_f05"
        ],
        6
    )
)

print(
    "Optimized validation Macro F0.5:",
    round(
        optimization_config[
            "validation_macro_f05"
        ],
        6
    )
)

print(
    "Best name threshold:",
    BEST_NAME_THRESH
)

print(
    "Best address threshold:",
    BEST_ADDR_THRESH
)

print(
    "Test entities processed:",
    f"{len(test_s1):,}"
)

print(
    "Output:",
    matching_path
)

print("=" * 70)


# ============================================================
# 30. IMPORTANT
# ============================================================
#
# For the FINAL full-data run:
#
# FAST_TEST_MODE = False
#
# Then restart/run the notebook from the beginning.
#
# This will process the full test_source1.tsv,
# test_source2.tsv and test_source3.tsv.
#
# The validation Macro F0.5 is the meaningful measured
# entity-resolution metric here. It is not valid to claim
# 100% unseen-test accuracy without ground-truth labels.
# ============================================================
