import re
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz

LEGAL_SUFFIXES = {
    "incorporated", "inc", "corporation", "corp", "limited", "ltd", "llc",
    "llp", "private", "pvt", "company", "co", "limitedliabilitycompany"
}

def normalize_text(x):
    if pd.isna(x): return ""
    x = str(x).lower().replace("&", " and ")
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def normalize_name(x):
    s = normalize_text(x)
    tokens = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def normalize_address(x):
    return normalize_text(x)

def compact(x):
    return re.sub(r"\s+", "", str(x))

def first_token(x):
    parts = str(x).split()
    return parts[0] if parts else ""

CORE_COLS = ["entity_id", "business_name", "business_address", "country"]

def prepare_dataframe(df):
    out = df[CORE_COLS].copy().fillna("")
    out["norm_name"] = out["business_name"].map(normalize_name)
    out["norm_address"] = out["business_address"].map(normalize_address)
    out["compact_name"] = out["norm_name"].map(compact)
    out["compact_address"] = out["norm_address"].map(compact)
    out["country_name_key"] = out["country"].astype(str) + "|" + out["compact_name"]
    out["country_addr_key"] = out["country"].astype(str) + "|" + out["compact_address"]
    out["country_name_addr_key"] = out["country"].astype(str) + "|" + out["compact_name"] + "|" + out["compact_address"]
    out["country_name_prefix_key"] = out["country"].astype(str) + "|" + out["compact_name"].str[:6]
    out["country_addr_prefix_key"] = out["country"].astype(str) + "|" + out["compact_address"].str[:8]
    out["country_name_first_token_key"] = out["country"].astype(str) + "|" + out["norm_name"].map(first_token)
    return out

BLOCK_COLUMNS = [
    "country_name_key", "country_addr_key", "country_name_addr_key",
    "country_name_prefix_key", "country_addr_prefix_key", "country_name_first_token_key"
]

def candidates_for_row(row_dict, index_dict, max_candidates=500):
    candidates = set()
    for column in BLOCK_COLUMNS:
        key = row_dict.get(column)
        if not key: continue
        values = index_dict.get(column, {}).get(key, [])
        candidates.update(values)
        if len(candidates) >= max_candidates: break
    return candidates

def jaccard_tokens(a, b):
    A = set(str(a).split())
    B = set(str(b).split())
    if not A or not B: return 0.0
    return len(A & B) / len(A | B)

def pair_features(a, b):
    name1 = str(a.get("norm_name", ""))
    name2 = str(b.get("norm_name", ""))
    addr1 = str(a.get("norm_address", ""))
    addr2 = str(b.get("norm_address", ""))
    compact_name1 = str(a.get("compact_name", compact(name1)))
    compact_name2 = str(b.get("compact_name", compact(name2)))
    compact_addr1 = str(a.get("compact_address", compact(addr1)))
    compact_addr2 = str(b.get("compact_address", compact(addr2)))
    country1 = str(a.get("country", "")).strip().lower()
    country2 = str(b.get("country", "")).strip().lower()

    name_exact = int(compact_name1 != "" and compact_name1 == compact_name2)
    addr_exact = int(compact_addr1 != "" and compact_addr1 == compact_addr2)
    country_exact = int(country1 != "" and country1 == country2)

    name_jaccard = jaccard_tokens(name1, name2)
    addr_jaccard = jaccard_tokens(addr1, addr2)
    
    name_fuzzy = fuzz.token_set_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    addr_fuzzy = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    
    name_ratio = fuzz.ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    addr_ratio = fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0

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

def optimized_match(f, name_threshold=0.8, address_threshold=0.8):
    if f["name_exact"] and f["addr_exact"]: return True
    if f["name_exact"] and f["addr_fuzzy"] >= address_threshold: return True
    if f["addr_exact"] and f["name_fuzzy"] >= name_threshold: return True
    if f["name_jaccard"] >= name_threshold and f["addr_jaccard"] >= address_threshold: return True
    if f["name_fuzzy"] >= 0.97 and f["addr_fuzzy"] >= 0.90 and f["country_exact"]: return True
    if f["addr_fuzzy"] >= 0.97 and f["name_fuzzy"] >= 0.90 and f["country_exact"]: return True
    return False

# In-memory storage for S2 and S3 for the API
class MatchingService:
    def __init__(self):
        self.s2_df = pd.DataFrame()
        self.s3_df = pd.DataFrame()
        self.idx_s2 = {}
        self.idx_s3 = {}
        self.s2_by_id = {}
        self.s3_by_id = {}
        self.is_ready = False

    def load_data(self, s2_path, s3_path):
        # We simulate loading data here, skipping it for now since we don't have the TSV files
        pass

service = MatchingService()
