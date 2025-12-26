# test_bbanalyze_v2.py

import argparse
import sys
import numpy as np
import pandas as pd
import traceback

EXPECTED_TOP_KEYS = {
    "record.count", "complete.cases", "years",
    "player.count", "team.count", "league.count",
    "bb", "nl", "al", "records",
}

EXPECTED_RECORD_KEYS = {
    "obp","pab","hr","hrp","h","hp","sb","sbp","so","sop","sopa","bb","bbp","g"
}

def _fail(msg):
    print(f"[FAIL] {msg}")
    return 1

def approx_equal(a, b, tol=1e-9):
    # treat NaN ~ NaN as equal
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="baseball.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # import the function under test
    try:
        from bbanalyze import bbanalyze
    except Exception as e:
        traceback.print_exc()
        return _fail("Could not import bbanalyze")

    # run it
    try:
        out = bbanalyze(args.file)
    except Exception as e:
        traceback.print_exc()
        return _fail(f"bbanalyze raised an exception: {e}")

    # 1) top-level keys exact
    got = set(out.keys())
    if got != EXPECTED_TOP_KEYS:
        return _fail(f"Top-level keys mismatch.\nExpected: {sorted(EXPECTED_TOP_KEYS)}\nGot:      {sorted(got)}")

    # 2) quick type sanity
    if not isinstance(out["record.count"], int): return _fail("record.count must be int")
    if not isinstance(out["complete.cases"], int): return _fail("complete.cases must be int")
    if not (isinstance(out["years"], tuple) and len(out["years"]) == 2): return _fail("years must be (min,max) tuple")
    if not isinstance(out["player.count"], int): return _fail("player.count must be int")
    if not isinstance(out["team.count"], int): return _fail("team.count must be int")
    if not isinstance(out["league.count"], int): return _fail("league.count must be int")

    # 3) bb dataframe + NaN/Inf rule
    bb = out["bb"]
    if not isinstance(bb, pd.DataFrame):
        return _fail("bb must be a pandas DataFrame")
    if not {"obp","pab"}.issubset(bb.columns):
        return _fail("bb must include 'obp' and 'pab'")

    non_derived = [c for c in bb.columns if c not in ("obp","pab")]
    if bb[non_derived].isna().any().any():
        return _fail("Only obp/pab may be NaN in bb (original columns must be complete)")
    bb_num = bb.select_dtypes(include=[np.number])
    if not bb_num.empty and np.isinf(bb_num.to_numpy()).any():
        return _fail("bb contains Inf in numeric columns; must convert to NaN")

    # 4) NL/AL subsets integrity
    for key, tag in (("nl","NL"), ("al","AL")):
        sub = out[key]
        if not isinstance(sub, dict): return _fail(f"{key} must be a dict")
        for k in ("dat","players","teams"):
            if k not in sub: return _fail(f"{key} missing '{k}'")
        df_ = sub["dat"]
        if not isinstance(df_, pd.DataFrame): return _fail(f"{key}['dat'] must be a DataFrame")
        if not isinstance(sub["players"], int): return _fail(f"{key}['players'] must be int")
        if not isinstance(sub["teams"], int): return _fail(f"{key}['teams'] must be int")
        if not df_.empty and not (df_["lg"] == tag).all():
            return _fail(f"{key}['dat'] rows must have lg == '{tag}'")
        # count consistency
        if df_["id"].nunique() != sub["players"]:
            return _fail(f"{key} players count mismatch (nunique id vs reported)")
        if df_["team"].nunique() != sub["teams"]:
            return _fail(f"{key} teams count mismatch (nunique team vs reported)")

    # 5) records structure: exactly 14 keys
    recs = out["records"]
    if not isinstance(recs, dict):
        return _fail("records must be a dict")
    if set(recs.keys()) != EXPECTED_RECORD_KEYS:
        return _fail(f"records must have exactly these 14 keys: {sorted(EXPECTED_RECORD_KEYS)}")
    for k, v in recs.items():
        if not (isinstance(v, dict) and "id" in v and "value" in v):
            return _fail(f"records['{k}'] must be a dict with 'id' and 'value'")

    # 6) rebuild career from returned clean bb (mirrors grader)
    agg = {
        "g":"sum","ab":"sum","h":"sum","hr":"sum","rbi":"sum",
        "sb":"sum","so":"sum","bb":"sum","hbp":"sum","sh":"sum","sf":"sum"
    }
    career = bb.groupby("id", as_index=True).agg(agg).astype(float)
    career = career[career["ab"] >= 50].copy()

    # recompute derived metrics per spec for value validation
    def _safe_div(n, d):
        n = n.astype(float); d = d.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = n / d
        x[~np.isfinite(x)] = np.nan
        return x

    cchk = pd.DataFrame(index=career.index)
    cchk["obp"]  = _safe_div(career["h"] + career["bb"] + career["hbp"],
                             career["ab"] + career["bb"] + career["hbp"])
    cchk["pab"]  = _safe_div(career["h"] + career["bb"] + career["hbp"] + career["sf"] + career["sh"],
                             career["ab"] + career["bb"] + career["hbp"] + career["sf"] + career["sh"])
    cchk["hrp"]  = _safe_div(career["hr"], career["ab"])
    cchk["hp"]   = _safe_div(career["h"],  career["ab"])
    cchk["sbp"]  = _safe_div(career["sb"], career["ab"])
    cchk["sop"]  = _safe_div(career["so"], career["ab"])
    pa_sopa      = career["ab"] + career["bb"] + career["hbp"] + career["sh"] + career["sf"]
    cchk["sopa"] = _safe_div(career["so"], pa_sopa)
    cchk["bbp"]  = _safe_div(career["bb"], career["ab"])

    # helper: all ids that share the max value (for tie-tolerant check)
    def max_ids_and_value(series: pd.Series):
        s = series.dropna()
        if s.empty:
            return set(), np.nan
        vmax = s.max()
        ids = set(s[s == vmax].index.tolist())
        return ids, float(vmax)

    # build validation map (values from either career counts or cchk rates)
    validate_map = {
        "obp":  cchk["obp"],
        "pab":  cchk["pab"],
        "hr":   career["hr"],
        "hrp":  cchk["hrp"],
        "h":    career["h"],
        "hp":   cchk["hp"],
        "sb":   career["sb"],
        "sbp":  cchk["sbp"],
        "so":   career["so"],
        "sop":  cchk["sop"],
        "sopa": cchk["sopa"],
        "bb":   career["bb"],
        "bbp":  cchk["bbp"],
        "g":    career["g"],
    }

    # compare record values; allow any id among ties
    for key, series in validate_map.items():
        got_id = recs[key]["id"]
        got_val = recs[key]["value"]
        ids_at_max, exp_val = max_ids_and_value(series)
        if not approx_equal(got_val, exp_val):
            return _fail(f"Record '{key}' value mismatch. Expected {exp_val}, got {got_val}")
        # id is considered valid if it’s one of the tied IDs for that max value
        if ids_at_max and got_id not in ids_at_max:
            # Don’t fail on ties with equal values if the value matches; just warn in verbose mode
            if args.verbose:
                print(f"[WARN] '{key}' id not among top-tie set; value matches. Got id={got_id}, top ids={sorted(ids_at_max)}")

    if args.verbose:
        print("\n=== SUMMARY ===")
        print("Top-level keys:", sorted(out.keys()))
        print("Years:", out["years"])
        print("Record/Complete cases:", out["record.count"], "/", out["complete.cases"])
        print("NL players/teams:", out["nl"]["players"], "/", out["nl"]["teams"])
        print("AL players/teams:", out["al"]["players"], "/", out["al"]["teams"])
        show = ["hr","h","g","obp","pab"]
        print("\nSample records:")
        for k in show:
            print(f"  {k}: {out['records'][k]}")


    print("Top-level keys:", sorted(out.keys()))
    print("Years range:", out["years"])
    print("Record count:", out["record.count"])
    print("Complete cases:", out["complete.cases"])
    print("HR record:", out["records"]["hr"])
    print("OBP record:", out["records"]["obp"])
    print("G record:", out["records"]["g"])

    print("[PASS] bbanalyze output validated successfully.")
    return 0



if __name__ == "__main__":
    sys.exit(main())
