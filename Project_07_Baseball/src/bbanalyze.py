# bbanalyze.py
# Project 7 – Manipulating Data
# Function: bbanalyze(filename="baseball.csv")

import pandas as pd
import numpy as np

def bbanalyze(filename: str = "baseball.csv"):
    """
    Reads the baseball dataset and returns a dictionary exactly as specified

    Special note:
    -exact top-level keys with dots
    - Only obp and pab may contain NaN values (due to 0 denominators)
    - All other columns in bb must contain no NaNs
    - Inf values are replaced with NaN
    -14 career records from players with >=50 career AB
    """

    # ---------- 1. Read dataset ----------
    df = pd.read_csv(filename)

    #drop accidental index columns created by to_csv
    bad_idx_cols = df.columns[
        df.columns.astype(str).str.match(r"^Unnamed:") | (df.columns.astype(str).str.lower() == "index")]
    if len(bad_idx_cols) > 0:
        df = df.drop(columns=list(bad_idx_cols))


    # ---------- 2. Summary information from raw dataset ----------
    record_count   = len(df)
    complete_cases = df.dropna().shape[0]
    years          = (int(df["year"].min()), int(df["year"].max()))
    player_count   = df["id"].nunique()
    team_count     = df["team"].nunique()

    # league.count – ignore empty strings and NAs
    lg_ser = df["lg"].astype(str).replace("", np.nan)
    league_count = lg_ser.dropna().nunique()

    # ---------- 3. Work only with complete cases ----------
    # no missing values allowed in original columns
    bb = df.dropna().copy() # no NoNs allowed in original columns

    #capture original column order Before adding derived columns
    _orig_cols = bb.columns.tolist()


    # ---------- 4. Safe division helper ----------
    def _safe_div(num, den):
        num = pd.to_numeric(num, errors="coerce").astype(float)
        den = pd.to_numeric(den, errors="coerce").astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
           out = num / den
        out[~np.isfinite(out)] = np.nan   # convert Inf or invalid to NaN
        return out

    # ---------- 5. Add derived columns (obp, pab) ----------
    # OBP = (H + BB + HBP) / (AB + BB + HBP)
    bb["obp"] = _safe_div(bb["h"] + bb["bb"] + bb["hbp"],
                          bb["ab"] + bb["bb"] + bb["hbp"] )
    # PAB = (H + BB + HBP + SF + SH) / (AB + BB + HBP + SF + SH)
    bb["pab"] = _safe_div(
        bb["h"] + bb["bb"] + bb["hbp"] + bb["sf"] + bb["sh"],
        bb["ab"] + bb["bb"] + bb["hbp"] + bb["sf"] + bb["sh"]
    )
    bb = bb[_orig_cols + ["obp", "pab"]]

    # (Only obp/pab can have NaN now)
    # Verify all other columns contain no NaNs
    non_derived_cols = [c for c in _orig_cols] # only original columns
    assert not bb[non_derived_cols].isna().any().any(), \
        "Only obp/pab may contain NaN in bb; original columns must be complete."

    # ---------- 6. Create NL and AL subsets ----------
    nl_dat = bb.loc[bb["lg"] == "NL"].copy()
    al_dat = bb.loc[bb["lg"] == "AL"].copy()


    nl = {
        "dat": nl_dat,
        "players": nl_dat["id"].nunique(),
        "teams": nl_dat["team"].nunique(),
    }
    al = {
        "dat": al_dat,
        "players": al_dat["id"].nunique(),
        "teams": al_dat["team"].nunique(),
    }

    # ---------- 7. Compute career totals (group by player id) ----------
    agg_cols = {
        "g": "sum", "ab": "sum", "h": "sum", "hr": "sum", "rbi": "sum",
        "sb": "sum", "so": "sum", "bb": "sum", "hbp": "sum", "sh": "sum", "sf": "sum"
    }
    career = bb.groupby("id", as_index=True).agg(agg_cols).astype(float)

    # only players with 50 or more career AB
    career = career[career["ab"] >= 50].copy()

    # ---------- 8) Career rate metrics (formulas per spec) ----------
    # obp = (H + BB + HBP) / (AB + BB + HBP)
    career["obp"] = _safe_div(
        career["h"] + career["bb"] + career["hbp"],
        career["ab"] + career["bb"] + career["hbp"]
    )

    # pab = (H + BB + HBP + SF + SH) / (AB + BB + HBP + SF + SH)
    career["pab"] = _safe_div(
        career["h"] + career["bb"] + career["hbp"] + career["sf"] + career["sh"],
        career["ab"] + career["bb"] + career["hbp"] + career["sf"] + career["sh"]
    )

    # other rates for records
    career["hrp"] = _safe_div(career["hr"], career["ab"])  # HR / AB
    career["hp"] = _safe_div(career["h"], career["ab"])  # H  / AB
    career["sbp"] = _safe_div(career["sb"], career["ab"])  # SB / AB
    career["sop"] = _safe_div(career["so"], career["ab"])  # SO / AB

    # sopa denominator = plate appearances: AB + BB + HBP + SH + SF
    _pa = career["ab"] + career["bb"] + career["hbp"] + career["sh"] + career["sf"]
    career["sopa"] = _safe_div(career["so"], _pa)

    career["bbp"] = _safe_div(career["bb"], career["ab"])  # BB / AB

    # Clean any residual infs (defensive; _safe_div should already NaN them)
    career.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ---------- 9) Stable record finder (deterministic tie-break) ----------
    def _stable_record(df_: pd.DataFrame, metric: str):
        """
        Return {"id": player_id, "value": metric_value} from df_ for the given metric.
        - Ignores NaNs
        - Picks highest metric; if ties, chooses smallest id (deterministic)
        """
        if metric not in df_:
            return {"id": None, "value": np.nan}
        tmp = df_.reset_index()[["id", metric]].dropna(subset=[metric])
        if tmp.empty:
            return {"id": None, "value": np.nan}
        tmp = tmp.sort_values([metric, "id"], ascending=[False, True])
        top = tmp.iloc[0]
        return {"id": top["id"], "value": float(top[metric])}

    # ---------- 10) Records (exactly 14) ----------
    records = {
        "obp": _stable_record(career, "obp"),
        "pab": _stable_record(career, "pab"),
        "hr": _stable_record(career, "hr"),  # most homeruns (count)
        "hrp": _stable_record(career, "hrp"),  # HR / AB
        "h": _stable_record(career, "h"),  # most hits (count)
        "hp": _stable_record(career, "hp"),  # H / AB
        "sb": _stable_record(career, "sb"),  # most stolen bases (count)
        "sbp": _stable_record(career, "sbp"),  # SB / AB
        "so": _stable_record(career, "so"),  # most strikeouts (count)
        "sop": _stable_record(career, "sop"),  # SO / AB
        "sopa": _stable_record(career, "sopa"),  # SO / (AB+BB+HBP+SH+SF)
        "bb": _stable_record(career, "bb"),  # most walks (count)
        "bbp": _stable_record(career, "bbp"),  # BB / AB
        "g": _stable_record(career, "g"),  # most game appearances (sum)
    }

    # ---------- 11) Assemble return dict with exact key names ----------
    result = {
        "record.count": record_count,
        "complete.cases": complete_cases,
        "years": years,
        "player.count": player_count,
        "team.count": team_count,
        "league.count": league_count,
        "bb": bb,
        "nl": {
            "dat": bb.loc[bb["lg"] == "NL"].copy(),
            "players": int(bb.loc[bb["lg"] == "NL", "id"].nunique()),
            "teams": int(bb.loc[bb["lg"] == "NL", "team"].nunique()),
        },
        "al": {
            "dat": bb.loc[bb["lg"] == "AL"].copy(),
            "players": int(bb.loc[bb["lg"] == "AL", "id"].nunique()),
            "teams": int(bb.loc[bb["lg"] == "AL", "team"].nunique()),
        },
        "records": records,
    }


    return result
