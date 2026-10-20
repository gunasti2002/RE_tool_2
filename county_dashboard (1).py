"""
US County Dashboard
-------------------
Maps every US county and ranks them on four measures: population, violent crime
rate, education (bachelor's degree or higher) and median household income. Each
measure can be shown as its current level, its 5-year change, or its 1-year
change. Counties can be filtered (state, population, education, income) and
grouped into states.

Data sources (downloaded automatically on first run, then cached in ./data):
  * Population         - Census Population Estimates Program, Vintage 2025
                         (CO-EST2025-alldata: July 1 estimates 2020-2025)
  * Median income      - ACS 5-year estimates, table B19013
  * Education          - ACS 5-year estimates, table B15003 (age 25+)
                         (ACS releases 2015-2019, 2019-2023 and 2020-2024)
  * Violent crime      - FBI Crime Data Explorer, monthly violent-crime counts
                         reported by each police agency, summed to counties
  * County shapes      - Census cartographic boundary file, 2024, 1:20m

Run:
    pip install -r requirements.txt
    python3 county_dashboard.py --html      # builds county_dashboard.html and opens it
    streamlit run county_dashboard.py        # same dashboard inside Streamlit
    python3 county_dashboard.py --export     # one merged CSV, one row per county

The first run downloads FBI data for roughly 17,000 police agencies, which can
take 20-40 minutes. It is saved state by state in data/fbi, so later runs are
instant and an interrupted download resumes where it stopped.

Optional: set a free Census API key (https://api.census.gov/data/key_signup.html)
in the environment as CENSUS_API_KEY to avoid rate limits.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import io
import json
import os
import re
import sys
import threading
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
FBI_DIR = DATA_DIR / "fbi"

PEP_URLS = [
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv",
    # Fallback to the prior vintage if the 2025 file moves
    "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/totals/co-est2024-alldata.csv",
]
ACS_LATEST = 2024  # ACS 5-year 2020-2024. Bump to 2025 once the 2021-2025 release is out (Dec 2026).
CHR_URLS = [
    "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2026.csv",
    "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2025.csv",
    "https://www.countyhealthrankings.org/sites/default/files/media/document/analytic_data2024.csv",
]
SHAPE_URL = "https://www2.census.gov/geo/tiger/GENZ2024/shp/cb_2024_us_county_20m.zip"

HEADERS = {"User-Agent": "county-dashboard/2.0 (research use)"}
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------
def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=120, **kw)
    r.raise_for_status()
    return r


def _first_working(urls: list[str]) -> tuple[str, bytes]:
    errors = []
    for u in urls:
        try:
            return u, _get(u).content
        except Exception as e:  # noqa: BLE001
            errors.append(f"{u}: {e}")
    raise RuntimeError("All sources failed:\n" + "\n".join(errors))


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


# --------------------------------------------------------------------------
# 1. Population (Census PEP)
# --------------------------------------------------------------------------
def parse_pep(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(_decode(raw)), dtype={"STATE": str, "COUNTY": str})
    df = df[pd.to_numeric(df["SUMLEV"], errors="coerce") == 50].copy()
    st = df["STATE"].str.zfill(2)
    years = sorted(int(c[-4:]) for c in df.columns if re.fullmatch(r"POPESTIMATE\d{4}", c))
    out = pd.DataFrame({
        "fips": st + df["COUNTY"].str.zfill(3),
        "county": df["CTYNAME"],
        "state": df["STNAME"],
        "abbr": st.map(STATE_ABBR),
    })
    for y in years:
        out[f"pop_{y}"] = pd.to_numeric(df[f"POPESTIMATE{y}"], errors="coerce")
    out = out[out["abbr"].notna()].drop_duplicates("fips").reset_index(drop=True)
    out.attrs["years"] = years
    return out


def load_population() -> pd.DataFrame:
    cache = DATA_DIR / "pep.csv"
    if not cache.exists():
        _, raw = _first_working(PEP_URLS)
        cache.write_bytes(raw)
    return parse_pep(cache.read_bytes())


# --------------------------------------------------------------------------
# 2. ACS: median income + education, three releases
# --------------------------------------------------------------------------
ACS_VARS = {
    "B19013_001E": "median_income",
    "B15003_001E": "adults25",
    "B15003_022E": "edu_ba",
    "B15003_023E": "edu_ma",
    "B15003_024E": "edu_prof",
    "B15003_025E": "edu_phd",
}


def acs_label(year: int) -> str:
    """ACS 5-year release ending in `year`, e.g. 2024 -> '2020-24'."""
    return f"{year - 4}–{str(year)[-2:]}"


def parse_acs(rows: list[list[str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
    for code in ACS_VARS:
        df[code] = pd.to_numeric(df[code], errors="coerce")
        df.loc[df[code] < 0, code] = np.nan          # Census sentinel values (-666666666 etc.)
    df = df.rename(columns=ACS_VARS)
    df["bachelors_pct"] = (df[["edu_ba", "edu_ma", "edu_prof", "edu_phd"]].sum(axis=1, min_count=4)
                           / df["adults25"]) * 100
    return df[["fips", "median_income", "bachelors_pct", "adults25"]].drop_duplicates("fips")


def _fetch_acs_json(year: int) -> list:
    key = os.environ.get("CENSUS_API_KEY", "").strip()
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    attempts = []
    for use_key in ([True, False] if key else [False]):
        params = {"get": ",".join(ACS_VARS), "for": "county:*", "in": "state:*"}
        if use_key:
            params["key"] = key
        try:
            r = requests.get(url, params=params, headers=BROWSER_UA, timeout=120)
            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 100:
                        return data
                except ValueError:
                    pass
            snippet = " ".join(r.text.split())[:200]
            attempts.append(f"ACS {year} (key={'yes' if use_key else 'no'}): "
                            f"HTTP {r.status_code} -> {snippet or '[empty response]'}")
        except requests.RequestException as e:
            attempts.append(f"ACS {year}: {e}")
    raise RuntimeError(
        f"Could not download ACS {year} income/education data from api.census.gov.\n"
        + "\n".join(attempts)
        + "\n\nFix: get a free key at https://api.census.gov/data/key_signup.html , then run\n"
          "  export CENSUS_API_KEY=your_key_here\nin the terminal before running this again."
    )


def load_acs_year(year: int) -> pd.DataFrame:
    cache = DATA_DIR / f"acs_{year}.json"
    legacy = DATA_DIR / "acs.json"               # written by earlier versions (2020-24 release)
    if not cache.exists() and year == ACS_LATEST and legacy.exists():
        cache.write_text(legacy.read_text())
    if not cache.exists():
        cache.write_text(json.dumps(_fetch_acs_json(year)))
    return parse_acs(json.loads(cache.read_text()))


def acs_years() -> tuple[int, int, int]:
    """(5-years-earlier release, previous release, latest release)."""
    return ACS_LATEST - 5, ACS_LATEST - 1, ACS_LATEST


def load_acs_all(log=_log) -> tuple[dict[int, pd.DataFrame], list[str]]:
    out, problems = {}, []
    for y in sorted(acs_years(), reverse=True):
        log(f"    ACS {acs_label(y)}...")
        try:
            out[y] = load_acs_year(y)
        except Exception as e:  # noqa: BLE001
            problems.append(f"ACS {acs_label(y)} income/education data unavailable: {e}")
    return out, problems


# --------------------------------------------------------------------------
# 3. Violent crime: FBI Crime Data Explorer, agency counts summed to counties
# --------------------------------------------------------------------------
CDE_BASE = "https://cde.ucr.cjis.gov/LATEST/"
CDE_UA = "Mozilla/5.0 (compatible; county-dashboard/2.0; personal research)"
CDE_WORKERS = 6            # parallel requests; kept modest to be polite to the FBI server
CDE_MIN_MONTHS = 6         # an agency-year needs at least this many months reported
CRIME_MIN_COVERAGE = 0.5   # county rate shown only if reporting agencies cover >= 50% of residents
# Agencies whose jurisdiction is an area inside one county (or a few). State police,
# tribal, and special-purpose agencies are left out: their records mix statewide
# headquarters with local units and cannot be placed in a county reliably.
CDE_INCLUDED_TYPES = {
    "County", "Parish", "City", "Municipality", "Borough", "City and Borough",
    "Census Area", "University or College",
}

_tls = threading.local()
_retry_lock = threading.Lock()
_retry_count = 0          # how many times the FBI server asked us to wait / failed and we retried


def _cde_session() -> requests.Session:
    s = getattr(_tls, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": CDE_UA, "Accept": "application/json"})
        _tls.session = s
    return s


def cde_get(path: str, params: dict | None = None, retries: int = 4):
    url = CDE_BASE + path
    err: Exception = RuntimeError(f"No response from {url}")
    for attempt in range(retries + 1):
        r = None
        try:
            r = _cde_session().get(url, params=params, timeout=30)
            if r.status_code == 200:
                if not r.content.strip():
                    return {}
                try:
                    return r.json()
                except ValueError:
                    err = RuntimeError(f"FBI server sent a non-JSON page for {url}")
            elif r.status_code in (408, 429, 500, 502, 503, 504):
                err = RuntimeError(f"HTTP {r.status_code} from {url}")
            else:
                raise RuntimeError(f"HTTP {r.status_code} from {r.url}: {' '.join(r.text.split())[:200]}")
        except requests.RequestException as e:
            err = e
        if attempt < retries:
            global _retry_count
            with _retry_lock:
                _retry_count += 1
            wait = min(2 ** attempt, 30)
            ra = r.headers.get("Retry-After", "") if r is not None else ""
            if ra.isdigit():
                wait = min(int(ra), 60)
            time.sleep(wait)
    raise err


def cde_latest_year() -> int:
    """Last calendar year the FBI has complete data for (from its max_data_date)."""
    cache = FBI_DIR / "latest_year.json"
    if cache.exists():
        return int(json.loads(cache.read_text())["latest"])
    probe = dt.date.today().year - 2
    resp = cde_get("summarized/national/V",
                   {"from": f"01-{probe}", "to": f"12-{probe}", "type": "counts"})
    latest = probe
    mdd = ((resp or {}).get("cde_properties") or {}).get("max_data_date") or {}
    stamp = mdd.get("UCR") if isinstance(mdd, dict) else None
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", str(stamp or "").strip())
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        latest = year if month == 12 else year - 1
    FBI_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"latest": latest, "max_data_date": stamp,
                                 "checked": dt.date.today().isoformat()}))
    return latest


def summarize_agency_response(resp: dict, state_name: str | None) -> dict:
    """Monthly agency counts -> {year: [count_sum, months_reported, mean_population]}."""
    offenses = (resp or {}).get("offenses") or {}
    actuals = offenses.get("actuals") or offenses.get("counts") or {}
    skip = {"United States Offenses"}
    if state_name:
        skip.add(f"{state_name} Offenses")
    key = next((k for k in actuals if k.endswith(" Offenses") and k not in skip), None)
    if key is None:
        return {}
    label = key[: -len(" Offenses")]
    counts = actuals.get(key) or {}
    pops = (((resp.get("populations") or {}).get("population") or {}).get(label)) or {}
    acc: dict[str, list[float]] = {}
    for period, v in counts.items():
        if v is None or "-" not in period:
            continue
        year = period.split("-")[-1]
        rec = acc.setdefault(year, [0.0, 0, 0.0, 0])
        rec[0] += float(v)
        rec[1] += 1
    for period, p in pops.items():
        if p is None or "-" not in period:
            continue
        year = period.split("-")[-1]
        if year in acc:
            acc[year][2] += float(p)
            acc[year][3] += 1
    return {y: [round(r[0], 2), int(r[1]), round(r[2] / r[3], 1) if r[3] else 0.0]
            for y, r in acc.items()}


def _list_state_agencies(st: str) -> list[dict]:
    resp = cde_get(f"agency/byStateAbbr/{st}") or {}
    seen, out = set(), []
    for group, lst in resp.items():
        for a in lst or []:
            if not isinstance(a, dict) or not a.get("ori") or a["ori"] in seen:
                continue
            if a.get("agency_type_name") not in CDE_INCLUDED_TYPES:
                continue
            seen.add(a["ori"])
            out.append({"ori": a["ori"], "type": a.get("agency_type_name"),
                        "name": a.get("agency_name"), "state_name": a.get("state_name"),
                        "counties": a.get("counties") or group})
    return out


def fetch_state_crime(st: str, y0: int, y1: int, log=_log) -> dict:
    cache = FBI_DIR / f"crime_{st}_{y0}-{y1}.json"
    blob = json.loads(cache.read_text()) if cache.exists() else None
    if blob is None:
        blob = {"agencies": _list_state_agencies(st), "data": {}}
    todo = [a for a in blob["agencies"] if a["ori"] not in blob["data"]]
    if not todo:
        return blob

    def one(a):
        resp = cde_get(f"summarized/agency/{a['ori']}/V",
                       {"from": f"01-{y0}", "to": f"12-{y1}", "type": "counts"})
        return a["ori"], summarize_agency_response(resp, a.get("state_name"))

    failed, done = 0, 0
    start = time.time()
    retries_at_start = _retry_count
    last_note = start
    with cf.ThreadPoolExecutor(max_workers=CDE_WORKERS) as pool:
        pending = {pool.submit(one, a) for a in todo}
        while pending:
            finished, pending = cf.wait(pending, timeout=20, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                done += 1
                try:
                    ori, summary = fut.result()
                    blob["data"][ori] = summary
                except Exception:  # noqa: BLE001  retried on the next run
                    failed += 1
                if done % 50 == 0:
                    cache.write_text(json.dumps(blob))   # saved progress survives Ctrl+C
            now = time.time()
            if now - last_note >= 20 and pending:
                last_note = now
                rate = done / max(now - start, 1)
                eta = (len(todo) - done) / rate / 60 if rate > 0 else None
                msg = f"      {st}: {done:,}/{len(todo):,} agencies"
                if eta is not None:
                    msg += f" (about {max(eta, 0.1):.0f} min left for {st})" if eta >= 1 else " (under a minute left)"
                slow = _retry_count - retries_at_start
                if slow:
                    msg += f"; FBI server is slow or busy, {slow} retries so far"
                log(msg)
    cache.write_text(json.dumps(blob))
    log(f"      {st}: done, {len(todo):,} agencies in {(time.time() - start) / 60:.1f} min")
    if failed:
        log(f"      {st}: {failed} agencies could not be downloaded (will retry next run)")
    return blob


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    s = re.sub(r"\bSAINTE\b", "STE", s)
    s = re.sub(r"\bSAINT\b", "ST", s)
    return re.sub(r"[^A-Z]", "", s)


_CENSUS_SUFFIXES = (" City and Borough", " Census Area", " Municipality", " Borough",
                    " Planning Region", " County", " Parish")


def census_county_key(name: str) -> str:
    n = str(name).strip()
    if n.endswith(" city"):                       # independent city (VA, MD, MO)
        return _norm_name(n[:-5]) + "CITY"
    for suf in _CENSUS_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    return _norm_name(n)


def cde_county_key(name: str) -> str:
    n = re.sub(r"\s+(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY|PLANNING REGION)$", "",
               str(name).strip().upper())
    return _norm_name(n)


def build_county_crime(pop: pd.DataFrame, per_state: dict, latest: int) -> tuple[pd.DataFrame, dict]:
    """Sum agency counts into counties.

    Level: latest-year rate from every agency that reported >= CDE_MIN_MONTHS months.
    Changes: only agencies that reported in BOTH years being compared, so a change
    is never caused by an agency starting or stopping reporting.
    Agencies listed in several counties are split by county population.
    """
    pop_col = f"pop_{max(pop.attrs['years'])}"
    cpop = dict(zip(pop["fips"], pop[pop_col]))
    index: dict[str, dict[str, str]] = defaultdict(dict)
    for f, name, ab in zip(pop["fips"], pop["county"], pop["abbr"]):
        index[ab][census_county_key(name)] = f

    level = defaultdict(lambda: [0.0, 0.0])
    comp = {k: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]) for k in (5, 1)}
    stats = Counter()
    unmatched = Counter()

    def annual(ys, y):
        r = ys.get(str(y))
        if not r or r[1] < CDE_MIN_MONTHS:
            return None
        return r[0] * 12.0 / r[1], float(r[2] or 0.0)

    for st, blob in per_state.items():
        for ag in blob.get("agencies", []):
            ys = blob.get("data", {}).get(ag["ori"])
            if ys is None:
                stats["not_downloaded"] += 1
                continue
            stats["agencies"] += 1
            names = [n for n in re.split(r"[;,]", ag.get("counties") or "") if n.strip()]
            fipses = []
            if st == "DC" and index["DC"]:           # DC agencies are listed without a county
                fipses = list(index["DC"].values())[:1]
            for n in names:
                k = cde_county_key(n)
                f = index[st].get(k) or (index[st].get(k[:-4]) if k.endswith("CITY") else None)
                if f and f not in fipses:
                    fipses.append(f)
            if not fipses:
                stats["unmatched"] += 1
                unmatched[f"{st}: {ag.get('counties')}"] += 1
                continue
            w = [max(float(cpop.get(f) or 0), 0.0) for f in fipses]
            tot = sum(w)
            shares = [x / tot for x in w] if tot > 0 else [1 / len(fipses)] * len(fipses)
            cur = annual(ys, latest)
            if cur is None:
                continue
            stats["reporting_latest"] += 1
            for f, s in zip(fipses, shares):
                level[f][0] += cur[0] * s
                level[f][1] += cur[1] * s
            for k in (5, 1):
                base = annual(ys, latest - k)
                if base is None:
                    continue
                for f, s in zip(fipses, shares):
                    c = comp[k][f]
                    c[0] += cur[0] * s
                    c[1] += cur[1] * s
                    c[2] += base[0] * s
                    c[3] += base[1] * s

    recs = []
    for f in pop["fips"]:
        cp = float(cpop.get(f) or 0)
        cnt, cov_pop = level[f] if f in level else (0.0, 0.0)
        cov = cov_pop / cp if cp > 0 else None
        ok = cov_pop > 0 and cov is not None and cov >= CRIME_MIN_COVERAGE
        rec = {"fips": f,
               "crime_rate": cnt / cov_pop * 1e5 if ok else None,
               "crime_pop": cov_pop if ok else None,
               "crime_cov": cov}
        for k in (5, 1):
            a, ap, b, bp = comp[k][f] if f in comp[k] else (0.0, 0.0, 0.0, 0.0)
            good = ap > 0 and bp > 0 and cp > 0 and ap / cp >= CRIME_MIN_COVERAGE
            for name, v in zip(("a", "ap", "b", "bp"), (a, ap, b, bp)):
                rec[f"c{k}_{name}"] = v if good else None
        recs.append(rec)
    df = pd.DataFrame(recs).drop_duplicates("fips")
    stats["counties_with_rate"] = int(df["crime_rate"].notna().sum())
    meta = {"stats": dict(stats), "unmatched_examples": [k for k, _ in unmatched.most_common(15)]}
    return df, meta


def load_fbi_crime(pop: pd.DataFrame, log=_log) -> tuple[pd.DataFrame, dict]:
    FBI_DIR.mkdir(parents=True, exist_ok=True)
    log("    Checking which years the FBI has...")
    latest = cde_latest_year()
    y0, y1 = latest - 5, latest
    states = sorted(set(pop["abbr"].dropna()))
    fresh = [st for st in states if not (FBI_DIR / f"crime_{st}_{y0}-{y1}.json").exists()]
    if fresh:
        log(f"    Downloading FBI agency data for {y0}-{y1} ({len(fresh)} states to go). "
            "This is the slow part: 20-40 minutes the first time, saved for next time.")
    per_state, failed_states, streak = {}, [], 0
    for i, st in enumerate(states, 1):
        log(f"    FBI crime data: {st} ({i}/{len(states)})")
        try:
            per_state[st] = fetch_state_crime(st, y0, y1, log)
            streak = 0
        except Exception as e:  # noqa: BLE001
            failed_states.append(st)
            streak += 1
            log(f"      {st} failed: {e}")
            if streak >= 3 and not per_state:
                raise RuntimeError(f"Could not reach the FBI Crime Data Explorer: {e}") from e
    if not per_state:
        raise RuntimeError("No FBI crime data could be downloaded.")
    df, meta = build_county_crime(pop, per_state, latest)
    meta.update({"latest": latest, "failed_states": failed_states})
    return df, meta


# ---- County Health Rankings: fallback level only (no yearly series) ----
def parse_chr(raw: bytes) -> pd.DataFrame:
    head = raw[:500].lstrip().lower()
    if head.startswith(b"<") or b"<html" in head:
        raise ValueError("Downloaded a web page, not the CSV data file")
    grid = pd.read_csv(io.BytesIO(raw), header=None, dtype=str, low_memory=False, encoding="latin-1")
    labels = [" | ".join(str(grid.iat[r, c]).lower() for r in range(min(3, len(grid))))
              for c in range(grid.shape[1])]

    def find(*tests):
        for test in tests:
            for i, lab in enumerate(labels):
                if test(lab):
                    return i
        return None

    fips_i = find(lambda l: "fipscode" in l, lambda l: "5-digit fips" in l,
                  lambda l: "fips" in l and "county" not in l and "state" not in l)
    crime_i = find(lambda l: "v043_rawvalue" in l, lambda l: "v043" in l and "raw" in l,
                   lambda l: "violent crime" in l and "raw value" in l)
    if fips_i is None or crime_i is None:
        raise ValueError(f"Could not find FIPS / violent crime columns. First headers: {labels[:6]}")
    out = pd.DataFrame({"fips": grid.iloc[:, fips_i], "crime": grid.iloc[:, crime_i]})
    out = out[out["fips"].str.fullmatch(r"\d{4,5}", na=False)]
    out["fips"] = out["fips"].str.zfill(5)
    out = out[~out["fips"].str.endswith("000")]
    out["violent_crime_rate"] = pd.to_numeric(out["crime"], errors="coerce")
    if out["violent_crime_rate"].notna().sum() < 100:
        raise ValueError("Crime column found but almost empty")
    return out[["fips", "violent_crime_rate"]].reset_index(drop=True)


def load_chr_crime() -> pd.DataFrame:
    cache = DATA_DIR / "chr.csv"
    if cache.exists():
        try:
            return parse_chr(cache.read_bytes())
        except Exception:  # noqa: BLE001
            cache.unlink()
    errors = []
    for url in CHR_URLS:
        try:
            raw = _get(url).content
            df = parse_chr(raw)
            cache.write_bytes(raw)
            return df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    raise RuntimeError("Could not load County Health Rankings crime data:\n" + "\n".join(errors))


# --------------------------------------------------------------------------
# 4. County shapes -> GeoJSON
# --------------------------------------------------------------------------
def _round_coords(coords, nd=4):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], nd), round(coords[1], nd)]
    return [_round_coords(c, nd) for c in coords]


def load_geojson() -> dict:
    cache = DATA_DIR / "counties.geojson"
    if not cache.exists():
        import shapefile  # pyshp

        zpath = DATA_DIR / "counties_shp.zip"
        zpath.write_bytes(_get(SHAPE_URL).content)
        with zipfile.ZipFile(zpath) as z:
            base = next(n[:-4] for n in z.namelist() if n.endswith(".shp"))
            reader = shapefile.Reader(
                shp=io.BytesIO(z.read(base + ".shp")),
                shx=io.BytesIO(z.read(base + ".shx")),
                dbf=io.BytesIO(z.read(base + ".dbf")),
            )
            features = []
            for sr in reader.shapeRecords():
                geom = sr.shape.__geo_interface__
                features.append({
                    "type": "Feature",
                    "id": sr.record["GEOID"],
                    "properties": {"GEOID": sr.record["GEOID"]},
                    "geometry": {"type": geom["type"],
                                 "coordinates": _round_coords(geom["coordinates"])},
                })
        cache.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        zpath.unlink(missing_ok=True)
    return json.loads(cache.read_text())


# --------------------------------------------------------------------------
# Merge everything into one county table
# --------------------------------------------------------------------------
# Column order of each county row sent to the dashboard (kept in sync with the JS `I` map)
ROW_FIELDS = [
    "fips", "county", "state", "abbr",
    "p0", "p1", "p5",                         # population: latest, 1 yr earlier, 5 yrs earlier
    "cL", "cpL", "cov",                       # crime rate latest, population covered, coverage share
    "a5", "ap5", "b5", "bp5",                 # crime 5-yr comparison: counts & pops (latest, base)
    "a1", "ap1", "b1", "bp1",                 # crime 1-yr comparison
    "e0", "e1", "e5", "ea0", "ea1", "ea5",    # bachelor's+ % and adults 25+ for each ACS release
    "i0", "i1", "i5",                         # median household income for each ACS release
]


def _num(v, nd=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return int(round(f)) if nd is None else round(f, nd)


def build_dataset(log=_log) -> tuple[pd.DataFrame, dict]:
    log("  [1/4] Population data...")
    pop = load_population()
    pyears = pop.attrs["years"]
    L = pyears[-1]
    p5y = L - 5 if (L - 5) in pyears else pyears[0]
    df = pd.DataFrame({
        "fips": pop["fips"], "county": pop["county"], "state": pop["state"], "abbr": pop["abbr"],
        "p0": pop[f"pop_{L}"], "p1": pop[f"pop_{L - 1}"], "p5": pop[f"pop_{p5y}"],
    })
    problems: list[str] = []

    log("  [2/4] Income & education data (3 ACS releases)...")
    acs, acs_problems = load_acs_all(log)
    problems += acs_problems
    ay5, ay1, ay0 = acs_years()
    for tag, y in (("0", ay0), ("1", ay1), ("5", ay5)):
        if y in acs:
            a = acs[y].rename(columns={"bachelors_pct": f"e{tag}", "adults25": f"ea{tag}",
                                       "median_income": f"i{tag}"})
            df = df.merge(a, on="fips", how="left")
        else:
            df[f"e{tag}"] = np.nan
            df[f"ea{tag}"] = np.nan
            df[f"i{tag}"] = np.nan

    log("  [3/4] Crime data (FBI Crime Data Explorer)...")
    crime_meta: dict = {"source": None}
    try:
        crime, m = load_fbi_crime(pop, log)
        crime_meta = {"source": "fbi", **m}
        crime = crime.rename(columns={
            "crime_rate": "cL", "crime_pop": "cpL", "crime_cov": "cov",
            "c5_a": "a5", "c5_ap": "ap5", "c5_b": "b5", "c5_bp": "bp5",
            "c1_a": "a1", "c1_ap": "ap1", "c1_b": "b1", "c1_bp": "bp1"})
        df = df.merge(crime, on="fips", how="left")
        if m.get("failed_states"):
            problems.append("FBI crime data could not be downloaded for: "
                            + ", ".join(m["failed_states"]) + ". Run the export again to retry.")
    except Exception as e:  # noqa: BLE001
        log(f"    WARNING: FBI crime data unavailable ({e}). Falling back to County Health Rankings.")
        problems.append("FBI crime data could not be downloaded, so the crime layer shows the "
                        "County Health Rankings rate instead, without 5-year or 1-year changes.")
        try:
            chr_ = load_chr_crime().rename(columns={"violent_crime_rate": "cL"}).drop_duplicates("fips")
            df = df.merge(chr_, on="fips", how="left")
            df["cpL"] = np.where(df["cL"].notna(), df["p0"], np.nan)
            crime_meta = {"source": "chr"}
        except Exception as e2:  # noqa: BLE001
            problems.append(f"No crime data available: {e2}")
            df["cL"] = np.nan
            df["cpL"] = np.nan
    for c in ROW_FIELDS:
        if c not in df.columns:
            df[c] = np.nan

    meta = {
        "pop_years": [p5y, L - 1, L],
        "acs_years": [ay5, ay1, ay0],
        "crime": crime_meta,
        "problems": problems,
    }
    return df[ROW_FIELDS].copy(), meta


def dataset_rows(df: pd.DataFrame) -> list[list]:
    prec = {"cL": 1, "cpL": None, "cov": 3, "a5": 1, "ap5": None, "b5": 1, "bp5": None,
            "a1": 1, "ap1": None, "b1": 1, "bp1": None, "e0": 2, "e1": 2, "e5": 2}
    rows = []
    for rec in df.itertuples(index=False):
        row = []
        for name, v in zip(ROW_FIELDS, rec):
            if name in ("fips", "county", "state", "abbr"):
                row.append(None if pd.isna(v) else str(v))
            else:
                row.append(_num(v, prec.get(name)))
        rows.append(row)
    return rows


def readable_table(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Flat CSV-friendly version of the dataset with the changes computed."""
    def pct(a, b):
        return np.where((b > 0) & a.notna() & b.notna(), (a / b - 1) * 100, np.nan)

    py5, py1, py0 = meta["pop_years"]
    ay5, ay1, ay0 = meta["acs_years"]
    cy = (meta.get("crime") or {}).get("latest")
    out = pd.DataFrame({"fips": df["fips"], "county": df["county"], "state": df["state"]})
    out[f"population_{py0}"] = df["p0"]
    out[f"population_change_{py5}_{py0}_pct"] = pct(df["p0"], df["p5"])
    out[f"population_change_{py1}_{py0}_pct"] = pct(df["p0"], df["p1"])
    tag = cy if cy else "latest"
    out[f"violent_crime_per_100k_{tag}"] = df["cL"]
    out["crime_reporting_coverage"] = df.get("cov")
    if cy:
        for k in (5, 1):
            ra = df[f"a{k}"] / df[f"ap{k}"]
            rb = df[f"b{k}"] / df[f"bp{k}"]
            out[f"violent_crime_rate_change_{cy - k}_{cy}_pct"] = pct(ra, rb)
    out[f"bachelors_pct_{acs_label(ay0)}"] = df["e0"]
    out[f"bachelors_change_{acs_label(ay5)}_to_{acs_label(ay0)}_pts"] = df["e0"] - df["e5"]
    out[f"bachelors_change_{acs_label(ay1)}_to_{acs_label(ay0)}_pts"] = df["e0"] - df["e1"]
    out[f"median_income_{acs_label(ay0)}"] = df["i0"]
    out[f"median_income_change_{acs_label(ay5)}_to_{acs_label(ay0)}_pct"] = pct(df["i0"], df["i5"])
    out[f"median_income_change_{acs_label(ay1)}_to_{acs_label(ay0)}_pct"] = pct(df["i0"], df["i1"])
    out.columns = [c.replace("–", "-") for c in out.columns]
    return out


def sources_and_notes(meta: dict) -> tuple[list[dict], list[str]]:
    py5, py1, py0 = meta["pop_years"]
    ay5, ay1, ay0 = meta["acs_years"]
    crime = meta.get("crime") or {}
    sources = [
        {"name": f"Annual county population estimates, Vintage {py0} (CO-EST{py0}-alldata)",
         "publisher": "U.S. Census Bureau, Population Estimates Program",
         "used": "Population and population change",
         "years": f"July 1, {py5} to July 1, {py0}",
         "url": "https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-total.html"},
        {"name": "American Community Survey 5-year estimates: table B19013 (median household income) "
                 "and table B15003 (educational attainment, age 25+)",
         "publisher": "U.S. Census Bureau, via api.census.gov",
         "used": "Median income, bachelor's degree share, and their changes",
         "years": f"{acs_label(ay5)}, {acs_label(ay1)} and {acs_label(ay0)} releases",
         "url": "https://www.census.gov/data/developers/data-sets/acs-5year.html"},
    ]
    if crime.get("source") == "fbi":
        cy = crime["latest"]
        sources.append({
            "name": "Crime Data Explorer: monthly violent crime offenses reported by each law "
                    "enforcement agency (Summary Reporting System counts)",
            "publisher": "FBI Uniform Crime Reporting (UCR) Program",
            "used": "Violent crime rate and its changes (agency counts summed to counties)",
            "years": f"{cy - 5}, {cy - 1} and {cy}",
            "url": "https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/explorer/crime/crime-trend"})
    elif crime.get("source") == "chr":
        sources.append({
            "name": "County Health Rankings analytic data, violent crime measure (v043)",
            "publisher": "University of Wisconsin Population Health Institute (from FBI UCR data)",
            "used": "Violent crime rate (fallback; no yearly changes)",
            "years": "Multi-year average in the latest release",
            "url": "https://www.countyhealthrankings.org/health-data/methodology-and-sources/data-documentation"})
    sources.append({
        "name": "Cartographic boundary file: counties, 1:20,000,000 (cb_2024_us_county_20m)",
        "publisher": "U.S. Census Bureau",
        "used": "County shapes on the map",
        "years": "2024",
        "url": "https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"})

    notes = [
        "Population change is the percent change in July 1 resident population.",
        f"Education is the share of adults 25+ with a bachelor's degree or higher. Its changes are in "
        f"percentage points (e.g. 25% to 27% = +2.0 pts). The 5-year change compares the "
        f"{acs_label(ay5)} and {acs_label(ay0)} ACS releases, which share no survey years.",
        f"The 1-year changes for education and income compare the {acs_label(ay1)} and "
        f"{acs_label(ay0)} ACS releases. These overlap by four survey years, so they move slowly "
        f"and small counties can show noise.",
        "Median income changes are in nominal dollars (not adjusted for inflation). State figures "
        "for median income are population-weighted averages of county medians.",
    ]
    if crime.get("source") == "fbi":
        st = crime.get("stats", {})
        notes += [
            "Violent crime (murder, rape, robbery, aggravated assault) per 100,000 residents of the "
            "areas covered by reporting agencies. Monthly counts from city police, sheriffs, "
            "county police and campus police are summed to the county. State police, tribal and "
            "special-purpose agencies are excluded because they cannot be placed in one county.",
            f"An agency counts for a year if it reported at least {CDE_MIN_MONTHS} months (partial "
            f"years are scaled to 12 months). A county's rate is shown only if reporting agencies "
            f"cover at least {int(CRIME_MIN_COVERAGE * 100)}% of its residents; otherwise it is "
            "blank. Crime changes use only agencies that reported in both years compared.",
            "Agencies serving several counties are split between them by county population. Rural "
            "areas policed mainly by state police can show low coverage and appear blank.",
            f"{st.get('reporting_latest', 0):,} agencies reported for {crime['latest']}; "
            f"{st.get('counties_with_rate', 0):,} counties have a crime rate.",
        ]
    notes.append("Connecticut replaced its 8 counties with 9 planning regions in Census data from "
                 "2022 on. The older ACS release uses the old counties, so Connecticut's 5-year "
                 "education and income changes are blank.")
    return sources, notes


# --------------------------------------------------------------------------
# Build the shareable HTML
# --------------------------------------------------------------------------
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
# plotly.js draws every map on top of small basemap files that it normally downloads
# when the page opens (newer plotly.js versions read them from /un/). We download the
# matching ones once and embed them, so the shared file also works offline.
# usa_110m draws the national map; world_110m is used when zooming into one state.
TOPO_BASE = {"un": "https://cdn.plot.ly/un/", "legacy": "https://cdn.plot.ly/"}
TOPO_NAMES = ("usa_110m", "world_110m")


def load_topojson(kind: str, name: str) -> dict:
    cache = DATA_DIR / f"{name}_{kind}.json"
    if not cache.exists():
        topo = _get(TOPO_BASE[kind] + name + ".json").json()
        if not (topo.get("objects") or {}):
            raise ValueError(f"unexpected basemap file for {name}")
        cache.write_text(json.dumps(topo))
    return json.loads(cache.read_text())


def plotly_tag_from(js: str | None, topos: dict | None) -> str:
    tag = "<script>" + js + "</script>" if js else PLOTLY_CDN
    if topos:
        parts = ["window.PlotlyGeoAssets=window.PlotlyGeoAssets||{topojson:{}};",
                 "window.PlotlyGeoAssets.topojson=window.PlotlyGeoAssets.topojson||{};"]
        for name, topo in topos.items():
            topo_js = json.dumps(topo, separators=(",", ":")).replace("</", "<\\/")
            parts.append(f"window.PlotlyGeoAssets.topojson['{name}']={topo_js};")
        tag += "<script>" + "".join(parts) + "</script>"
    return tag


def _plotly_tag() -> str:
    try:  # embed plotly.js so the file works offline
        from plotly.offline import get_plotlyjs
        js = get_plotlyjs()
    except Exception:  # noqa: BLE001
        js = None
    kind = "un" if js and "cdn.plot.ly/un/" in js else "legacy"
    topos = {}
    for name in TOPO_NAMES:
        try:
            topos[name] = load_topojson(kind, name)
        except Exception as e:  # noqa: BLE001  map still works online without it
            _log(f"    (Could not embed the {name} basemap: {e}. "
                 + ("The map will need internet to draw.)" if name == "usa_110m" else "Zooming to a state is turned off.)"))
    return plotly_tag_from(js, topos)


def build_html(rows: list[list], geo: dict, meta: dict, plotly_tag: str, generated: str) -> str:
    keep = {r[0] for r in rows}
    slim_geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": f["id"], "properties": {},
         "geometry": {"type": f["geometry"]["type"],
                      "coordinates": _round_coords(f["geometry"]["coordinates"], 3)}}
        for f in geo["features"] if f["id"] in keep
    ]}
    sources, notes = sources_and_notes(meta)
    py5, py1, py0 = meta["pop_years"]
    ay5, ay1, ay0 = meta["acs_years"]
    crime = meta.get("crime") or {}
    cy = crime.get("latest")
    acs_l = {"lvl": acs_label(ay0), "ch5": f"{acs_label(ay5)} to {acs_label(ay0)}",
             "ch1": f"{acs_label(ay1)} to {acs_label(ay0)}"}
    labels = {
        "pop": {"lvl": str(py0), "ch5": f"{py5}–{py0}", "ch1": f"{py1}–{py0}"},
        "crime": ({"lvl": str(cy), "ch5": f"{cy - 5}–{cy}", "ch1": f"{cy - 1}–{cy}"}
                  if cy else {"lvl": "latest", "ch5": "n/a", "ch1": "n/a"}),
        "edu": acs_l, "inc": acs_l,
    }
    payload = {"fields": ROW_FIELDS, "rows": rows, "geo": slim_geo, "labels": labels,
               "problems": meta.get("problems", []), "sources": sources, "notes": notes,
               "generated": generated, "crimeChanges": crime.get("source") == "fbi"}
    data_js = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (HTML_TEMPLATE.replace("__PLOTLY__", plotly_tag)
                         .replace("__DATA__", data_js))


def render_dashboard_html(log=_log) -> str:
    df, meta = build_dataset(log)
    log("  [4/4] County map shapes...")
    geo = load_geojson()
    return build_html(dataset_rows(df), geo, meta, _plotly_tag(),
                      dt.date.today().strftime("%B %d, %Y").replace(" 0", " "))


def export_html(out_path: Path | None = None, log=_log) -> Path:
    out_path = Path(out_path or Path(__file__).resolve().parent / "county_dashboard.html")
    log("Building the dashboard... (first run downloads data and can take a while)")
    html = render_dashboard_html(log)
    log("  Writing HTML file...")
    out_path.write_text(html, encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# Streamlit: shows the exact same dashboard
# --------------------------------------------------------------------------
def run_app() -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(page_title="US County Dashboard", layout="wide")

    @st.cache_data(show_spinner="Building the dashboard. The first run downloads FBI crime data "
                                "and can take 20-40 minutes; after that it loads in seconds.")
    def get_html() -> str:
        return render_dashboard_html(log=lambda m: None)

    components.html(get_html(), height=3300, scrolling=True)


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>US County Dashboard</title>
<style>
:root{
  --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1f; --muted:#6b6b70; --line:#e3e3e0;
  --accent:#2166ac; --pos:#2166ac; --neg:#b2182b; --empty:#d9d9d6; --chip:#f0f0ed; --act:#eef4fb;
}
@media (prefers-color-scheme: dark){
  :root{ --bg:#141416; --panel:#1d1d20; --text:#ececee; --muted:#9a9aa1; --line:#2e2e33;
         --accent:#6aa9e0; --pos:#6aa9e0; --neg:#ef7d7d; --empty:#3a3a3f; --chip:#26262a; --act:#1f2a36; }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 22px}
h2{font-size:19px;margin:34px 0 4px}
.h2sub{color:var(--muted);font-size:14px;margin:0 0 12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}
.controls{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:24px;align-items:end;margin-bottom:14px}
.ctl-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:8px}
input[type=range]{width:100%;accent-color:var(--accent);margin:0}
.ticks{position:relative;height:26px;margin-top:6px}
.ticks button{position:absolute;top:0;white-space:nowrap;background:none;border:0;padding:4px 0;font:inherit;font-size:13px;color:var(--muted);cursor:pointer}
.ticks button:nth-child(1){left:0}
.ticks button:nth-child(2){left:33.33%;transform:translateX(-50%)}
.ticks button:nth-child(3){left:66.67%;transform:translateX(-50%)}
.ticks button:nth-child(4){right:0}
.ticks button.on{color:var(--text);font-weight:600}
.seg{display:flex;background:var(--chip);border-radius:8px;padding:3px}
.seg button{flex:1;border:0;background:none;padding:7px 8px;border-radius:6px;font:inherit;font-size:13px;color:var(--muted);cursor:pointer;white-space:nowrap}
.seg button.on{background:var(--panel);color:var(--text);font-weight:600;box-shadow:0 1px 2px rgba(0,0,0,.12)}
select,input[type=search],input[type=number]{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--text);font:inherit;font-size:14px}
input[type=number]{-moz-appearance:textfield}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0 0}
.kpi{min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:21px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.kpi .d{font-size:13px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#map{width:100%;height:560px}
.note{font-size:13px;color:var(--muted);margin:10px 2px 0}
.filters{display:grid;grid-template-columns:repeat(4,minmax(0,1fr)) auto;gap:16px 18px;align-items:end}
.pair{display:flex;gap:6px;align-items:center}
.pair span{color:var(--muted);font-size:13px}
.fnote{font-size:13px;color:var(--muted);margin:12px 2px 0;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:8px;padding:8px 12px;font:inherit;font-size:13px;cursor:pointer;white-space:nowrap}
.linkbtn{border:0;background:none;color:var(--accent);font:inherit;font-size:13px;cursor:pointer;padding:0}
.tbar{display:flex;gap:12px;align-items:center;margin:18px 0 10px;flex-wrap:wrap}
.tbar input{max-width:300px}
.tbar .count{color:var(--muted);font-size:13px;margin-right:auto}
.tscroll{max-height:620px;overflow:auto;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
table{border-collapse:separate;border-spacing:0;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th{position:sticky;background:var(--panel);text-align:right;font-size:12px;font-weight:600;color:var(--muted);
   padding:0 12px;border-bottom:1px solid var(--line);white-space:nowrap;user-select:none;z-index:1}
thead tr:first-child th{top:0;height:30px}
thead tr:nth-child(2) th{top:30px;height:30px;cursor:pointer}
thead tr:first-child th.grp{text-align:center;color:var(--text);border-left:1px solid var(--line)}
th.l,td.l{text-align:left}
th.gstart,td.gstart{border-left:1px solid var(--line)}
th.sorted{color:var(--text)}
th.act,td.act{background:var(--act)}
.stk0,.stk1{position:sticky;background:var(--panel);z-index:2}
.stk0{left:0;min-width:58px;max-width:58px}
.stk1{left:58px;border-right:1px solid var(--line)}
thead th.stk0,thead th.stk1{z-index:3}
td.stk1{max-width:220px;overflow:hidden;text-overflow:ellipsis}
td{padding:7px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
tbody tr:last-child td{border-bottom:0}
.pos{color:var(--pos)}.neg{color:var(--neg)}.na{color:var(--muted)}
.warn{background:#fff4e5;color:#7a4b00;border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:14px}
@media (prefers-color-scheme: dark){.warn{background:#3a2c14;color:#f3c77a}}
.src{width:100%;font-size:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.src th{position:static;text-align:left;padding:10px 14px;height:auto}
.src td{text-align:left;white-space:normal;vertical-align:top;padding:10px 14px}
.src a{color:var(--accent)}
.method{color:var(--muted);font-size:14px;padding-left:20px}
.method li{margin:5px 0}
footer{margin-top:22px;font-size:13px;color:var(--muted)}
@media (max-width:900px){
  .filters{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:760px){
  .controls{grid-template-columns:minmax(0,1fr)}
  .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
  .kpi .v{font-size:17px}
  #map{height:360px}
  .ticks button{font-size:11px}
  .filters{grid-template-columns:minmax(0,1fr)}
  .seg button{font-size:12px;padding:7px 4px}
  .src thead{display:none}
  .src tr{display:block;border-bottom:1px solid var(--line)}
  .src td{display:block;padding:4px 14px}
  .src td:first-child{padding-top:12px;font-weight:600}
  .src td:last-child{padding-bottom:12px}
}
</style>
__PLOTLY__
</head>
<body>
<div class="wrap">
  <h1>US Counties Dashboard</h1>
  <p class="sub">Every US county on population, violent crime, education and income, with 5-year and 1-year changes.</p>
  <div id="warn"></div>

  <div class="panel">
    <div class="controls">
      <div>
        <div class="ctl-label">Map layer</div>
        <input id="layer" type="range" min="0" max="3" step="1" value="0" aria-label="Map layer">
        <div class="ticks" id="ticks"></div>
      </div>
      <div>
        <div class="ctl-label">Show</div>
        <div class="seg" id="meas"></div>
      </div>
    </div>
    <div id="map"></div>
    <p class="note" id="mapnote"></p>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="ctl-label" id="k1l">Counties shown</div><div class="v" id="k1"></div><div class="d" id="k1d"></div></div>
    <div class="kpi"><div class="ctl-label" id="k2l">Median</div><div class="v" id="k2"></div><div class="d" id="k2d"></div></div>
    <div class="kpi"><div class="ctl-label" id="k3l">Highest</div><div class="v" id="k3"></div><div class="d" id="k3d"></div></div>
    <div class="kpi"><div class="ctl-label" id="k4l">Lowest</div><div class="v" id="k4"></div><div class="d" id="k4d"></div></div>
  </div>

  <h2 id="ttitle">Explore the data</h2>
  <p class="h2sub">Filters and grouping apply to the map, the summary boxes and the table.</p>
  <div class="panel">
    <div class="filters">
      <div>
        <div class="ctl-label">State</div>
        <select id="fState"><option value="">All states</option></select>
      </div>
      <div>
        <div class="ctl-label">Population</div>
        <div class="pair">
          <select id="fPopMin" aria-label="Minimum population">
            <option value="">No min</option><option value="1000">1,000+</option><option value="5000">5,000+</option>
            <option value="10000" selected>10,000+</option><option value="25000">25,000+</option>
            <option value="50000">50,000+</option><option value="100000">100,000+</option>
            <option value="250000">250,000+</option><option value="500000">500,000+</option>
          </select><span>to</span>
          <select id="fPopMax" aria-label="Maximum population">
            <option value="">No max</option><option value="10000">10,000</option><option value="25000">25,000</option>
            <option value="50000">50,000</option><option value="100000">100,000</option><option value="250000">250,000</option>
            <option value="500000">500,000</option><option value="1000000">1,000,000</option>
          </select>
        </div>
      </div>
      <div>
        <div class="ctl-label">Bachelor's degree+ (%)</div>
        <div class="pair">
          <input id="fEduMin" type="number" min="0" max="100" step="1" placeholder="Min" aria-label="Minimum bachelor's percent"><span>to</span>
          <input id="fEduMax" type="number" min="0" max="100" step="1" placeholder="Max" aria-label="Maximum bachelor's percent">
        </div>
      </div>
      <div>
        <div class="ctl-label">Median income ($)</div>
        <div class="pair">
          <input id="fIncMin" type="number" min="0" step="1000" placeholder="Min" aria-label="Minimum median income"><span>to</span>
          <input id="fIncMax" type="number" min="0" step="1000" placeholder="Max" aria-label="Maximum median income">
        </div>
      </div>
      <div>
        <div class="ctl-label">Group by</div>
        <div class="seg" id="group">
          <button data-g="county" class="on">Counties</button>
          <button data-g="state">States</button>
        </div>
      </div>
    </div>
    <div class="fnote"><span id="fsummary"></span><button class="linkbtn" id="reset">Reset filters</button></div>
  </div>

  <div class="tbar">
    <input id="q" type="search" placeholder="Search county or state">
    <span class="count" id="count"></span>
    <button class="btn" id="csv">Download CSV</button>
  </div>
  <div class="tscroll">
    <table id="tbl">
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <h2>Data sources</h2>
  <p class="h2sub">Where every number on this page comes from.</p>
  <table class="src">
    <thead><tr><th>Dataset</th><th>Publisher</th><th>Used for</th><th>Years</th></tr></thead>
    <tbody id="sources"></tbody>
  </table>
  <h2>How the numbers are calculated</h2>
  <ul class="method" id="method"></ul>
  <footer id="gen"></footer>
</div>

<script>
const D = __DATA__;
const R = D.rows;
const I = {}; D.fields.forEach((f, i) => I[f] = i);
const LB = D.labels;

const nn = v => v !== null && v !== undefined && !Number.isNaN(v);
const pct = (a, b) => nn(a) && nn(b) && b > 0 ? (a / b - 1) * 100 : null;
const rateChg = (u, k) => {
  const a = u[I["a"+k]], ap = u[I["ap"+k]], b = u[I["b"+k]], bp = u[I["bp"+k]];
  return nn(a) && nn(ap) && nn(b) && nn(bp) && ap > 0 && bp > 0 && b > 0 ? ((a / ap) / (b / bp) - 1) * 100 : null;
};
const diff = (a, b) => nn(a) && nn(b) ? a - b : null;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const cssVar = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmtInt = v => Math.round(v).toLocaleString();
const fmtCh = (v, unit) => (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1) + unit;

const DIV = [[0,"#b2182b"],[0.25,"#ef8a62"],[0.5,"#f7f7f7"],[0.75,"#67a9cf"],[1,"#2166ac"]];
const DIV_INV = [[0,"#2166ac"],[0.25,"#67a9cf"],[0.5,"#f7f7f7"],[0.75,"#ef8a62"],[1,"#b2182b"]];
const CATS = [
  {key:"pop", label:"Population", tick:"Population", head:"Population",
   lvl:u => u[I.p0], ch5:u => pct(u[I.p0], u[I.p5]), ch1:u => pct(u[I.p0], u[I.p1]),
   fmt:fmtInt, unit:"%", log:true,
   scale:[[0,"#f2f0f7"],[0.35,"#cbc9e2"],[0.65,"#9e9ac8"],[1,"#54278f"]],
   desc:"Resident population (July 1 estimates)."},
  {key:"crime", label:"Violent crime rate", tick:"Crime", head:"Violent crime /100k",
   lvl:u => u[I.cL], ch5:u => rateChg(u, 5), ch1:u => rateChg(u, 1),
   fmt:fmtInt, unit:"%", invert:true, tickfmt:",.0f",
   scale:[[0,"#fee5d9"],[0.35,"#fcae91"],[0.65,"#fb6a4a"],[1,"#a50f15"]],
   desc:"Violent crimes per 100,000 residents of areas covered by reporting police agencies."},
  {key:"edu", label:"Education", tick:"Education", head:"Bachelor's degree+",
   lvl:u => u[I.e0], ch5:u => diff(u[I.e0], u[I.e5]), ch1:u => diff(u[I.e0], u[I.e1]),
   fmt:v => v.toFixed(1) + "%", unit:" pts", suffix:"%", tickfmt:".0f",
   scale:[[0,"#eff3ff"],[0.35,"#bdd7e7"],[0.65,"#6baed6"],[1,"#08519c"]],
   desc:"Share of adults 25+ with a bachelor's degree or higher. Changes are in percentage points."},
  {key:"inc", label:"Median income", tick:"Income", head:"Median income",
   lvl:u => u[I.i0], ch5:u => pct(u[I.i0], u[I.i5]), ch1:u => pct(u[I.i0], u[I.i1]),
   fmt:v => "$" + fmtInt(v), unit:"%", prefix:"$", tickfmt:",.0f",
   scale:[[0,"#edf8e9"],[0.35,"#bae4b3"],[0.65,"#74c476"],[1,"#006d2c"]],
   desc:"Median household income (nominal dollars)."}
];
const MEAS = [{key:"lvl", label:"Current level", short:"Level"},
              {key:"ch5", label:"5-year change", short:"5-yr"},
              {key:"ch1", label:"1-year change", short:"1-yr"}];

const S = {cat:0, meas:"ch5", group:"county", q:"", sortKey:"rank", sortDir:1};
const F = {state:"", popMin:10000, popMax:null, eduMin:null, eduMax:null, incMin:null, incMax:null};
const STATES = [...new Map(R.map(r => [r[I.abbr], r[I.state]])).entries()].sort((a, b) => a[1].localeCompare(b[1]));

function quantile(sorted, q){
  if (!sorted.length) return 0;
  const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
}
function valueOf(u, c, m){ const C = CATS[c]; return C[m](u); }
function fmtVal(v, c, m){
  if (!nn(v)) return null;
  return m === "lvl" ? CATS[c].fmt(v) : fmtCh(v, CATS[c].unit);
}
function measAvailable(c, m){ return !(CATS[c].key === "crime" && m !== "lvl" && !D.crimeChanges); }
function metricName(c, m){
  const C = CATS[c];
  return m === "lvl" ? `${C.label} (${LB[C.key].lvl})` : `${C.label}, ${MEAS.find(x => x.key === m).label.toLowerCase()} (${LB[C.key][m]})`;
}

// ---------- filters & grouping ----------
function passes(r){
  if (F.state && r[I.abbr] !== F.state) return false;
  const p = r[I.p0], e = r[I.e0], inc = r[I.i0];
  if (F.popMin !== null && !(nn(p) && p >= F.popMin)) return false;
  if (F.popMax !== null && !(nn(p) && p <= F.popMax)) return false;
  if (F.eduMin !== null && !(nn(e) && e >= F.eduMin)) return false;
  if (F.eduMax !== null && !(nn(e) && e <= F.eduMax)) return false;
  if (F.incMin !== null && !(nn(inc) && inc >= F.incMin)) return false;
  if (F.incMax !== null && !(nn(inc) && inc <= F.incMax)) return false;
  return true;
}
function aggregate(rows){
  const by = new Map();
  for (const r of rows){ if (!by.has(r[I.abbr])) by.set(r[I.abbr], []); by.get(r[I.abbr]).push(r); }
  const out = [];
  for (const [ab, rs] of by){
    const u = new Array(D.fields.length).fill(null);
    u[I.fips] = ab; u[I.county] = rs[0][I.state]; u[I.state] = rs[0][I.state]; u[I.abbr] = ab; u.n = rs.length;
    const sumIf = (idx, ok) => { let s = 0, k = 0; for (const r of rs) if (ok(r)) { s += r[idx]; k++; } return k ? s : null; };
    for (const f of ["p0","p1","p5"]) u[I[f]] = sumIf(I[f], r => nn(r[I[f]]));
    let cs = 0, ps = 0;
    for (const r of rs) if (nn(r[I.cL]) && nn(r[I.cpL])) { cs += r[I.cL] * r[I.cpL]; ps += r[I.cpL]; }
    u[I.cL] = ps > 0 ? cs / ps : null; u[I.cpL] = ps > 0 ? ps : null;
    for (const k of [5, 1]){
      const f = ["a"+k, "ap"+k, "b"+k, "bp"+k];
      const ok = r => f.every(x => nn(r[I[x]]));
      for (const x of f) u[I[x]] = sumIf(I[x], ok);
    }
    for (const [e, ea] of [["e0","ea0"],["e1","ea1"],["e5","ea5"]]){
      let s = 0, w = 0;
      for (const r of rs) if (nn(r[I[e]]) && nn(r[I[ea]]) && r[I[ea]] > 0) { s += r[I[e]] * r[I[ea]]; w += r[I[ea]]; }
      u[I[e]] = w > 0 ? s / w : null; u[I[ea]] = w > 0 ? w : null;
    }
    for (const i of ["i0","i1","i5"]){
      let s = 0, w = 0;
      for (const r of rs) if (nn(r[I[i]]) && nn(r[I.p0])) { s += r[I[i]] * r[I.p0]; w += r[I.p0]; }
      u[I[i]] = w > 0 ? s / w : null;
    }
    out.push(u);
  }
  return out;
}
function computeView(){
  const counties = R.filter(passes);
  const units = S.group === "state" ? aggregate(counties) : counties;
  const val = u => valueOf(u, S.cat, S.meas);
  const ranked = units.filter(u => nn(val(u))).sort((a, b) => val(b) - val(a));
  const rank = new Map();
  ranked.forEach((u, i) => rank.set(u[I.fips], (i > 0 && val(u) === val(ranked[i-1])) ? rank.get(ranked[i-1][I.fips]) : i + 1));
  return {counties, units, ranked, rank, val};
}

// ---------- map ----------
function unitName(u){ return S.group === "state" ? u[I.state] : `${u[I.county]}, ${u[I.abbr]}`; }
function hoverText(u, v){
  const parts = [`<b>${esc(unitName(u))}</b>` + (S.group === "state" ? ` (${u.n} counties shown)` : "")];
  CATS.forEach((C, c) => {
    const lv = C.lvl(u);
    const chs = ["ch5","ch1"].filter(m => measAvailable(c, m)).map(m => {
      const x = C[m](u); return `${m === "ch5" ? "5y" : "1y"} ${nn(x) ? fmtCh(x, C.unit) : "n/a"}`; });
    let line = `${C.head} ${LB[C.key].lvl}: ${nn(lv) ? C.fmt(lv) : "n/a"}` + (chs.length ? `  (${chs.join(", ")})` : "");
    if (C.key === "crime" && S.group === "county" && nn(u[I.cov]))
      line += `<br><span style="opacity:.7">  ${Math.round(Math.min(u[I.cov], 9.99) * 100)}% of residents covered by reporting agencies</span>`;
    parts.push(line);
  });
  const rk = v.rank.get(u[I.fips]);
  if (rk) parts.push(`<i>Rank ${rk.toLocaleString()} of ${v.ranked.length.toLocaleString()} by ${esc(CATS[S.cat].label.toLowerCase())} ${S.meas === "lvl" ? "" : MEAS.find(x => x.key === S.meas).label.toLowerCase()}</i>`);
  return parts.join("<br>");
}
function updateMap(v){
  const C = CATS[S.cat], m = S.meas;
  const isLog = m === "lvl" && C.log;
  const raw = v.units.map(u => v.val(u));
  const z = raw.map(x => nn(x) ? (isLog ? (x > 0 ? Math.log10(x) : null) : x) : null);
  const vals = z.filter(nn).sort((a, b) => a - b);
  let lo = quantile(vals, 0.02), hi = quantile(vals, 0.98);
  if (m !== "lvl"){ const mx = Math.max(Math.abs(lo), Math.abs(hi)) || 1; lo = -mx; hi = mx; }
  if (lo === hi) hi = lo + 1;
  const text = v.units.map(u => hoverText(u, v));
  const line = {width: S.group === "state" ? 0.6 : 0.25, color: cssVar("--panel")};
  const colorbar = {title:{text: m === "lvl" ? C.head : MEAS.find(x => x.key === m).label, side:"top"},
    thickness:12, len:0.75, outlinewidth:0, tickfont:{color: cssVar("--muted")}};
  if (isLog){
    const short = n => n >= 1e6 ? (n / 1e6) + "M" : n >= 1e3 ? (n / 1e3) + "K" : String(n);
    const nice = [];
    for (let e = 1; e <= 7; e++) for (const m of [1, 2, 5]) nice.push(m * 10 ** e);
    const inRange = nice.filter(n => Math.log10(n) >= lo && Math.log10(n) <= hi);
    const pick = inRange.length > 6 ? inRange.filter(n => String(n)[0] !== "2") : inRange;
    Object.assign(colorbar, {tickvals: pick.map(Math.log10), ticktext: pick.map(short)});
  } else if (m === "lvl"){
    Object.assign(colorbar, {tickformat: C.tickfmt || "", ticksuffix: C.suffix || "", tickprefix: C.prefix || ""});
  } else {
    Object.assign(colorbar, {tickformat: "+.1~f", ticksuffix: C.unit});
  }
  const scale = m === "lvl" ? C.scale : (C.invert ? DIV_INV : DIV);
  const main = {type:"choropleth", z, text, hovertemplate:"%{text}<extra></extra>", colorscale:scale,
    zmin:lo, zmax:hi, marker:{line}, colorbar};
  let base;
  if (S.group === "state"){
    const all = STATES.map(s => s[0]);
    base = {type:"choropleth", locationmode:"USA-states", locations:all, z:all.map(() => 0),
      colorscale:[[0, cssVar("--empty")],[1, cssVar("--empty")]], showscale:false, hoverinfo:"skip", marker:{line}};
    Object.assign(main, {locationmode:"USA-states", locations: v.units.map(u => u[I.abbr])});
  } else {
    const locs = (F.state ? R.filter(r => r[I.abbr] === F.state) : R).map(r => r[I.fips]);
    base = {type:"choropleth", geojson:D.geo, featureidkey:"id", locations:locs, z:locs.map(() => 0),
      colorscale:[[0, cssVar("--empty")],[1, cssVar("--empty")]], showscale:false, hoverinfo:"skip", marker:{line}};
    Object.assign(main, {geojson:D.geo, featureidkey:"id", locations: v.units.map(u => u[I.fips])});
  }
  const layout = {margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:"rgba(0,0,0,0)",
    font:{color: cssVar("--text"), family:"-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif"},
    hoverlabel:{align:"left"},
    geo:{scope:"usa", projection:{type:"albers usa"}, bgcolor:"rgba(0,0,0,0)",
         showlakes:false, showland:false, showcoastlines:false, showsubunits:false, showframe:false}};
  const canZoom = !!(window.PlotlyGeoAssets && PlotlyGeoAssets.topojson && PlotlyGeoAssets.topojson.world_110m);
  if (F.state && S.group === "county" && canZoom){   // zoom to the chosen state
    Object.assign(layout.geo, {scope:"world", projection:{type:"mercator"}, fitbounds:"locations",
                               visible:false, showcountries:false});
  }
  Plotly.react("map", [base, main], layout, {responsive:true, displaylogo:false, modeBarButtonsToRemove:["select2d","lasso2d"]});
  let note = `${metricName(S.cat, m)}. ${C.desc}`;
  if (m !== "lvl") note += C.invert ? " Red = rising, blue = falling." : " Blue = rising, red = falling.";
  note += " Gray = no data or filtered out.";
  document.getElementById("mapnote").textContent = note;
}

// ---------- KPIs ----------
function setK(i, label, value, detail){
  document.getElementById(`k${i}l`).textContent = label;
  document.getElementById(`k${i}`).textContent = value;
  document.getElementById(`k${i}d`).textContent = detail || "";
}
function updateKpis(v){
  const unitWord = S.group === "state" ? "States" : "Counties";
  setK(1, `${unitWord} shown`, v.units.length.toLocaleString(),
       S.group === "state" ? `${v.counties.length.toLocaleString()} counties` : `${v.ranked.length.toLocaleString()} with data for this layer`);
  const vs = v.ranked.map(v.val).sort((a, b) => a - b);
  const C = CATS[S.cat], m = S.meas;
  const short = m === "lvl" ? C.tick.toLowerCase() : `${C.tick.toLowerCase()} ${MEAS.find(x => x.key === m).short}`;
  setK(2, `Median ${short}`, vs.length ? fmtVal(quantile(vs, 0.5), S.cat, m) : "n/a", metricName(S.cat, m));
  const top = v.ranked[0], bot = v.ranked[v.ranked.length - 1];
  setK(3, "Highest", top ? unitName(top) : "n/a", top ? fmtVal(v.val(top), S.cat, m) : "");
  if (m !== "lvl"){
    setK(4, `${unitWord} rising`, vs.length ? Math.round(100 * vs.filter(x => x > 0).length / vs.length) + "%" : "n/a",
         bot ? `Lowest: ${unitName(bot)} ${fmtVal(v.val(bot), S.cat, m)}` : "");
  } else {
    setK(4, "Lowest", bot ? unitName(bot) : "n/a", bot ? fmtVal(v.val(bot), S.cat, m) : "");
  }
}

// ---------- table ----------
function columns(){
  const cols = [{key:"rank", label:"Rank"}];
  if (S.group === "state"){ cols.push({key:"name", label:"State", l:true}, {key:"n", label:"Counties"}); }
  else { cols.push({key:"name", label:"County", l:true}, {key:"state", label:"State", l:true}); }
  CATS.forEach((C, c) => MEAS.forEach((M, mi) => {
    if (!measAvailable(c, M.key)) return;
    cols.push({key:`${c}.${M.key}`, c, m:M.key, label: M.key === "lvl" ? LB[C.key].lvl : M.short,
               title: metricName(c, M.key), gstart: mi === 0});
  }));
  return cols;
}
function cell(u, col, v){
  switch(col.key){
    case "rank": return v.rank.get(u[I.fips]) ?? null;
    case "name": return S.group === "state" ? u[I.state] : u[I.county];
    case "state": return u[I.abbr];
    case "n": return u.n;
    default: return valueOf(u, col.c, col.m);
  }
}
let last = {rows: [], v: null};
function updateTable(v){
  const cols = columns();
  const q = S.q.trim().toLowerCase();
  let rows = v.units.filter(u => !q || unitName(u).toLowerCase().includes(q) || String(u[I.state]).toLowerCase().includes(q));
  const sc = cols.find(c => c.key === S.sortKey) || cols[0];
  rows = rows.slice().sort((a, b) => {
    const x = cell(a, sc, v), y = cell(b, sc, v);
    if (!nn(x) && !nn(y)) return 0; if (!nn(x)) return 1; if (!nn(y)) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * S.sortDir;
  });
  last = {rows, v, cols};
  const act = `${S.cat}.${S.meas}`;
  const fixed = cols.filter(c => c.c === undefined).length;
  let h1 = `<tr><th class="l stk0"></th><th colspan="${fixed - 1}" class="l"></th>`;
  CATS.forEach((C, c) => {
    const n = cols.filter(x => x.c === c).length;
    if (n) h1 += `<th class="grp" colspan="${n}">${esc(C.head)}</th>`;
  });
  h1 += "</tr>";
  const h2 = "<tr>" + cols.map(c => {
    const cls = [c.l ? "l" : "", c.gstart ? "gstart" : "", S.sortKey === c.key ? "sorted" : "", c.key === act ? "act" : "",
                 c.key === "rank" ? "stk0" : c.key === "name" ? "stk1" : ""].join(" ");
    const arrow = S.sortKey === c.key ? (S.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-k="${c.key}" class="${cls}" title="${esc(c.title || "Sort")}">${esc(c.label)}${arrow}</th>`;
  }).join("") + "</tr>";
  document.getElementById("thead").innerHTML = h1 + h2;
  const na = '<span class="na">n/a</span>';
  document.getElementById("tbody").innerHTML = rows.map(u => "<tr>" + cols.map(c => {
    const x = cell(u, c, v);
    const cls = [c.l ? "l" : "", c.gstart ? "gstart" : "", c.key === act ? "act" : "",
                 c.key === "rank" ? "stk0" : c.key === "name" ? "stk1" : ""];
    let txt;
    if (c.c === undefined){ txt = nn(x) ? (typeof x === "number" ? x.toLocaleString() : esc(x)) : na; }
    else if (!nn(x)) txt = na;
    else {
      txt = fmtVal(x, c.c, c.m);
      if (c.m !== "lvl" && x !== 0){
        const good = CATS[c.c].invert ? x < 0 : x > 0;
        cls.push(good ? "pos" : "neg");
      }
    }
    return `<td class="${cls.join(" ")}">${txt}</td>`;
  }).join("") + "</tr>").join("");
  document.getElementById("count").textContent = `${rows.length.toLocaleString()} ${S.group === "state" ? "states" : "counties"}`;
  document.getElementById("ttitle").textContent = `Explore the data: ranked by ${metricName(S.cat, S.meas).replace(/^./, s => s.toLowerCase())}`;
}

// ---------- wiring ----------
function readFilters(){
  const num = id => { const x = document.getElementById(id).value.trim(); if (x === "") return null; const n = Number(x.replace(/[$,%\s]/g, "")); return Number.isFinite(n) ? n : null; };
  F.state = document.getElementById("fState").value;
  F.popMin = num("fPopMin"); F.popMax = num("fPopMax");
  F.eduMin = num("fEduMin"); F.eduMax = num("fEduMax");
  F.incMin = num("fIncMin"); F.incMax = num("fIncMax");
  const bits = [];
  const rng = (lo, hi, f) => lo !== null && hi !== null ? `${f(lo)}–${f(hi)}` : lo !== null ? `${f(lo)}+` : `up to ${f(hi)}`;
  if (F.popMin !== null || F.popMax !== null) bits.push(`population ${rng(F.popMin, F.popMax, x => x.toLocaleString())}`);
  if (F.eduMin !== null || F.eduMax !== null) bits.push(`bachelor's+ ${rng(F.eduMin, F.eduMax, x => x + "%")}`);
  if (F.incMin !== null || F.incMax !== null) bits.push(`median income ${rng(F.incMin, F.incMax, x => "$" + x.toLocaleString())}`);
  const where = F.state ? ` in ${STATES.find(s => s[0] === F.state)[1]}` : "";
  document.getElementById("fsummary").textContent = `Showing counties${where}` + (bits.length ? " with " + bits.join(", ") : "") + ".";
}
function update(){
  document.querySelectorAll("#ticks button").forEach((b, i) => b.classList.toggle("on", i === S.cat));
  document.querySelectorAll("#meas button").forEach(b => {
    b.classList.toggle("on", b.dataset.m === S.meas);
    b.disabled = !measAvailable(S.cat, b.dataset.m); b.style.opacity = b.disabled ? .4 : 1;
  });
  document.querySelectorAll("#group button").forEach(b => b.classList.toggle("on", b.dataset.g === S.group));
  const v = computeView();
  updateMap(v); updateKpis(v); updateTable(v);
}
function setMetric(cat, meas){
  S.cat = cat; if (meas) S.meas = meas;
  if (!measAvailable(S.cat, S.meas)) S.meas = "lvl";
  document.getElementById("layer").value = S.cat;
  S.sortKey = "rank"; S.sortDir = 1;
  update();
}
const ticks = document.getElementById("ticks");
CATS.forEach((C, i) => { const b = document.createElement("button"); b.textContent = C.tick; b.onclick = () => setMetric(i); ticks.appendChild(b); });
const measEl = document.getElementById("meas");
MEAS.forEach(M => { const b = document.createElement("button"); b.dataset.m = M.key; b.textContent = M.label; b.onclick = () => setMetric(S.cat, M.key); measEl.appendChild(b); });
document.getElementById("layer").addEventListener("input", e => setMetric(+e.target.value));
const fState = document.getElementById("fState");
STATES.forEach(([ab, name]) => { const o = document.createElement("option"); o.value = ab; o.textContent = name; fState.appendChild(o); });
let tmr = null;
const onFilter = () => { clearTimeout(tmr); tmr = setTimeout(() => { readFilters(); update(); }, 200); };
["fState","fPopMin","fPopMax"].forEach(id => document.getElementById(id).addEventListener("change", onFilter));
["fEduMin","fEduMax","fIncMin","fIncMax"].forEach(id => document.getElementById(id).addEventListener("input", onFilter));
document.getElementById("group").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return; S.group = b.dataset.g; S.sortKey = "rank"; S.sortDir = 1;
  document.getElementById("q").placeholder = S.group === "state" ? "Search state" : "Search county or state";
  update(); });
document.getElementById("reset").onclick = () => {
  ["fState","fEduMin","fEduMax","fIncMin","fIncMax","fPopMax"].forEach(id => document.getElementById(id).value = "");
  document.getElementById("fPopMin").value = "10000"; readFilters(); update(); };
document.getElementById("thead").addEventListener("click", e => {
  const th = e.target.closest("th[data-k]"); if (!th) return;
  const k = th.dataset.k;
  if (S.sortKey === k) S.sortDir *= -1;
  else { S.sortKey = k; S.sortDir = (k === "name" || k === "state" || k === "rank") ? 1 : -1; }
  updateTable(last.v); });
document.getElementById("q").addEventListener("input", e => { S.q = e.target.value; updateTable(last.v); });
document.getElementById("csv").onclick = () => {
  const {rows, v, cols} = last;
  const head = cols.map(c => c.c === undefined ? c.label : `${CATS[c.c].head} ${c.m === "lvl" ? LB[CATS[c.c].key].lvl : MEAS.find(x => x.key === c.m).label} (${c.m === "lvl" ? "level" : CATS[c.c].unit.trim()})`);
  const q = s => '"' + String(s).replace(/"/g, '""') + '"';
  const lines = [head.map(q).join(",")].concat(rows.map(u => cols.map(c => {
    const x = cell(u, c, v); if (!nn(x)) return "";
    return typeof x === "string" ? q(x) : (Number.isInteger(x) ? x : x.toFixed(2)); }).join(",")));
  const blob = new Blob([lines.join("\n")], {type:"text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = S.group === "state" ? "state_rankings.csv" : "county_rankings.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", update);

document.getElementById("warn").innerHTML = (D.problems || []).map(n => `<div class="warn">${esc(n)}</div>`).join("");
document.getElementById("sources").innerHTML = D.sources.map(s =>
  `<tr><td><a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.name)}</a></td><td>${esc(s.publisher)}</td><td>${esc(s.used)}</td><td>${esc(s.years)}</td></tr>`).join("");
document.getElementById("method").innerHTML = D.notes.map(n => `<li>${esc(n)}</li>`).join("");
document.getElementById("gen").textContent = "Data downloaded and dashboard generated " + D.generated + ".";
readFilters();
update();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_app()
    elif "--export" in sys.argv:
        df, meta = build_dataset()
        out = readable_table(df, meta)
        path = DATA_DIR / "county_data.csv"
        out.to_csv(path, index=False)
        print(f"Wrote {len(out):,} counties to {path}")
    else:
        # Default (and --html): build the single shareable HTML file and open it
        path = export_html()
        size_mb = path.stat().st_size / 1e6
        print(f"Wrote {path} ({size_mb:.1f} MB). Send this file to anyone; "
              "they just double-click it to open it in their browser.")
        if "--no-open" not in sys.argv:
            import webbrowser
            webbrowser.open(path.as_uri())
