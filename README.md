# US County Dashboard

Maps all ~3,140 US counties on four measures, each viewable as its current level, 5-year change, or 1-year change:

| Measure | Level | 5-year change | 1-year change |
|---|---|---|---|
| Population | 2025 | 2020 to 2025 (%) | 2024 to 2025 (%) |
| Violent crime per 100k | latest FBI year | latest minus 5 years (%) | latest minus 1 year (%) |
| Bachelor's degree+ (adults 25+) | ACS 2020-24 | ACS 2015-19 to 2020-24 (pct. points) | ACS 2019-23 to 2020-24 (pct. points) |
| Median household income | ACS 2020-24 | ACS 2015-19 to 2020-24 (%) | ACS 2019-23 to 2020-24 (%) |

Below the map you can filter by state, population (min/max), bachelor's share (min/max), and median income (min/max), and group the results into states. Filters apply to the map, the summary boxes, and the table. Choosing a state zooms the map to it. The table ranks by whatever the map is showing, sorts by any column, and downloads as CSV. The bottom of the page lists every data source and explains the calculations.

## Run it

```bash
pip install -r requirements.txt
python3 county_dashboard.py --html
```

This builds `county_dashboard.html` next to the script and opens it in your browser. Send that file to anyone. They double-click it, with no Python or internet needed.

**The first run is slow (about 20-40 minutes)** because it downloads violent-crime counts for roughly 15,000 police agencies from the FBI, one agency at a time. It prints progress state by state and saves each state in `data/fbi/`, so later runs take seconds. If it gets interrupted, run it again and it continues where it stopped.

Other options:

- `streamlit run county_dashboard.py` shows the same dashboard inside Streamlit.
- `python3 county_dashboard.py --export` writes `data/county_data.csv`, one row per county with every level and change, which is handy for modeling.
- `--no-open` builds the HTML without opening the browser.

If the Census download complains, get a free key at https://api.census.gov/data/key_signup.html and run `export CENSUS_API_KEY=your_key` first.

## Refreshing data

Delete files in `data/` to re-download them:
- `data/fbi/` for crime (also delete `latest_year.json` there to pick up a new FBI year)
- `data/pep.csv` for population
- `data/acs_*.json` for income and education (bump `ACS_LATEST` to 2025 once the ACS 2021-2025 release is out in December 2026)

## How crime is calculated

The FBI does not publish county crime totals, so the script builds them: it adds up the monthly violent-crime counts reported by each city police department, sheriff, county police, and campus police department in the county. State police, tribal, and special-purpose agencies are left out because they can't be assigned to one county. Partial years with at least 6 months are scaled to 12 months. A county's rate is shown only if reporting agencies cover at least 50% of its residents. Changes compare only agencies that reported in both years, so an agency starting or stopping reporting doesn't look like a crime trend.

## Sources

| Data | Source |
|---|---|
| Population | Census Population Estimates, Vintage 2025 (CO-EST2025-alldata) |
| Median household income | ACS 5-year, table B19013 (2015-19, 2019-23, 2020-24) |
| Bachelor's degree or higher | ACS 5-year, table B15003, age 25+ (same releases) |
| Violent crime | FBI Crime Data Explorer, agency-level monthly counts |
| County boundaries | Census cartographic boundary file 2024, 1:20m |
