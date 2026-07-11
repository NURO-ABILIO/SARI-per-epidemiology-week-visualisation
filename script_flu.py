from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# =========================================================
# TASK 1 — DEFINE FILE PATHS
# =========================================================
BASE = Path("/Users/nuro/Documents/analysis/IDS/flu_and_sars/Script_data_flu_Sars/untitled folder/flu")
WW_FILE = BASE / "wastewater.xlsx"
BD_FILE = BASE / "Sars_and_Flu_clean.csv"
POS_FILE = BASE / "Sars_Influenza_Positivity_Data.csv"

OUT_FLU = BASE / "Influenza_typed_counts_clean.png"
OUT_POS_WEEKLY = BASE / "weekly_positivity_recomputed.csv"

# =========================================================
# TASK 2 — DEFINE MANUAL WEEKS OF INTEREST
# =========================================================
weeks = []
weeks += [(2024, w) for w in range(47, 53)]
weeks += [(2025, w) for w in range(1, 54)]
weeks += [(2026, w) for w in range(1, 26)]

time_index = pd.MultiIndex.from_tuples(weeks, names=["year", "week"])
labels = [f"{y}-W{w}" for y, w in weeks]
x = np.arange(len(time_index))
WEEKS_OF_INTEREST = set(weeks)

# =========================================================
# TASK 3 — DEFINE COLOR SCALES
# =========================================================
flu_colors = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#FF7F00",
    "#FFFF33",
    "#A65628",
    "#F781BF",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#666666",
    "#00A8E8",
    "#FF006E",
    "#8338EC",
    "#3A86FF",
    "#FB5607",
    "#06D6A0",
    "#118AB2",
    "#EF476F",
    "#FFD166",
    "#073B4C",
    "#F94144",
    "#43AA8B",
    "#577590",
    "#F8961E",
    "#90BE6D",
    "#F3722C"
]

def generate_distinct_colors(n, seed_colors):
    if n <= 0:
        return []
    if n <= len(seed_colors):
        return seed_colors[:n]

    extra_n = n - len(seed_colors)
    extra_cols = []

    for i in range(extra_n):
        h = i / max(extra_n, 1)
        s = 0.70
        v = 0.86
        rgb = mcolors.hsv_to_rgb((h, s, v))
        extra_cols.append(mcolors.to_hex(rgb))

    return seed_colors + extra_cols


# =========================================================
# TASK 4 — STANDARDIZE NAMES
# =========================================================
def norm_pathogen(value):
    s = str(value).strip().lower()
    if "influenza" in s or "flu" in s:
        return "Influenza"
    return None


def norm_setting(value):
    s = str(value).strip().lower()
    if "comunit" in s:
        return "Community"
    if "hospital" in s:
        return "Health Care Facility"
    if "wast" in s or "aguas" in s or "residu" in s:
        return "Wastewater"
    return None


def split_lineages(raw):
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "-", "–"}:
        return []
    s = s.replace(",", ";")
    parts = [p.strip(" .,") for p in s.split(";")]
    return [x for x in parts if x and x not in {"-", "–"}]


def clean_lineage_name(s):
    s = str(s).strip()
    if s in {"", "nan", "None"}:
        return s
    return s


# =========================================================
# TASK 5 — FILTER DATA TO WEEKS OF INTEREST
# =========================================================
def keep_only_weeks_of_interest(df):
    if df.empty:
        return df.copy()
    mask = df[["year", "week"]].apply(tuple, axis=1).isin(WEEKS_OF_INTEREST)
    return df.loc[mask].copy()


# =========================================================
# TASK 6 — PARSE WASTEWATER YEAR/WEEK STRUCTURE
# =========================================================
def parse_wastewater_week_year(df):
    week_col = ("Epidemiological Week", "Unnamed: 0_level_1")
    order_col = ("Order", "Unnamed: 1_level_1")

    raw_week = pd.to_numeric(df[week_col], errors="coerce")
    year_markers = raw_week[raw_week >= 2000].dropna().astype(int).tolist()
    default_year = year_markers[0] - 1 if year_markers else 2024

    years = []
    weeks_parsed = []
    current_year = default_year
    current_week = np.nan

    for val in raw_week.tolist():
        if pd.notna(val) and val >= 2000:
            current_year = int(val)
            current_week = np.nan
            years.append(np.nan)
            weeks_parsed.append(np.nan)
            continue

        if pd.notna(val):
            w = int(val)
            if pd.notna(current_week) and current_week >= 50 and w <= 5:
                current_year += 1
            current_week = w

        years.append(current_year)
        weeks_parsed.append(current_week)

    out = df.copy()
    out["year"] = years
    out["week"] = weeks_parsed

    out = out[pd.to_numeric(out[order_col], errors="coerce").notna()].copy()
    out = out[pd.to_numeric(out["week"], errors="coerce").notna()].copy()
    out["year"] = out["year"].astype(int)
    out["week"] = out["week"].astype(int)
    return out


# =========================================================
# TASK 7 — COMPUTE WEEKLY POSITIVITY FROM WASTEWATER RESULTS
# =========================================================
def weekly_pos_from_result(df, result_col, pathogen):
    s = df[result_col].astype(str).str.strip().str.upper()
    valid = s.isin(["POS", "NEG"])

    g = (
        pd.DataFrame(
            {
                "year": df["year"],
                "week": df["week"],
                "result": s,
                "valid": valid,
            }
        )
        .loc[lambda x: x["valid"]]
        .groupby(["year", "week"], as_index=False)
        .agg(
            tests=("result", "size"),
            positives=("result", lambda vals: (vals == "POS").sum()),
        )
    )

    g["positivity"] = np.where(g["tests"] > 0, g["positives"] / g["tests"] * 100.0, np.nan)
    g["setting"] = "Wastewater"
    g["pathogen"] = pathogen
    return g[["pathogen", "setting", "year", "week", "tests", "positives", "positivity"]]


# =========================================================
# TASK 8 — LOAD LINEAGE COUNTS FROM CSV
# =========================================================
def load_lineage_counts():
    bd = pd.read_csv(BD_FILE)

    bd["pathogen"] = bd["Patogeno"].apply(norm_pathogen)
    bd["setting"] = bd["vigilancia"].apply(norm_setting)
    bd["year"] = pd.to_numeric(bd["Ano"], errors="coerce")
    bd["week"] = pd.to_numeric(bd["week"], errors="coerce")

    bd = bd.dropna(subset=["pathogen", "setting", "year", "week"]).copy()
    bd["year"] = bd["year"].astype(int)
    bd["week"] = bd["week"].astype(int)

    rows = []
    for _, r in bd.iterrows():
        toks = split_lineages(r["Subtype/Lineage(Clade and Pagolin)"])
        if not toks:
            continue

        weight = 1.0 / len(toks)
        for t in toks:
            rows.append(
                {
                    "pathogen": r["pathogen"],
                    "setting": r["setting"],
                    "year": r["year"],
                    "week": r["week"],
                    "lineage": t,
                    "count": weight,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = keep_only_weeks_of_interest(out)
    out = (
        out.groupby(
            ["pathogen", "setting", "year", "week", "lineage"],
            as_index=False
        )["count"].sum()
    )
    return out


# =========================================================
# TASK 9 — LOAD COMMUNITY AND HOSPITAL POSITIVITY
# =========================================================
def load_community_hospital_pos():
    d = pd.read_csv(POS_FILE)

    setting_map = {
        "COMUNIDADE": "Community",
        "HOSPITAL": "Health Care Facility",
        "AGUAS RESIDUAIS": "Wastewater",
    }

    d["setting"] = d["setting"].map(setting_map)
    d["pathogen"] = d["tipo"].apply(norm_pathogen)

    d = d.dropna(subset=["setting", "pathogen"]).copy()
    d["year"] = pd.to_numeric(d["ano"], errors="coerce")
    d["week"] = pd.to_numeric(d["Week_num"], errors="coerce")
    d["tests"] = pd.to_numeric(d["tests"], errors="coerce")
    d["positives"] = pd.to_numeric(d["positives"], errors="coerce")
    d["positivity"] = pd.to_numeric(d["positivity"], errors="coerce")

    d = d.dropna(subset=["year", "week"]).copy()
    d["year"] = d["year"].astype(int)
    d["week"] = d["week"].astype(int)

    d = d[["pathogen", "setting", "year", "week", "tests", "positives", "positivity"]]
    d = keep_only_weeks_of_interest(d)
    return d


# =========================================================
# TASK 10 — COMBINE ALL POSITIVITY DATA
# =========================================================
def compose_positivity_table():
    ww = pd.read_excel(WW_FILE, sheet_name="BASE DE DADOS ", header=[0, 1])
    ww2 = parse_wastewater_week_year(ww)
    ww2 = keep_only_weeks_of_interest(ww2)

    flu = weekly_pos_from_result(ww2, ("Influenza Sequencing", "Result"), "Influenza")
    ww_pos = flu.copy()

    other = load_community_hospital_pos()
    other = other[other["setting"].isin(["Community", "Health Care Facility"])].copy()

    all_pos = pd.concat([other, ww_pos], ignore_index=True)
    all_pos = all_pos.sort_values(["pathogen", "setting", "year", "week"]).reset_index(drop=True)
    return all_pos, ww_pos


# =========================================================
# TASK 11 — BUILD COLOR MAPS
# =========================================================
def build_color_map(lineages):
    """
    Build grouped influenza colours using three main colour families:
      1. H1N1  -> blue tones
      2. H3N2  -> orange/brown tones
      3. B Vic -> green tones

    Each lineage inside the same group receives a different shade/tonality.
    """

    def lineage_group(lin):
        s = str(lin).lower()
        if "h1n1" in s:
            return "H1N1"
        if "h3n2" in s:
            return "H3N2"
        if "b vic" in s or "bvic" in s or "victoria" in s:
            return "B Vic"
        if "not detected (to repeat)" in s:
            return "Not detected"
        return "Other"

    def make_shades(cmap_name, n, start=0.35, end=0.90):
        if n <= 0:
            return []
        cmap = plt.get_cmap(cmap_name)
        vals = np.linspace(start, end, n)
        return [mcolors.to_hex(cmap(v)) for v in vals]

    groups = {
        "H1N1": [],
        "H3N2": [],
        "B Vic": [],
        "Other": [],
        "Not detected": []
    }

    for lin in sorted(lineages):
        groups[lineage_group(lin)].append(lin)

    color_map = {}

    # Main grouped palettes
    for lin, col in zip(groups["H1N1"], make_shades("Blues", len(groups["H1N1"]), 0.45, 0.90)):
        color_map[lin] = col

    for lin, col in zip(groups["H3N2"], make_shades("Oranges", len(groups["H3N2"]), 0.45, 0.90)):
        color_map[lin] = col

    for lin, col in zip(groups["B Vic"], make_shades("Greens", len(groups["B Vic"]), 0.45, 0.90)):
        color_map[lin] = col

    # Neutral colours for special/other categories
    other_palette = generate_distinct_colors(len(groups["Other"]), ["#7F7F7F", "#BDBDBD", "#525252", "#969696"])
    for lin, col in zip(groups["Other"], other_palette):
        color_map[lin] = col

    for lin in groups["Not detected"]:
        color_map[lin] = "#FFFFFF"

    return color_map


# =========================================================
# TASK 12 — PREPARE STACKED COUNT MATRIX
# =========================================================
def build_counts_matrix(df_setting, x_order):
    if df_setting.empty:
        return pd.DataFrame(index=x_order)

    pivot = df_setting.pivot_table(
        index="xkey",
        columns="lineage",
        values="count",
        aggfunc="sum",
        fill_value=0
    )
    return pivot.reindex(x_order, fill_value=0)


# =========================================================
# TASK 13 — STYLE PLOT AXES
# =========================================================
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.8)
    ax.grid(axis="x", alpha=0.03)
    ax.set_facecolor("#f7f7f7")


# =========================================================
# TASK 14 — ADD POSITIVITY LINE WITH FIXED 0–100 SCALE
# =========================================================
def add_pos_line(ax, xvals, series, show_ylabel=False):
    ax2 = ax.twinx()

    ax2.plot(
        xvals,
        series.values,
        color="purple",
        linewidth=2.4,
        solid_capstyle="round",
        zorder=5
    )

    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.tick_params(axis="y", colors="purple", labelsize=8)

    if show_ylabel:
        ax2.set_ylabel("Positivity rate (%)", color="purple", fontsize=10, fontweight="bold")
    else:
        ax2.set_yticklabels([])

    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    return ax2


# =========================================================
# TASK 15 — CREATE FINAL FIGURE
# =========================================================
def make_plot(pathogen, bars, positivity, out_file):
    settings = ["Community", "Health Care Facility", "Wastewater"]

    b = bars[bars["pathogen"] == pathogen].copy()
    p = positivity[positivity["pathogen"] == pathogen].copy()

    if b.empty and p.empty:
        print(f"No data for {pathogen}")
        return

    if not b.empty:
        b = b[b["lineage"].notna()].copy()
        b["lineage"] = b["lineage"].apply(clean_lineage_name)
        b = b[b["lineage"].astype(str).str.strip().ne("Positive (untyped)")].copy()

    lineages = sorted(b["lineage"].dropna().unique().tolist()) if not b.empty else []
    if len(lineages) == 0:
        print(f"No typed lineages for {pathogen}")
        return

    color_map = build_color_map(lineages)

    x_order = [f"{y}_W{w}" for y, w in weeks]
    x_labels = labels

    if not b.empty:
        b["xkey"] = b["year"].astype(str) + "_W" + b["week"].astype(str)
    if not p.empty:
        p["xkey"] = p["year"].astype(str) + "_W" + p["week"].astype(str)

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

    for i, setting in enumerate(settings):
        ax = axes[i]
        ds = b[b["setting"] == setting].copy()
        pivot = build_counts_matrix(ds, x_order)

        bottom = np.zeros(len(x_order), dtype=float)

        for lin in pivot.columns:
            vals = pivot[lin].values
            if np.nanmax(vals) <= 0:
                continue

            # Style only "Not detected (To repeat)" as dashed empty bars.
            if "not detected (to repeat)" in str(lin).lower():
                ax.bar(
                    x,
                    np.where(vals > 0, 1, 0),
                    bottom=0,
                    facecolor="none",
                    edgecolor="gray",
                    linestyle="--",
                    linewidth=1.2,
                    width=0.82,
                    zorder=1
                )
            else:
                ax.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=color_map.get(lin, "#CCCCCC"),
                    edgecolor="none",
                    width=0.82,
                    zorder=2
                )
                bottom += vals

        ax.set_title(setting, fontsize=14, fontweight="bold")
        ax.set_ylabel("Count", fontsize=11, fontweight="bold")
        style_axis(ax)

        max_count = float(np.nanmax(bottom)) if len(bottom) else 0.0
        if max_count <= 0:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, max_count * 1.15)

        pos_series = (
            p[p["setting"] == setting]
            .groupby(["year", "week"])["positivity"]
            .mean()
            .reindex(time_index)
            .fillna(0)
        )

        add_pos_line(ax, x, pos_series, show_ylabel=True)

    sep_2024_2025 = len([(2024, w) for w in range(47, 53)]) - 0.5
    sep_2025_2026 = len([(2024, w) for w in range(47, 53)]) + len([(2025, w) for w in range(1, 54)]) - 0.5

    for ax in axes:
        ax.axvline(sep_2024_2025, color="black", linestyle="--", linewidth=1)
        ax.axvline(sep_2025_2026, color="black", linestyle="--", linewidth=1)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(x_labels, rotation=90, fontsize=8)
    axes[-1].set_xlabel("Epidemiological Week", fontsize=12, fontweight="bold")

    handles = [
        Patch(facecolor=color_map[lab], edgecolor="none", label=lab)
        for lab in lineages
    ]
    handles.append(
        plt.Line2D([0], [0], color="purple", linewidth=2.4, label="Positivity rate")
    )

    fig.legend(
        handles=handles,
        loc="center right",
        title=f"{pathogen} lineage",
        frameon=False,
        fontsize=9,
        title_fontsize=11,
        ncol=1
    )

    fig.suptitle(
        f"{pathogen} weekly lineage distribution and positivity",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    subprocess.run(["open", str(out_file)])


# =========================================================
# TASK 16 — RUN FULL ANALYSIS
# =========================================================
def main():
    lineage = load_lineage_counts()
    positivity, ww_pos = compose_positivity_table()

    bars_all = lineage.copy()

    if not bars_all.empty:
        bars_all = keep_only_weeks_of_interest(bars_all)
    if not positivity.empty:
        positivity = keep_only_weeks_of_interest(positivity)

    make_plot("Influenza", bars_all, positivity, OUT_FLU)

    positivity.to_csv(OUT_POS_WEEKLY, index=False)

    print(f"Saved: {OUT_FLU}")
    print(f"Saved: {OUT_POS_WEEKLY}")


if __name__ == "__main__":
    main()

