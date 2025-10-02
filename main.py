import pandas as pd
from functools import reduce
from pathlib import Path

SKATER_WEIGHTS = {
    "G": 3.0,
    "A": 2.0,
    "PIM": 0.25,
    "PPP": 0.5,
    "SHP": 1.0,
    "GWG": 1.0,
    "FOW": 0.1,
    "FOL": -0.1,
    "SOG": 0.25,
    "HIT": 0.1,
    "BLK": 0.5,
}

GOALIE_WEIGHTS = {
    "W": 4.0,
    "GA": -1.0,
    "SV": 0.2,
    "SO": 3.0,
    "T/O": 1.5,
}

def compute_skater_score(stats):
    g = stats.get("G", 0) or 0
    a = stats.get("A", 0) or 0
    pim = stats.get("PIM", 0) or 0
    ppg = stats.get("PPG", 0) or 0
    ppa = stats.get("PP", 0) or 0
    shg = stats.get("SHG", 0) or 0
    sha = stats.get("SH", 0) or 0
    gwg = stats.get("GWG", 0) or 0
    fow = stats.get("FOW", 0) or 0
    fol = stats.get("FOL", 0) or 0
    sog = stats.get("SOG", 0) or 0
    hit = stats.get("HIT", 0) or 0
    blk = stats.get("BLK", 0) or 0

    ppp = ppg + ppa # powerplay points
    shp = shg + sha # short-handed points

    score = (
        SKATER_WEIGHTS["G"] * g
        + SKATER_WEIGHTS["A"] * a
        + SKATER_WEIGHTS["PIM"] * pim
        + SKATER_WEIGHTS["PPP"] * ppp
        + SKATER_WEIGHTS["SHP"] * shp
        + SKATER_WEIGHTS["GWG"] * gwg
        + SKATER_WEIGHTS["FOW"] * fow
        + SKATER_WEIGHTS["FOL"] * fol
        + SKATER_WEIGHTS["SOG"] * sog
        + SKATER_WEIGHTS["HIT"] * hit
        + SKATER_WEIGHTS["BLK"] * blk
    )
    return score

def compute_goalie_score(stats):
    # Original goalie stats
    w = stats.get("W", 0) or 0
    ga = stats.get("GA", 0) or 0
    ot = stats.get("T/O", 0) or 0
    so = stats.get("SO", 0) or 0
    sv = stats.get("SV", 0) or 0

    score = (
        GOALIE_WEIGHTS["W"] * w
        + GOALIE_WEIGHTS["GA"] * ga
        + GOALIE_WEIGHTS["SV"] * sv
        + GOALIE_WEIGHTS["SO"] * so
        + GOALIE_WEIGHTS["T/O"] * ot
    )
    return score

def format_seasons(start_year, end_year):
    player_dfs = []
    for season in range(start_year, end_year):
        season_formatted = f"{season}{season+1}"
        skaters_file = f"./skaters/skaters_{season_formatted}.csv"
        goalies_file = f"./goalies/goalies_{season_formatted}.csv"

        skaters_df = pd.read_csv(skaters_file).drop_duplicates(subset=["-9999"], keep="first")
        goalies_df = pd.read_csv(goalies_file).drop_duplicates(subset=["-9999"], keep="first")
        
        merged_df = pd.merge(
            skaters_df,
            goalies_df,
            on=["-9999", "Player", "Pos", "G", "A", "PIM"],
            how="outer"
        )
        merged_df.fillna(0, inplace=True)
        merged_df = merged_df[[c for c in merged_df.columns if c != "-9999"] + ["-9999"]]

        scores = []
        for _, row in merged_df.iterrows():
            stats = row.to_dict()

            if row["Pos"] in ["G"]:
                score = compute_goalie_score(stats)
            else:
                score = compute_skater_score(stats)

            scores.append(score)
        merged_df["FantasyScore"] = scores

        merged_df["FantasyScore"] = merged_df["FantasyScore"].map("{:.2f}".format)

        player_dfs.append(merged_df)
    return player_dfs

def compute_fantasy_rankings(player_dfs):
    positions_map = {
        "C": "centers",
        "LW": "left_wings",
        "RW": "right_wings",
        "D": "defensemen",
        "G": "goalies"
    }

    # Read CSVs and keep relevant columns
    dfs = []
    for i, df in enumerate(player_dfs, 1):
        df = df[['-9999', 'Player', 'Pos', 'FantasyScore']]
        df = df.rename(columns={'FantasyScore': f'score_{i}'})
        dfs.append(df)

    # Merge all seasons on unique player ID (-9999) and Player name
    combined = reduce(lambda left, right: pd.merge(
        left, right, on=['-9999', 'Player'], how='outer', suffixes=('', '_y')
    ), dfs)

    # Collect all Pos columns
    pos_cols = [c for c in combined.columns if c.startswith('Pos')]

    # Take the first non-null value across these columns
    combined['Pos'] = combined[pos_cols].bfill(axis=1).iloc[:, 0]

    # Drop the old Pos columns except the new one
    combined = combined.drop(columns=[c for c in pos_cols if c != 'Pos'])

    # Compute average FantasyScore across seasons
    score_cols = [c for c in combined.columns if c.startswith('score_')]
    combined['avg_score'] = combined[score_cols].astype(float).mean(axis=1, skipna=True)
    combined['appearances'] = combined[score_cols].notna().sum(axis=1)

    # Reorder columns
    combined = combined.drop(columns="-9999")
    cols_order = ['Player', 'Pos', 'avg_score', 'appearances'] + score_cols
    combined = combined[cols_order]

    # Sort by avg_score descending
    combined["avg_score"] = combined["avg_score"].astype(float)
    combined = combined.sort_values(by='avg_score', ascending=False).reset_index(drop=True)
    combined.index += 1  # rank starting at 1

    # Format avg_score to 2 decimals
    combined['avg_score'] = combined['avg_score'].apply(lambda x: f"{x:.2f}")

    combined.to_csv(f"fantasy_rankings.csv", index=True)
    print(f"Combined CSV with all positions created.")

    # Create separate CSVs for each position
    for code, filename in positions_map.items():
        pos_df = combined[combined['Pos'] == code].copy()
        pos_df["avg_score"] = pos_df["avg_score"].astype(float)
        pos_df = pos_df.sort_values(by='avg_score', ascending=False).reset_index(drop=True)
        pos_df.index += 1
        out_csv = f"top_{filename}.csv"
        pos_df.to_csv(out_csv, index=True)
        print(f"Wrote {len(pos_df)} players to: {out_csv}")

def main():
    season_rankings = format_seasons(2021, 2025)
    compute_fantasy_rankings(season_rankings)

if __name__ == "__main__":
    main()

