#!/usr/bin/env python3
import csv
import sys
import itertools
import unicodedata
from difflib import SequenceMatcher


def normalize(s: str) -> str:
    if s is None:
        return ""
    # Remove diacritics, lowercase, and strip non-visible chars
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_only = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower().strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_duplicates(csv_path: str, threshold: float = 0.65) -> list:
    # Read all team names
    teams = []  # list of (index, original_name, normalized)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # detect header
        header = next(reader, None)
        if header is None:
            return []
        # Determine which column is team_name based on header names
        # We expect a header like ["team_name", "seasons"]
        name_col = None
        for i, h in enumerate(header):
            if isinstance(h, str) and h.strip().lower() in ("team_name", "team", "name"):
                name_col = i
                break
        if name_col is None:
            # Fallback: assume first column is the name
            name_col = 0

        for idx, row in enumerate(reader, start=1):
            if not row:
                continue
            team_name = row[name_col] if name_col < len(row) else row[0]
            teams.append((idx, team_name, normalize(team_name)))

    # Build candidate pairs by similarity on normalized form and raw form
    results = []
    for (i1, name1, norm1), (i2, name2, norm2) in itertools.combinations(teams, 2):
        # compute similarity on raw names as well as on normalized forms
        s = similarity(name1.lower(), name2.lower())
        if s >= threshold:
            results.append({
                "idx1": i1,
                "name1": name1,
                "idx2": i2,
                "name2": name2,
                "similarity_raw": round(s, 4),
                "norm1": norm1,
                "norm2": norm2,
            })
    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity_raw"], reverse=True)
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: find_all_teams_duplicates.py <path_to_all_teams_csv> [threshold]", file=sys.stderr)
        sys.exit(2)
    csv_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    dups = find_duplicates(csv_path, threshold=threshold)
    if not dups:
        print("No potential duplicates found.")
        return
    # Output a simple CSV listing potential duplicates
    out_path = "duplicates_all_teams_candidates.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx1","name1","idx2","name2","similarity_raw","norm1","norm2"])
        for row in dups:
            writer.writerow([
                row["idx1"], row["name1"], row["idx2"], row["name2"], row["similarity_raw"], row["norm1"], row["norm2"],
            ])
    print(f"Wrote {len(dups)} potential duplicate pairs to {out_path}")


if __name__ == "__main__":
    main()
