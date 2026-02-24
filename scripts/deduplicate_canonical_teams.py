#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict


def load_mapping(mapping_csv_path: str) -> dict:
    alias_to_can = {}
    with open(mapping_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            alias_raw = row[0].strip().strip('"')
            canonical = row[1].strip().strip('"')
            alias = alias_raw
            if alias.lower().startswith("v "):
                alias = alias[2:].strip()
            alias_to_can[alias] = canonical
    return alias_to_can


def canonicalize_name(name: str, mapping: dict) -> str:
    if not name:
        return name
    key = name.strip()
    if key in mapping:
        return mapping[key]
    lower = key.lower()
    for alias, can in mapping.items():
        if alias.lower() == lower:
            return can
    return key


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/deduplicate_canonical_teams.py <mapping_csv> <input_all_teams_csv> <output_csv>")
        sys.exit(2)
    mapping_csv, input_csv, output_csv = sys.argv[1], sys.argv[2], sys.argv[3]

    mapping = load_mapping(mapping_csv)
    canonical_rows = defaultdict(list)  # canonical -> list of seasons strings

    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        header = next(reader, None)
        if header is None:
            print("Input CSV is empty.")
            sys.exit(1)
        name_col = 0
        for i, h in enumerate(header):
            if isinstance(h, str) and h.strip().lower() in ("team_name", "team", "name"):
                name_col = i
                break

        # Find seasons column index (last column)
        seasons_col = len(header) - 1
        for row in reader:
            if not row:
                continue
            team_raw = row[name_col] if name_col < len(row) else row[0]
            canon = canonicalize_name(team_raw, mapping)
            seasons = row[seasons_col] if seasons_col < len(row) else ""
            canonical_rows[canon].append(seasons)

    # Deduplicate seasons per canonical and normalize formatting
    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["team_name","seasons"])
        for canon in sorted(canonical_rows.keys()):
            # Merge all seasons strings and split on comma to deduplicate tokens
            all_tokens = []
            for s in canonical_rows[canon]:
                if not s:
                    continue
                parts = [p.strip() for p in s.split(",") if p.strip()]
                all_tokens.extend(parts)
            unique = sorted(set(all_tokens))
            seasons_out = ", ".join(unique)
            writer.writerow([canon, seasons_out])

    print(f"Wrote canonical, deduplicated teams to {output_csv} (count: {len(canonical_rows)})")


if __name__ == "__main__":
    main()
