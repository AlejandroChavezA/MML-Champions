#!/usr/bin/env python3
import csv
import sys


def load_mapping(mapping_csv_path: str) -> dict:
    alias_to_can = {}
    with open(mapping_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if not row or len(row) < 2:
                continue
            alias_raw = row[0].strip().strip('"')
            canonical = row[1].strip().strip('"')
            # Normalize alias: remove leading 'v ' if present and trim
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
    # case-insensitive fallback
    lower = key.lower()
    for alias, can in mapping.items():
        if alias.lower() == lower:
            return can
    return key


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/unify_all_teams_with_mapping.py <mapping_csv> <input_all_teams_csv> <output_csv>")
        sys.exit(2)
    mapping_csv, input_csv, output_csv = sys.argv[1], sys.argv[2], sys.argv[3]

    mapping = load_mapping(mapping_csv)
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.reader(f_in)
        header = next(reader, None)
        if header is None:
            print("Input CSV is empty.")
            sys.exit(1)
        # Identify the team_name column index
        name_col = 0
        for i, h in enumerate(header):
            if isinstance(h, str) and h.strip().lower() in ("team_name", "team", "name"):
                name_col = i
                break

        rows_out = []
        for row in reader:
            if not row:
                continue
            original = row[name_col] if name_col < len(row) else row[0]
            canon = canonicalize_name(original, mapping)
            new_row = list(row)
            new_row[name_col] = canon
            rows_out.append(new_row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for r in rows_out:
            writer.writerow(r)

    print(f"Wrote {len(rows_out)} canonicalized rows to {output_csv}")


if __name__ == "__main__":
    main()
