#!/usr/bin/env python3
import csv
from collections import defaultdict


def main():
    input_csv = "data/all_teams_canonical.csv"
    counts = defaultdict(int)
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0] if len(row) > 0 else ""
            counts[name] += 1

    duplicates = [(name, cnt) for name, cnt in counts.items() if cnt > 1]
    if not duplicates:
        print("No duplicate canonicals found.")
        return
    print("Duplicate canonicals found:")
    for name, cnt in sorted(duplicates, key=lambda x: x[0]):
        print(f"- {name}: {cnt} occurrences")


if __name__ == "__main__":
    main()
