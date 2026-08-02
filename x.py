#!/usr/bin/env python3
"""
x.py

Explore one or more InterpBench reports.

Usage
-----
python x.py report1.report [report2.report ...]
"""

import sys
import interpbench as ib


def main():
    if len(sys.argv) < 2:
        print("Usage: python x.py <report1.report> [report2.report ...]")
        sys.exit(1)

    merged_report = None

    for filename in sys.argv[1:]:
        try:
            report = ib.Report.load(filename)
        except Exception as e:
            print(f"Error loading report '{filename}': {e}")
            continue

        if merged_report is None:
            merged_report = report
        else:
            merged_report.extend(report)   # or: merged_report += report

    if merged_report is None:
        print("No reports were successfully loaded.")
        sys.exit(1)

    ib.plot(merged_report)


if __name__ == "__main__":
    main()