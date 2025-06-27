#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import polars as pl
from datetime import date

# Indicators
sys.path.append('indicators')
from indicators.probability_trend import ProbTrend
ALL = ["ProbTrend"]

def load_test_data(data_file='./ETH.csv'):
    """Load test data from CSV file."""
    try:
        df = pl.read_csv(data_file, separator=';')

        return df.with_columns(
            pl.col('date').str.to_date().alias('date'),  # Ensure date is in correct format
            pl.int_range(0, pl.len()).alias("index")
        )
    except Exception as e:
        print(f"✗ Error loading test data from {data_file}: {e}")
        sys.exit(1)

def load_expected_results(indicator):
    """Load expected results from CSV file."""
    csv_file = f"./{indicator}.csv"
    try:
        if not Path(csv_file).exists():
            print(f"✗ Expected results file not found: {csv_file}")
            return None

        expected_df = pl.read_csv(csv_file, separator=';')

        return expected_df
    except Exception as e:
        print(f"✗ Error loading expected results from {csv_file}: {e}")
        return None


def run_indicator(df, indicator_name):
    """Run the specified indicator with given parameters."""
    indicator_func = globals()[indicator_name]

    try:
        return indicator_func(df)
    except Exception as e:
        print(f"✗ Error running {indicator_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="Test individual indicators against expected results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--indicator', help='Name of the indicator to test', default='ALL')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed comparison results')
    parser.add_argument('--data-file', default='./ETH.csv',
                       help='Input data file (default: ETH.csv)')

    args = parser.parse_args()

    indis_to_test = f"{args.indicator}".split(',')
    if args.indicator == 'ALL':
        indis_to_test = ALL

    print(f"📁 Using data file: {args.data_file}")

    for indi in indis_to_test:
        print("-" * 50)
        print(f"🧪 Testing indicator: {indi}")

        try:
            df = load_test_data(args.data_file)
            result = run_indicator(df, indi)

            # Load expected results if available
            expected_df = load_expected_results(indi)

            new_df =  expected_df.with_columns(
                pl.col(indi).fill_null(0).cast(pl.Float64).alias('expected_result'),
                result.fill_null(0).cast(pl.Float64).alias('new_result')
            ).filter(
                (pl.col('date').str.to_date() >= date(2018,1,1)) &
                (pl.col('expected_result') != pl.col('new_result'))
            )

            if new_df.shape[0] == 0:
                print(f"✅ {indi} test PASSED")
            else:
                print(new_df.select(['date', 'expected_result', 'new_result']))
                print(f"❌ {indi} test FAILED")
        except Exception as e:
            print(f"✗ Error during testing {indi}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
