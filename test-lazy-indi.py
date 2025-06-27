#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import polars as pl
from datetime import date

# Indicators
sys.path.append('indicators')
from indicators.probability_trend_lazy import ProbTrend

ALL = ["ProbTrend"]

def load_test_data(data_file=Path(__file__).parent / 'ETH.csv'):
    """Load test data from CSV file."""
    try:
        # Use scan_csv for lazy loading
        df_lazy = pl.scan_csv(data_file, separator=';')

        return df_lazy.with_columns(
            pl.col('date').str.to_date().alias('date'),  # Ensure date is in correct format
            pl.int_range(0, pl.len()).alias("index")
        )
    except Exception as e:
        print(f"✗ Error loading test data from {data_file}: {e}")
        sys.exit(1)

def load_expected_results(indicator):
    """Load expected results from CSV file."""
    csv_file = Path(__file__).parent / f"{indicator}.csv"
    try:
        if not Path(csv_file).exists():
            print(f"✗ Expected results file not found: {csv_file}")
            return None # Or return an empty LazyFrame: pl.LazyFrame()

        # Use scan_csv for lazy loading and ensure 'date' column is pl.Date
        expected_df_lazy = pl.scan_csv(csv_file, separator=';').with_columns(
            pl.col('date').str.to_date().alias('date')
        )
        return expected_df_lazy
    except Exception as e:
        print(f"✗ Error loading expected results from {csv_file}: {e}")
        return None

def run_indicator(df, indicator_name):
    """Run the specified indicator with given parameters."""
    indicator_func = globals()[indicator_name]

    try:
        # df is a LazyFrame. Indicators are now expected to handle LazyFrames
        # and return a Polars Expression.
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
    parser.add_argument('--data-file', default=Path(__file__).parent / 'ETH.csv',
                       help='Input data file (default: ETH.csv)')

    args = parser.parse_args()

    indis_to_test = f"{args.indicator}".split(',')
    if args.indicator == 'ALL':
        indis_to_test = ALL

    print(f"📁 Using data file: {args.data_file}")

    df_lazy = load_test_data(args.data_file) # LazyFrame

    for indi in indis_to_test:
        print("-" * 50)

        # if indi == "ProbTrend":
        #     print(f"⚠️ Skipping ProbTrend due to known Polars Rust panic during execution.")
        #     continue

        print(f"🧪 Testing indicator: {indi}")

        try:

            # Run indicator - it now takes a LazyFrame and returns an Expression
            # Use df_lazy directly (or a clone). Result is an expression.
            result_expr = run_indicator(df_lazy, indi)

            # Load expected results - this is a LazyFrame
            expected_lazy = load_expected_results(indi)

            if expected_lazy is None:
                print(f"⚠️ No expected results found for {indi}. Skipping comparison.")
                continue

            # Ensure result_series is treated as a column in the lazy pipeline
            # The 'date' column and the original indicator column (e.g., 'EnhancedHMA') are in expected_lazy
            # We are comparing the column named `indi` in `expected_lazy` with `result_series`

            # Create the comparison LazyFrame
            # We need to align expected_lazy (which has 'date' and the original indicator column)
            # with result_series.
            # Assuming expected_lazy has 'date' and a column named `indi` with the expected values.
            # result_expr is an expression that should be evaluated on df_lazy.

            # 1. Evaluate result_expr in the context of df_lazy
            # Ensure result_expr is indeed an expression; if it's a literal Series (e.g. from non-refactored indicator), convert it.
            df_with_new_result_lazy = df_lazy.with_columns(new_result_calculated=result_expr)

            # 2. Join df_with_new_result_lazy (contains date, new_result_calculated, and original OHLCV data)
            # with expected_lazy (contains date, and 'indi' column for expected result).
            # Rename the indicator's column in expected_lazy to avoid name clashes and for clarity.
            expected_lazy_prepared = expected_lazy.rename({indi: 'expected_value_from_file'})

            # Select only necessary columns before join to avoid large intermediate frames if not needed,
            # though lazy execution should optimize this. 'date' is key.
            # df_with_new_result_lazy might have many columns. We need 'date' and 'new_result_calculated'.
            # expected_lazy_prepared has 'date' and 'expected_value_from_file'.
            joined_lazy = df_with_new_result_lazy.select(['date', 'new_result_calculated']).join(
                expected_lazy_prepared, # Contains 'date' and 'expected_value_from_file'
                on='date',
                how='inner'
            )

            # 3. Now build the comparison logic on `joined_lazy`
            comparison_lazy = joined_lazy.with_columns(
                pl.col('expected_value_from_file').fill_null(0).cast(pl.Float64).alias('expected_result_cast'),
                pl.col('new_result_calculated').fill_null(0).cast(pl.Float64).alias('new_result_cast')
            ).filter(
                (pl.col('date') >= date(2018,1,1)) & # Date filter from original data
                (pl.col('expected_result_cast') != pl.col('new_result_cast'))
            )

            # Collect the results of the comparison
            # diff_df = comparison_lazy.collect(engine='gpu')
            diff_df = comparison_lazy.collect() #optimizations=pl.QueryOptFlags(fast_projection = True))

            if diff_df.shape[0] == 0:
                print(f"✅ {indi} test PASSED")
            else:
                print(diff_df.select(['date', 'expected_result_cast', 'new_result_cast']))
                print(f"❌ {indi} test FAILED")
        except Exception as e:
            print(f"✗ Error during testing {indi}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
