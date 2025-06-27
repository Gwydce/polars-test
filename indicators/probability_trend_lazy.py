import polars as pl
import polars_talib as pltl

# Refactored to accept and return Polars Expression
def _ratQ(input_expr: pl.Expr, lookback: int) -> pl.Expr:
    """Calculates a 2-period rational quadratic kernel weighted average."""
    weights = [(1 + (i**2) / (lookback**2 * 2.0))**(-1.0) for i in [1, 0]]
    # input_expr is expected to be priceChangePercentage_expr
    return input_expr.fill_nan(0).fill_null(0).rolling_mean(
        window_size=2,
        weights=weights,
    )

# Refactored to accept LazyFrame and return tuple of Polars Expressions
def _internal_ProbTrend_dirmov(df_lazy: pl.LazyFrame, length: int) -> tuple[pl.Expr, pl.Expr]:
    up_expr = pl.col('high').diff()
    down_expr = -pl.col('low').diff()

    int_plusDM_expr = pl.when((up_expr > down_expr) & (up_expr > 0)) \
                        .then(up_expr) \
                        .otherwise(pl.lit(0.0))
    int_minusDM_expr = pl.when((down_expr > up_expr) & (down_expr > 0)) \
                         .then(down_expr) \
                         .otherwise(pl.lit(0.0))

    # Revert to original .ta accessor style for TRANGE
    truerange_expr = pl.col("close").ta.trange(high=pl.col("high"), low=pl.col("low"))

    minus_ewm_expr = int_minusDM_expr.fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False)
    plus_ewm_expr = int_plusDM_expr.fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False)
    truerange_ewm_expr = truerange_expr.fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False)

    safe_truerange_ewm_expr = pl.when(truerange_ewm_expr == 0).then(None).otherwise(truerange_ewm_expr)

    plus_intermediate_expr = (((plus_ewm_expr * 100) / safe_truerange_ewm_expr)).fill_null(0)
    minus_intermediate_expr = (((minus_ewm_expr * 100) / safe_truerange_ewm_expr)).fill_null(0)

    return plus_intermediate_expr, minus_intermediate_expr

# Refactored to accept LazyFrame and return list of Polars Expressions
def _internal_ProbTrend_adx(df_lazy: pl.LazyFrame, dilen: int, adxlen: int) -> list[pl.Expr]:
    plus_expr, minus_expr = _internal_ProbTrend_dirmov(df_lazy, dilen)
    summ_expr = (plus_expr + minus_expr)

    safe_summ_expr = pl.when(summ_expr == 0).then(pl.lit(1.0)).otherwise(summ_expr) # Avoid division by zero
    adx_operand_expr = (abs(plus_expr - minus_expr) / safe_summ_expr)
    adx_ewm_expr = adx_operand_expr.ewm_mean(alpha=1.0/adxlen, adjust=False)
    adx_expr = (100 * adx_ewm_expr)
    return [adx_expr, plus_expr, minus_expr]

# Refactored main function
def ProbTrend(df_lazy: pl.LazyFrame, # Changed df to df_lazy
    lookbackPeriod=43,
    source_col='close',
    rsiPeriod=14,
    adxPeriod=6,
    cciPeriod=13,
    vwmaPeriod=9,
    weightPrice=50,
    weightRSI=20,
    weightADX=15,
    weightCCI=10,
    weightVWMA=5) -> pl.Expr: # Returns Expression
    lookbackPeriod = int(lookbackPeriod)
    rsiPeriod      = int(rsiPeriod)
    adxPeriod      = int(adxPeriod)
    cciPeriod      = int(cciPeriod)
    vwmaPeriod     = int(vwmaPeriod)
    # Weights should be float for lazy operations
    weightPrice    = float(weightPrice)
    weightRSI      = float(weightRSI)
    weightADX      = float(weightADX)
    weightCCI      = float(weightCCI)
    weightVWMA     = float(weightVWMA)

    source_expr = pl.col(source_col)
    priceChange_expr = source_expr.diff()
    # Ensure source_expr.shift(1) cannot be zero for division
    safe_shifted_source_expr = pl.when(source_expr.shift(1) == 0).then(None).otherwise(source_expr.shift(1))
    priceChangePercentage_expr = (priceChange_expr / safe_shifted_source_expr * 100)
    rationalQuadraticKernel_expr = _ratQ(priceChangePercentage_expr, lookbackPeriod)

    # Price logic expressions
    upwardTrends_expr = (rationalQuadraticKernel_expr > pl.lit(0.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    downwardTrends_expr = (rationalQuadraticKernel_expr < pl.lit(0.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    totalTrends_expr = (upwardTrends_expr + downwardTrends_expr)
    safe_totalTrends_expr = pl.when(totalTrends_expr == 0).then(None).otherwise(totalTrends_expr)
    probabilityUpward_expr = (((upwardTrends_expr / safe_totalTrends_expr * 100).round(2)).fill_null(0))
    probabilityDownward_expr = (((downwardTrends_expr / safe_totalTrends_expr * 100).round(2)).fill_null(0))

    # RSI logic expressions
    rsiValue_expr = pl.col('close').ta.rsi(timeperiod=rsiPeriod)
    upwardTrendsRSI_expr = (rsiValue_expr > pl.lit(50.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    downwardTrendsRSI_expr = (rsiValue_expr < pl.lit(50.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    totalTrendsRSI_expr = (upwardTrendsRSI_expr + downwardTrendsRSI_expr)
    safe_totalTrendsRSI_expr = pl.when(totalTrendsRSI_expr == 0).then(None).otherwise(totalTrendsRSI_expr)
    probabilityUpwardRSI_expr = (((upwardTrendsRSI_expr / safe_totalTrendsRSI_expr * 100).round(2)).fill_null(0))
    probabilityDownwardRSI_expr = (((downwardTrendsRSI_expr / safe_totalTrendsRSI_expr * 100).round(2)).fill_null(0))

    # ADX logic expressions
    adx_expr, plus_expr, minus_expr = _internal_ProbTrend_adx(df_lazy, adxPeriod, adxPeriod)
    rolled_adx_condition_expr = (adx_expr > pl.lit(25.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    upwardTrendsADX_expr = pl.when(plus_expr > minus_expr).then(rolled_adx_condition_expr).otherwise(pl.lit(0.0)).cast(pl.Float64)
    downwardTrendsADX_expr = pl.when(plus_expr < minus_expr).then(rolled_adx_condition_expr).otherwise(pl.lit(0.0)).cast(pl.Float64)
    totalTrendsADX_expr = (upwardTrendsADX_expr + downwardTrendsADX_expr)
    safe_totalTrendsADX_expr = pl.when(totalTrendsADX_expr == 0).then(None).otherwise(totalTrendsADX_expr)
    probabilityUpwardADX_expr = (((upwardTrendsADX_expr / safe_totalTrendsADX_expr * 100).round(2)).fill_null(0))
    probabilityDownwardADX_expr = (((downwardTrendsADX_expr / safe_totalTrendsADX_expr * 100).round(2)).fill_null(0))

    # CCI logic expressions (reverting to original .ta accessor style)
    # Original was: pl.col("close").ta.cci(pl.col("close"), pl.col("close"), timeperiod=cciPeriod)
    cciValue_expr = pl.col("close").ta.cci(high=pl.col("close"), low=pl.col("close"), timeperiod=cciPeriod) # Matched original args

    upwardTrendsCCI_expr = (cciValue_expr > pl.lit(0.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    downwardTrendsCCI_expr = (cciValue_expr < pl.lit(0.0)).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    totalTrendsCCI_expr = (upwardTrendsCCI_expr + downwardTrendsCCI_expr)
    safe_totalTrendsCCI_expr = pl.when(totalTrendsCCI_expr == 0).then(None).otherwise(totalTrendsCCI_expr)
    probabilityUpwardCCI_expr = (((upwardTrendsCCI_expr / safe_totalTrendsCCI_expr * 100).round(2)).fill_null(0))
    probabilityDownwardCCI_expr = (((downwardTrendsCCI_expr / safe_totalTrendsCCI_expr * 100).round(2)).fill_null(0))

    # VWMA logic expressions
    sma_vol_x_close_expr = (pl.col('volume') * pl.col('close')).ta.sma(timeperiod=vwmaPeriod)
    sma_vol_expr = pl.col('volume').ta.sma(timeperiod=vwmaPeriod)
    safe_sma_vol_expr = pl.when(sma_vol_expr == 0).then(None).otherwise(sma_vol_expr)
    vwmaValue_expr = sma_vol_x_close_expr / safe_sma_vol_expr

    upwardTrendsVWMA_expr = (vwmaValue_expr > pl.col('close')).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)
    downwardTrendsVWMA_expr = (vwmaValue_expr < pl.col('close')).shift(1).rolling_sum(window_size=lookbackPeriod).cast(pl.Float64)  
    totalTrendsVWMA_expr = (upwardTrendsVWMA_expr + downwardTrendsVWMA_expr)
    safe_totalTrendsVWMA_expr = pl.when(totalTrendsVWMA_expr == 0).then(None).otherwise(totalTrendsVWMA_expr)
    probabilityUpwardVWMA_expr = (((upwardTrendsVWMA_expr / safe_totalTrendsVWMA_expr * 100).round(2)).fill_null(0))
    probabilityDownwardVWMA_expr = (((downwardTrendsVWMA_expr / safe_totalTrendsVWMA_expr * 100).round(2)).fill_null(0))

    # Combined probabilities expressions
    total_weight = float(weightPrice + weightRSI + weightADX + weightCCI + weightVWMA) # Ensure float division
    combinedProbabilityUpward_expr = ((weightPrice * probabilityUpward_expr +
                                    weightRSI * probabilityUpwardRSI_expr +
                                    weightADX * probabilityUpwardADX_expr +
                                    weightCCI * probabilityUpwardCCI_expr +
                                    weightVWMA * probabilityUpwardVWMA_expr) / total_weight)
    combinedProbabilityDownward_expr = ((weightPrice * probabilityDownward_expr +
                                    weightRSI * probabilityDownwardRSI_expr +
                                    weightADX * probabilityDownwardADX_expr +
                                    weightCCI * probabilityDownwardCCI_expr +
                                    weightVWMA * probabilityDownwardVWMA_expr) / total_weight)
    combinedOscillatorValue_expr = (combinedProbabilityUpward_expr - combinedProbabilityDownward_expr)

    # Final signal expression
    cross_up_cond_expr = (combinedOscillatorValue_expr > pl.lit(0.0)) | \
                        ((combinedOscillatorValue_expr == pl.lit(0.0)) & (combinedOscillatorValue_expr.shift(1) < pl.lit(0.0)))
    cross_down_cond_expr = (combinedOscillatorValue_expr < pl.lit(0.0)) | \
                        ((combinedOscillatorValue_expr == pl.lit(0.0)) & (combinedOscillatorValue_expr.shift(1) > pl.lit(0.0)))

    signal_expr = pl.when(cross_up_cond_expr.fill_null(False)).then(pl.lit(1.0)) \
                    .when(cross_down_cond_expr.fill_null(False)).then(pl.lit(-1.0)) \
                    .otherwise(None).forward_fill()

    return signal_expr
