import polars as pl
import polars_talib as pltl  # This enables the .ta accessor

def _ratQ(series_pl: pl.Series, lookback: int) -> pl.Series:
    """Calculates a 2-period rational quadratic kernel weighted average."""
    weights = [(1 + (i**2) / (lookback**2 * 2.0))**(-1.0) for i in [1, 0]]

    return series_pl.fill_nan(0).fill_null(0).rolling_mean(
        window_size=2,
        weights=weights,
    )

def _internal_ProbTrend_dirmov(df: pl.DataFrame, length: int) -> tuple[pl.Series, pl.Series]:
    df_with_calculations = df.with_columns(
        pl.col('high').diff().alias("up"),
        (-pl.col('low').diff()).alias("down")
    ).with_columns(
        # Calculate directional movements
        pl.when((pl.col("up") > pl.col("down")) & (pl.col("up") > 0))
          .then(pl.col("up"))
          .otherwise(0.0).alias("_int_plusDM"),
        pl.when((pl.col("down") > pl.col("up")) & (pl.col("down") > 0))
          .then(pl.col("down"))
          .otherwise(0.0).alias("_int_minusDM")
    ).with_columns( # hasta aca estamos bien
        # Calculate true range using .ta accessor
        pl.col("close").ta.trange(pl.col("high"), pl.col("low")).alias("truerange")
    ).with_columns(
        pl.col("_int_minusDM").fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False).alias("minus_ewm"),
        pl.col("_int_plusDM").fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False).alias("plus_ewm"),
        pl.col("truerange").fill_nan(None).ewm_mean(alpha=1.0/length, adjust=False).alias("truerange_ewm")
    ).with_columns(
        plus_intermediate=(((pl.col("plus_ewm") * 100) / pl.col("truerange_ewm"))),
        minus_intermediate=(((pl.col("minus_ewm") * 100) / pl.col("truerange_ewm")))
    )

    plus = df_with_calculations.get_column("plus_intermediate").fill_null(0)
    minus = df_with_calculations.get_column("minus_intermediate").fill_null(0)

    return plus, minus

def _internal_ProbTrend_adx(df: pl.DataFrame, dilen: int, adxlen: int) -> list[pl.Series]:
    plus, minus = _internal_ProbTrend_dirmov(df, dilen) # These are Polars Series
    summ = (plus + minus)

    adx_operand = (abs(plus - minus) / pl.when(summ == 0).then(1).otherwise(summ))
    adx_ewm = adx_operand.ewm_mean(alpha=1.0/adxlen, adjust=False)
    adx = (100 * adx_ewm)
    return [adx, plus, minus]

def ProbTrend(df: pl.DataFrame,
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
              weightVWMA=5) -> pl.Series:
    lookbackPeriod = int(lookbackPeriod)
    rsiPeriod      = int(rsiPeriod)
    adxPeriod      = int(adxPeriod)
    cciPeriod      = int(cciPeriod)
    vwmaPeriod     = int(vwmaPeriod)
    weightPrice    = int(weightPrice)
    weightRSI      = int(weightRSI)
    weightADX      = int(weightADX)
    weightCCI      = int(weightCCI)
    weightVWMA     = int(weightVWMA)

    # Use Polars expressions for price change calculations
    df_with_price = df.with_columns([
        pl.col(source_col).diff().alias("priceChange")
    ]).with_columns([
        (pl.col("priceChange") / pl.col(source_col).shift(1) * 100).alias("priceChangePercentage")
    ])

    priceChangePercentage = df_with_price.get_column("priceChangePercentage")

    # _ratQ now works with polars Series
    rationalQuadraticKernel = _ratQ(priceChangePercentage, lookbackPeriod)

    # Price logic
    upwardTrends = (rationalQuadraticKernel > 0).shift(1).rolling_sum(window_size=lookbackPeriod)
    downwardTrends = (rationalQuadraticKernel < 0).shift(1).rolling_sum(window_size=lookbackPeriod)
    totalTrends = (upwardTrends + downwardTrends)
    # Replace division by zero with null, then fill null with 0, then round.
    probabilityUpward = (((upwardTrends / totalTrends.replace(0,None) * 100).round(2)).fill_null(0))
    probabilityDownward = (((downwardTrends / totalTrends.replace(0,None) * 100).round(2)).fill_null(0))

    # RSI logic using .ta accessor
    df_with_rsi = df.with_columns([
        pl.col('close').ta.rsi(timeperiod=rsiPeriod).alias("rsiValue")
    ])
    rsiValue = df_with_rsi.get_column("rsiValue")

    upwardTrendsRSI = (rsiValue > 50).shift(1).rolling_sum(window_size=lookbackPeriod)
    downwardTrendsRSI = (rsiValue < 50).shift(1).rolling_sum(window_size=lookbackPeriod)
    totalTrendsRSI = (upwardTrendsRSI + downwardTrendsRSI)
    probabilityUpwardRSI = (((upwardTrendsRSI / totalTrendsRSI.replace(0,None) * 100).round(2)).fill_null(0))
    probabilityDownwardRSI = (((downwardTrendsRSI / totalTrendsRSI.replace(0,None) * 100).round(2)).fill_null(0))

    # ADX logic
    adx, plus, minus = _internal_ProbTrend_adx(df, adxPeriod, adxPeriod) # These are Polars Series

    # The complex apply loop was already replaced by a vectorized version.
    rolled_adx_condition = (adx > 25).shift(1).rolling_sum(window_size=lookbackPeriod)

    upwardTrendsADX = pl.when(plus > minus).then(rolled_adx_condition).otherwise(0).cast(pl.Float64)
    downwardTrendsADX = pl.when(plus < minus).then(rolled_adx_condition).otherwise(0).cast(pl.Float64)

    totalTrendsADX = (upwardTrendsADX + downwardTrendsADX)
    probabilityUpwardADX = (((upwardTrendsADX / totalTrendsADX.replace(0,None) * 100).round(2)).fill_null(0))
    probabilityDownwardADX = (((downwardTrendsADX / totalTrendsADX.replace(0,None) * 100).round(2)).fill_null(0))

    # CCI logic using .ta accessor
    df_with_cci = df.with_columns([
        pl.col("close").ta.cci(pl.col("close"), pl.col("close"), timeperiod=cciPeriod).alias("cciValue")
    ])
    cciValue = df_with_cci.get_column("cciValue")

    upwardTrendsCCI = (cciValue > 0).shift(1).rolling_sum(window_size=lookbackPeriod)
    downwardTrendsCCI = (cciValue < 0).shift(1).rolling_sum(window_size=lookbackPeriod)
    totalTrendsCCI = (upwardTrendsCCI + downwardTrendsCCI)
    probabilityUpwardCCI = (((upwardTrendsCCI / totalTrendsCCI.replace(0,None) * 100).round(2)).fill_null(0))
    probabilityDownwardCCI = (((downwardTrendsCCI / totalTrendsCCI.replace(0,None) * 100).round(2)).fill_null(0))

    # VWMA logic using .ta accessor for SMA calculations
    df_with_vwma = df.with_columns([
        # Calculate VWMA components using .ta accessor
        (pl.col('volume') * pl.col('close')).ta.sma(timeperiod=vwmaPeriod).alias("sma_vol_x_close"),
        pl.col('volume').ta.sma(timeperiod=vwmaPeriod).alias("sma_vol")
    ]).with_columns([
        # Calculate VWMA
        (pl.col("sma_vol_x_close") / pl.col("sma_vol").replace(0,None)).alias("vwmaValue")
    ])

    vwmaValue = df_with_vwma.get_column("vwmaValue")
    close_pl = df.get_column('close') # Re-fetch if not available, or pass around

    upwardTrendsVWMA = (vwmaValue > close_pl).shift(1).rolling_sum(window_size=lookbackPeriod)
    downwardTrendsVWMA = (vwmaValue < close_pl).shift(1).rolling_sum(window_size=lookbackPeriod)
    totalTrendsVWMA = (upwardTrendsVWMA + downwardTrendsVWMA)
    probabilityUpwardVWMA = (((upwardTrendsVWMA / totalTrendsVWMA.replace(0,None) * 100).round(2)).fill_null(0))
    probabilityDownwardVWMA = (((downwardTrendsVWMA / totalTrendsVWMA.replace(0,None) * 100).round(2)).fill_null(0))

    combinedProbabilityUpward = ((weightPrice * probabilityUpward + weightRSI * probabilityUpwardRSI + weightADX * probabilityUpwardADX + weightCCI * probabilityUpwardCCI + weightVWMA * probabilityUpwardVWMA) / (weightPrice + weightRSI + weightADX + weightCCI + weightVWMA))
    combinedProbabilityDownward = ((weightPrice * probabilityDownward + weightRSI * probabilityDownwardRSI + weightADX * probabilityDownwardADX + weightCCI * probabilityDownwardCCI + weightVWMA * probabilityDownwardVWMA) / (weightPrice + weightRSI + weightADX + weightCCI + weightVWMA))
    combinedOscillatorValue = (combinedProbabilityUpward - combinedProbabilityDownward)

    # Using pl.when for cross_up and cross_down logic
    cross_up_cond = (combinedOscillatorValue > 0) | ((combinedOscillatorValue == 0) & (combinedOscillatorValue.shift(1) < 0))
    cross_down_cond = (combinedOscillatorValue < 0) | ((combinedOscillatorValue == 0) & (combinedOscillatorValue.shift(1) > 0))

    return pl.when(cross_up_cond.fill_null(False)).then(1.0).when(cross_down_cond.fill_null(False)).then(-1.0).otherwise(None).forward_fill()
