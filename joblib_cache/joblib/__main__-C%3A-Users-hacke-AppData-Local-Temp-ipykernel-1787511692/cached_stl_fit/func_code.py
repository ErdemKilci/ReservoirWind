# first line: 1
@memory.cache
def cached_stl_fit(series, period, seasonal, trend, seasonal_deg, trend_deg, robust):
    stl = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        seasonal_deg=seasonal_deg,
        trend_deg=trend_deg,
        robust=robust
    )
    return stl.fit()
