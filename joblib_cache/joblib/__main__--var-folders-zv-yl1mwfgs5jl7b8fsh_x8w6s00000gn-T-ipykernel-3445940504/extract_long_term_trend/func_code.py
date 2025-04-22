# first line: 1
@memory.cache
def extract_long_term_trend(series):
    # Use a 2-year moving average to get the true trend
    return series.rolling(window=365*24*2, center=True).mean()
