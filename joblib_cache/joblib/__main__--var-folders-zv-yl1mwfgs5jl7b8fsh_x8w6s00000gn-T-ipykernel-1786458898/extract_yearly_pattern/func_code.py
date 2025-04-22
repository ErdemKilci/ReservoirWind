# first line: 1
@memory.cache
def extract_yearly_pattern(daily_resid):
    # Make sure the yearly trend parameter is valid
    yearly_trend = min(len(daily_resid) - 1, 731)  # 2 years or less
    if yearly_trend % 2 == 0:
        yearly_trend -= 1  # Make sure it's odd

    # Apply STL again on the daily residuals to extract yearly pattern
    yearly_stl = STL(
        daily_resid,
        period=365,  # Yearly seasonality
        seasonal=7,  # Less smoothing for yearly pattern
        trend=yearly_trend,  # Proper trend value
        robust=True
    )

    return yearly_stl.fit()
