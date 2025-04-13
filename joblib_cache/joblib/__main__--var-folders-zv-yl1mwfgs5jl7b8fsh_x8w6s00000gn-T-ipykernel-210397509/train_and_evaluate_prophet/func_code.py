# first line: 1
@prophet_memory.cache
def train_and_evaluate_prophet(
    df_train,
    df_test,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.01,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    add_country_holidays=True,
    add_lag_features=True,
    fourier_order_yearly=20,
    fourier_order_weekly=10,
    fourier_order_daily=10,
    seed=42
):
    np.random.seed(seed)

    train_prophet = prepare_prophet_data(df_train, add_features=add_lag_features)
    test_prophet  = prepare_prophet_data(df_test,  add_features=add_lag_features)

    start_time = time.time()

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        interval_width=0.95
    )

    # Custom seasonalities with large Fourier orders
    if yearly_seasonality:
        model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order_yearly)
    if weekly_seasonality:
        model.add_seasonality(name='weekly', period=7, fourier_order=fourier_order_weekly)
    if daily_seasonality:
        # For hourly data, a "daily" cycle might need a period of ~24 hours
        # But Prophet’s “daily_seasonality” expects period=1 (one day in “days” if ds is daily).
        # If your data is hourly, you can define a custom sub-daily seasonality with period=1 
        # for each day. For more granular sub-daily patterns, do something like:
        model.add_seasonality(name='daily', period=1, fourier_order=fourier_order_daily)

    # Optionally add a more fine-grained sub-daily seasonal component for 8- or 12-hour cycles:
    # model.add_seasonality(name='half_day', period=0.5, fourier_order=6)

    if add_country_holidays:
        model.add_country_holidays(country_name='US')  # Adjust country if relevant

    # Add regressors
    # Important: Prophet will see these regressors in log scale if we transformed them above
    if add_lag_features:
        for col in ['lag_1h','lag_24h','hour','weekday','month','temperature','day_of_year']:
            if col in train_prophet.columns:
                model.add_regressor(col)

    model.fit(train_prophet)
    train_time = time.time() - start_time

    future  = test_prophet.copy()
    forecast = model.predict(future)

    # Inverse the log transform
    y_pred_prophet = np.expm1(forecast['yhat'].values.reshape(-1, 1))
    y_test = np.expm1(test_prophet['y'].values.reshape(-1, 1))  # invert log transform of test

    mae  = mean_absolute_error(y_test, y_pred_prophet)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
    mape = np.mean(np.abs((y_test - y_pred_prophet) / y_test)) * 100
    r2   = r2_score(y_test, y_pred_prophet)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'preds': y_pred_prophet,
        'train_time': train_time,
        'forecast': forecast
    }

    return metrics, model
