# first line: 1
@prophet_memory.cache
def train_and_evaluate_prophet_extended(df_train, df_test, **kwargs):
    """
    Extended version of train_and_evaluate_prophet to handle additional parameters
    """
    train_prophet = prepare_prophet_data(df_train, add_features=kwargs.get('add_lag_features', True))
    test_prophet = prepare_prophet_data(df_test, add_features=kwargs.get('add_lag_features', True))

    # Extract parameters with defaults based on previous best config
    growth = kwargs.get('growth', 'linear')
    n_changepoints = kwargs.get('n_changepoints', 25)
    changepoint_range = kwargs.get('changepoint_range', 0.95)

    # Initialize Prophet model with expanded parameters
    model_params = {
        'growth': growth,
        'n_changepoints': n_changepoints,
        'changepoint_range': changepoint_range,
        'yearly_seasonality': kwargs.get('yearly_seasonality', False),
        'weekly_seasonality': kwargs.get('weekly_seasonality', False),
        'daily_seasonality': kwargs.get('daily_seasonality', False),
        'changepoint_prior_scale': kwargs.get('changepoint_prior_scale', 0.01),
        'seasonality_prior_scale': kwargs.get('seasonality_prior_scale', 0.01),
        'holidays_prior_scale': kwargs.get('holidays_prior_scale', 10.0),
        'interval_width': 0.95
    }

    # Add capacity parameters if using logistic growth
    if growth == 'logistic':
        # Add floor and cap to training data
        train_prophet['floor'] = kwargs.get('floor', 0)
        train_prophet['cap'] = kwargs.get('cap', train_prophet['y'].max() * 1.2)
        test_prophet['floor'] = kwargs.get('floor', 0)
        test_prophet['cap'] = kwargs.get('cap', train_prophet['y'].max() * 1.2)

    model = Prophet(**model_params)

    # Custom seasonality
    if kwargs.get("yearly_seasonality"): 
        model.add_seasonality('yearly', 365.25, fourier_order=kwargs['fourier_order_yearly'])
    if kwargs.get("weekly_seasonality"): 
        model.add_seasonality('weekly', 7, fourier_order=kwargs['fourier_order_weekly'])
    if kwargs.get("daily_seasonality"): 
        model.add_seasonality('daily', 1, fourier_order=kwargs['fourier_order_daily'])

    if kwargs.get("add_country_holidays", False):
        model.add_country_holidays("NO")

    if kwargs.get("add_lag_features", True):
        for reg in ['lag_1h', 'lag_24h', 'hour', 'weekday', 'month', 'temperature']:
            if reg in train_prophet.columns:
                model.add_regressor(reg)
        for reg in [c for c in train_prophet.columns if c.startswith("feature_")]:
            model.add_regressor(reg)

    start_time = time.time()
    model.fit(train_prophet)
    train_time = time.time() - start_time

    forecast = model.predict(test_prophet)
    y_pred = forecast['yhat'].values.reshape(-1, 1)
    y_true = test_prophet['y'].values.reshape(-1, 1)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2,
        'preds': y_pred, 'forecast': forecast, 'train_time': train_time
    }, model
