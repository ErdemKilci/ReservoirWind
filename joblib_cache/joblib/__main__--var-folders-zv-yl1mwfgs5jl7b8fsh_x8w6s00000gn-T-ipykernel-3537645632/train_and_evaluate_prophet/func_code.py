# first line: 1
@prophet_memory.cache
def train_and_evaluate_prophet(
    df_train,
    df_test,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    add_country_holidays=True,
    add_lag_features=True,
    fourier_order_yearly=10,
    fourier_order_weekly=5,
    fourier_order_daily=3,
    seed=42
):
    """
    Train Prophet model with advanced configuration and evaluate on test data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Prepare data in prophet format with additional features
    train_prophet = prepare_prophet_data(df_train, add_features=add_lag_features)
    test_prophet = prepare_prophet_data(df_test, add_features=add_lag_features)

    # Create and train Prophet model with advanced configuration
    start_time = time.time()

    model = Prophet(
        yearly_seasonality=False,  # We'll add this with custom Fourier order
        weekly_seasonality=False,  # We'll add this with custom Fourier order
        daily_seasonality=False,   # We'll add this with custom Fourier order
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        interval_width=0.95
    )

    # Add custom seasonalities with higher Fourier orders for better fit
    if yearly_seasonality:
        model.add_seasonality(name='yearly', period=365.25, fourier_order=fourier_order_yearly)

    if weekly_seasonality:
        model.add_seasonality(name='weekly', period=7, fourier_order=fourier_order_weekly)

    if daily_seasonality:
        model.add_seasonality(name='daily', period=1, fourier_order=fourier_order_daily)

    # Add country holidays if requested (often important for load forecasting)
    if add_country_holidays:
        model.add_country_holidays(country_name='NO')  # Change to appropriate country

    # Add lag features as regressors - these are CRITICAL for load forecasting
    if add_lag_features:
        if 'lag_1h' in train_prophet.columns:
            model.add_regressor('lag_1h')
        if 'lag_24h' in train_prophet.columns:
            model.add_regressor('lag_24h')

        # Add other features from your ESN model
        if 'hour' in train_prophet.columns:
            model.add_regressor('hour')
        if 'weekday' in train_prophet.columns:
            model.add_regressor('weekday')
        if 'month' in train_prophet.columns:
            model.add_regressor('month')

        # Add temperature if available
        if 'temperature' in train_prophet.columns:
            model.add_regressor('temperature')

        # Add any additional features
        additional_regressors = [col for col in train_prophet.columns 
                                if col.startswith('feature_')]
        for regressor in additional_regressors:
            model.add_regressor(regressor)

    # Fit the model
    model.fit(train_prophet)
    train_time = time.time() - start_time

    # Make predictions on test data
    future = test_prophet.copy()
    forecast = model.predict(future)

    # Extract predictions
    y_pred_prophet = forecast['yhat'].values.reshape(-1, 1)
    y_test = test_prophet['y'].values.reshape(-1, 1)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred_prophet)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
    mape = np.mean(np.abs((y_test - y_pred_prophet) / y_test)) * 100
    r2 = r2_score(y_test, y_pred_prophet)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'preds': y_pred_prophet,
        'train_time': train_time,
        'forecast': forecast  # Save full forecast for component plots
    }

    return metrics, model
