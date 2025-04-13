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
    seed=42
):
    """
    Train Prophet model and evaluate on test data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Prepare data in prophet format
    train_prophet = prepare_prophet_data(df_train)
    test_prophet = prepare_prophet_data(df_test)

    # Create and train Prophet model
    start_time = time.time()

    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        interval_width=0.95
    )

    # Add additional regressors if needed (using some of your existing features)
    # This provides similar information as your 'features' in the main code
    # if 'Hour' in df_train.columns:
    #     train_prophet['hour'] = df_train['Hour']
    #     test_prophet['hour'] = df_test['Hour']
    #     model.add_regressor('hour')

    # Fit the model
    model.fit(train_prophet)
    train_time = time.time() - start_time

    # Make predictions on test data
    future = test_prophet[['ds']].copy()
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
