# first line: 1
@memory.cache  # joblib cache decorator
def train_and_evaluate_model(
    model_type='lstm',        # 'lstm' or 'cnn'
    hidden_dim=64,
    num_layers=1,
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    seed=42
):
    """
    Trains either LSTM or CNN+LSTM on a single train set,
    returns final predictions on the test set and metrics.
    Caching is applied: if the same (model_type, hidden_dim, num_layers, learning_rate, epochs, batch_size, seed)
    is called again, joblib will skip re-computing and load from cache.
    """

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare data
    X_train_torch = reshape_for_lstm(X_train_torch_base)
    y_train_torch = y_train_torch_base

    # DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_all_scaled.shape[1]

    # Build model
    if model_type == 'lstm':
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1)
    else:
        model = CNNLSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    model.train()
    for epoch_i in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        # epoch_loss = total_loss / len(train_loader.dataset)
        # (Optional print) print(f"[{model_type.upper()}] Epoch {epoch_i+1}/{epochs}, Loss={epoch_loss:.6f}")

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        y_pred_test = model(reshape_for_lstm(X_test_torch))
    y_pred_test_np = y_pred_test.detach().numpy()

    # Inverse transform
    y_pred_unscaled = scalerY_full.inverse_transform(y_pred_test_np)

    # Compute errors
    mae = np.mean(np.abs(y_test_final - y_pred_unscaled))
    rmse = np.sqrt(np.mean((y_test_final - y_pred_unscaled)**2))
    mape = np.mean(np.abs((y_test_final - y_pred_unscaled) / y_test_final)) * 100
    r2 = r2_score(y_test_final, y_pred_unscaled)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'preds': y_pred_unscaled
    }
    return metrics
