import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler


# Loop through datasets for model training
for data_idx,(features,labels) in enumerate(zip(usFeatures,usLabels)):
    X_train,X_test=train_test_split(features,test_size=0.2,shuffle=False)
    y_train,y_test=train_test_split(labels,test_size=0.2,shuffle=False)
    X_train=X_train.to(device)
    X_test=X_test.to(device)
    y_train=y_train.to(device)
    y_test=y_test.to(device)
    model = LSTMModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Testing the model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, test_outputs)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, test_outputs)
    mae = mean_absolute_error(y_test, test_outputs)

    # Print evaluation metrics
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'R-squared (R2): {r2:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, test_outputs, alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(f"Dataset {data_idx + 1}: True Values vs. Predictions")
    plt.grid(True)
    plt.show()

    # Store evaluation results
    evaluation_results.append({
        "Dataset": data_idx + 1,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae,
    })