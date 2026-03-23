import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure absolute reproducibility of the synthetic data
np.random.seed(42)

# -----------------------------
# 1. Core Mathematical Functions
# -----------------------------
def sigmoid(z):
    """Maps unbounded linear log-odds into bounded probabilities (0 to 1)."""
    return 1 / (1 + np.exp(-z))

def calculate_attrition_probability(age, aum, equity_ratio):
    """
    Defines a simplified economic relationship between portfolio characteristics and client churn.
    Older clients exhibit higher loyalty; higher AUM reduces flight risk; aggressive equity positioning increases volatility.
    """
    logit = (
        -3.5
        + 0.015 * (age - 40)          
        - 0.0000002 * aum             
        + 1.5 * (equity_ratio - 0.4)  
    )
    return sigmoid(logit)

# -----------------------------
# 2. Data Architecture & Generation
# -----------------------------
def generate_client_data(n_clients, bank_name, age_mean, aum_mean, equity_ratio_mean):
    """Constructs synthetic portfolios representing distinct institutional risk profiles."""
    
    age = np.random.normal(age_mean, 5, n_clients).astype(int)
    aum = np.random.normal(aum_mean, aum_mean * 0.2, n_clients)
    equity_ratio = np.clip(np.random.normal(equity_ratio_mean, 0.1, n_clients), 0, 1)

    # Derive the true risk probability based on features, then execute a binomial trial
    true_probabilities = calculate_attrition_probability(age, aum, equity_ratio)
    attrition = np.random.binomial(1, true_probabilities)

    data = {
        "Institution_ID": [bank_name] * n_clients,
        "Client_Age": age,
        "AUM_EUR": aum.round(2),
        "Equity_Ratio": equity_ratio.round(2),
        "Attrition_Risk": attrition
    }
    return pd.DataFrame(data)

# Instantiate isolated datasets to reflect strict regulatory privacy constraints
bank_a = generate_client_data(1000, "Private_Wealth_Office", 60, 5000000, 0.40)
bank_b = generate_client_data(1000, "Standard_Retail_Bank", 45, 50000, 0.20)
bank_c = generate_client_data(1000, "FinTech_NeoBroker", 28, 15000, 0.85)

# -----------------------------
# 3. Localized Optimization (Gradient Descent)
# -----------------------------
def train_local_model(df, features, target, lr=0.01, epochs=1000):
    """Executes logistic regression via gradient descent entirely within the isolated data silo."""
    X = df[features].values
    y = df[target].values

    # Z-score standardization to stabilize the gradient calculations
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.c_[np.ones(X.shape[0]), X] # Append intercept

    weights = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        z = np.dot(X, weights)
        preds = sigmoid(z)

        # Calculate Binary Cross-Entropy loss (incorporating a 1e-9 smoothing term for computational stability)
        loss = -np.mean(y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9))
        losses.append(loss)

        # Derive the gradient and update weights
        gradient = np.dot(X.T, (preds - y)) / len(y)
        weights -= lr * gradient

    return weights, losses

def evaluate_model(df, weights, features, target):
    """
    Evaluates the in-sample predictive accuracy of the supplied weight matrix.
    Note: Standardization is applied locally to match the training environment methodology.
    """
    X = df[features].values
    y = df[target].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = np.c_[np.ones(X.shape[0]), X]

    preds = sigmoid(np.dot(X, weights)) > 0.5
    return (preds == y).mean()

features = ["Client_Age", "AUM_EUR", "Equity_Ratio"]
target = "Attrition_Risk"

# Train localized models behind institutional firewalls
weights_a, loss_a = train_local_model(bank_a, features, target)
weights_b, loss_b = train_local_model(bank_b, features, target)
weights_c, loss_c = train_local_model(bank_c, features, target)

# -----------------------------
# 4. Federated Aggregation
# -----------------------------
# Aggregate localized models using sample-size weighted Federated Averaging
total_clients = len(bank_a) + len(bank_b) + len(bank_c)

global_weights = (
    weights_a * len(bank_a)
    + weights_b * len(bank_b)
    + weights_c * len(bank_c)
) / total_clients

# Output operational proof of the global model's efficacy
print("--- Global Weight Vector (Intercept, Age, AUM, Equity) ---")
print(np.round(global_weights, 4))
print("\n--- Global Model Diagnostic (In-Sample Accuracy) ---")
print(f"Private Wealth Office Accuracy: {evaluate_model(bank_a, global_weights, features, target):.2%}")
print(f"Standard Retail Bank Accuracy:  {evaluate_model(bank_b, global_weights, features, target):.2%}")
print(f"FinTech NeoBroker Accuracy:     {evaluate_model(bank_c, global_weights, features, target):.2%}")

# -----------------------------
# 5. Visualization
# -----------------------------
plt.plot(loss_a, label="Private Wealth Office")
plt.plot(loss_b, label="Standard Retail Bank")
plt.plot(loss_c, label="FinTech NeoBroker")
plt.title("Optimization Dynamics: Binary Cross-Entropy Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss Function")
plt.legend()
plt.show()
