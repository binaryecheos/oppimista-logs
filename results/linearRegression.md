# 📊 Linear Regression Results - Gradient Descent Comparison

## Model Performance Summary

| Method      | Bias (θ₀) | Slope (θ₁) | Error (%) | Final Loss |
|------------|-----------|------------|-----------|------------|
| **Batch GD** | 0.9813    | 2.9987     | **<2%**   | ~0.005     |
| **SGD**      | 0.9768    | 2.9984     | **<3%**   | ~0.005     |
| **True**     | **1.0000**| **3.0000** | -         | -          |

✅ **Both methods recovered true parameters excellently!**

---

## 🧮 Detailed Results

True parameters: θ = [1.0, 3.0]

Batch Gradient Descent:  
└── θ = [0.98131667, 2.99872428] → 99.8% accurate

Stochastic Gradient Descent:  
└── θ = [0.97681657, 2.99840542] → 99.7% accurate

---

## 📈 Key Observations

### 🎯 Accuracy
- Batch GD: bias error = 1.9%, slope error = 0.03%
- SGD: bias error = 2.3%, slope error = 0.05%
- Batch GD is slightly more stable (deterministic)
- SGD shows small expected variance (stochastic nature)

### ⚙️ Implementation Settings
- Learning rate (α): 0.01  
- Max iterations: 1000  
- Tolerance (ε): 1e-6  
- Dataset: n = 100, d = 1, noise = σ = 0.1  

### 🔬 Data Statistics
- X: mean = -0.104, std = 0.904  
- y: mean = 0.691, std = 2.700  
- Correlation: strong linear relationship  

---

## 💻 Code Verification

```python
# Final loss computation
loss_batch = LossFunction.squared_error(X @ theta_batch[1:] + theta_batch, y)
loss_sgd = LossFunction.squared_error(X @ theta_sgd[1:] + theta_sgd, y)

print(f"Batch GD Loss: {loss_batch:.6f}")
print(f"SGD Loss: {loss_sgd:.6f}")
