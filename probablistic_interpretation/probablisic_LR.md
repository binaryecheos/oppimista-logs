# 📊 Probabilistic Linear Regression (Least Squares ↔ MLE)

A clean, no-BS implementation of **Linear Regression** showing how  
**least squares optimization** and **maximum likelihood estimation (MLE)**  
are literally the *same thing* once you assume Gaussian noise.

Built with **high noise (σ² = 1.0)** so it actually feels real.

---

## 👀 What’s this repo?

Linear regression is usually taught in two parallel ways:

- minimize squared error (optimization view)
- maximize likelihood (probabilistic view)

This repo shows that:

> **they converge to the exact same parameters**

…and visualizes *why*.

---

## 🧪 Data Generation

We generate synthetic data from:

$$
y = \theta_0 + \theta_1 x + \varepsilon,
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

**Ground truth setup:**
- Bias (θ₀): `1.0`
- Slope (θ₁): `3.0`
- Noise variance (σ²): `1.0`
- Samples: `100`

Higher noise = more scatter, wider uncertainty, more realistic behavior.

---

## 📐 Least Squares (Normal Equation)

We solve linear regression in closed form by minimizing squared error:

$$
J(\theta) = \frac{1}{2n} \sum_{i=1}^n (y^{(i)} - \theta^\top x^{(i)})^2
$$

The solution is:

$$
\hat{\theta} = (X^\top X)^{-1} X^\top y
$$

This gives:
- the best linear fit (least squares)
- parameters close to ground truth
- the same solution as MLE (next section)

---

## 📉 Loss & Gradients

The repo explicitly defines:
- squared error loss
- analytical gradient

This makes it easy to extend later to:
- Gradient Descent
- SGD
- momentum / Adam

---

## 🎯 Probabilistic View (MLE)

Assuming Gaussian observation noise:

$$
p(y \mid x; \theta)
= \mathcal{N}(y; \theta^\top x, \sigma^2)
$$

The log-likelihood becomes:

$$
\ell(\theta)
= -\frac{n}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\sum (y - X\theta)^2
$$

Maximizing this gives the **same θ** as minimizing squared error.

Different perspective, same math.

---

## 🔍 Estimating Noise Variance

Instead of assuming σ² is known, we estimate it from data:

$$
\hat{\sigma}^2
= \frac{1}{n} \sum (y - X\hat{\theta})^2
$$

This matches how regression works in real datasets.

---

## 🌄 Cost vs Likelihood (Visual Proof)

We visualize:
- squared error surface \( J(\theta) \)
- negative log-likelihood surface \( -\ell(\theta) \)

Even with high noise:

$$
\arg\min_\theta J(\theta)
=
\arg\max_\theta \ell(\theta)
$$

Same optimum. Always.

---

## 📏 Parameter Uncertainty (Confidence Intervals)

We compute approximate **95% confidence intervals** using:

$$
\mathrm{Var}(\hat{\theta})
= \hat{\sigma}^2 (X^\top X)^{-1}
$$

Higher noise ⇒ wider intervals ⇒ honest uncertainty.

---

## 📈 Predictive Uncertainty

Predictions include uncertainty, not just a line:

$$
\mathrm{Var}(y_* \mid x_*)
= \hat{\sigma}^2
\left(1 + x_*^\top (X^\top X)^{-1} x_*\right)
$$

This shows where the model is confident and where it’s guessing more.

---

## ✅ Takeaways

- Least squares = MLE under Gaussian noise
- Noise doesn’t break theory, it exposes uncertainty
- Confidence intervals matter
- Predictive uncertainty matters more
- Linear regression is deeper than it looks

---

## 🚀 Possible Extensions

- Gradient Descent / SGD
- MAP estimation with priors
- Bayesian Linear Regression
- Higher-dimensional features
- Logistic regression

---

## 🛠️ Tech Stack

- Python 3
- NumPy
- Matplotlib

---

## 📝 Notes

This repo is meant for:
- ML fundamentals
- intuition-first learning
- interview prep
- probabilistic modeling foundations

---

*Linear Regression — explained like a human wrote it.*
