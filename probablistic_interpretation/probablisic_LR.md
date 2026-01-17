# 📊 Probabilistic Linear Regression (Least Squares ↔ MLE)

A clean, hands-on implementation of **Linear Regression** showing how  
**least squares optimization** and **maximum likelihood estimation (MLE)**  
are actually the *same thing* under Gaussian noise.

Built with **realistic noise** so it doesn’t look like a toy demo.

---

## 👀 What’s this about?

Linear regression is usually taught in two ways:

- minimize squared error (optimization view)
- maximize likelihood assuming Gaussian noise (probabilistic view)

This repo shows:
> **they land on the exact same solution**

and visualizes *why* that happens.

---

## 🧪 Data Setup

Synthetic data generated from:

\[
y = \theta_0 + \theta_1 x + \varepsilon,
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\]

**Ground truth:**
- θ₀ (bias): `1.0`
- θ₁ (slope): `3.0`
- Noise variance σ²: `1.0` (intentionally high)
- Samples: `100`

Higher noise = more realistic scatter + visible uncertainty.

---

## 📐 Least Squares (Normal Equation)

We solve linear regression in closed form:

\[
\hat{\theta} = (X^\top X)^{-1} X^\top y
\]

This:
- minimizes squared error
- recovers parameters close to ground truth
- also turns out to be the MLE

---

## 📉 Loss & Gradients

The code explicitly defines:
- squared error loss  
- analytical gradient  

So it’s easy to extend this to:
- Gradient Descent
- SGD
- momentum / Adam later

---

## 🎯 Probabilistic View (MLE)

Assuming Gaussian noise:

\[
p(y \mid x; \theta)
= \mathcal{N}(y; \theta^\top x, \sigma^2)
\]

Log-likelihood:

\[
\ell(\theta)
= -\frac{n}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\sum (y - X\theta)^2
\]

Maximizing this gives the **same θ** as least squares.

---

## 🔍 Estimating Noise (σ²)

Instead of cheating with the true noise, we estimate it:

\[
\hat{\sigma}^2 = \frac{1}{n} \sum (y - X\hat{\theta})^2
\]

This matches how things work in real datasets.

---

## 🌄 Cost vs Likelihood (Visual Proof)

The repo visualizes:
- squared error surface \( J(\theta) \)
- negative log-likelihood surface \( -\ell(\theta) \)

Even with high noise:

\[
\arg\min J(\theta)
=
\arg\max \ell(\theta)
\]

Different math, same answer.

---

## 📏 Parameter Uncertainty

We compute **95% confidence intervals**:

\[
\text{Var}(\hat{\theta}) = \hat{\sigma}^2 (X^\top X)^{-1}
\]

Higher noise ⇒ wider intervals ⇒ more honest uncertainty.

---

## 📈 Predictive Uncertainty

The model also outputs **predictive intervals**, not just a single line.

This shows:
- where predictions are confident
- where the model is guessing more

Much closer to how regression is used in practice.

---

## ✅ Takeaways

- Least squares = MLE under Gaussian noise
- Noise doesn’t break theory, it exposes uncertainty
- Confidence intervals matter
- Predictive uncertainty matters more
- Linear regression is deeper than it looks

---

## 🚀 Things you can extend next

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
- intuition > formulas
- interview prep
- building blocks for probabilistic ML

---

*Linear Regression, but actually explained.*
