# 📊 Probabilistic Linear Regression (Deterministic & MLE View)

This directory demonstrates **Linear Regression** from both:
- a **deterministic optimization** perspective (least squares), and  
- a **probabilistic modeling** perspective (maximum likelihood estimation).

Using synthetic data with known parameters, we show that **minimizing squared error is equivalent to maximizing likelihood under Gaussian noise**.

---

## 📌 Overview

Linear regression can be derived in two equivalent ways:

1. **Least Squares Optimization**  
   Minimize the squared error loss between predictions and targets.

2. **Maximum Likelihood Estimation (MLE)**  
   Assume Gaussian noise in observations and maximize the log-likelihood.

This project implements both approaches and **verifies their equivalence analytically and visually**.

---

## 🧪 Data Generation

The dataset is generated from a known linear model:

\[
y = \theta_0 + \theta_1 x + \varepsilon,
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
\]

**True parameters:**
- Bias (θ₀): `1.0`
- Slope (θ₁): `3.0`
- Noise variance (σ²): `0.01`
- Number of samples: `100`

This controlled setup allows direct comparison between learned and true parameters.

---

## 📐 Deterministic View: Normal Equation

The squared error objective is defined as:

\[
J(\theta) = \frac{1}{2n} \sum_{i=1}^n (y^{(i)} - \theta^\top x^{(i)})^2
\]

The closed-form minimizer is:

\[
\hat{\theta} = (X^\top X)^{-1} X^\top y
\]

This solution:
- Minimizes the squared error
- Recovers parameters close to the ground truth
- Serves as a reference for probabilistic estimation

---

## 📉 Loss Function & Gradient

The implementation explicitly defines:
- Squared error cost function  
- Analytical gradient of the loss  

This bridges closed-form solutions with gradient-based optimization methods.

---

## 🎯 Probabilistic Model

Assuming Gaussian noise leads to the likelihood:

\[
p(y^{(i)} \mid x^{(i)}; \theta)
= \mathcal{N}(y^{(i)}; \theta^\top x^{(i)}, \sigma^2)
\]

The log-likelihood is:

\[
\ell(\theta)
= -\frac{n}{2}\log(2\pi\sigma^2)
- \frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - \theta^\top x^{(i)})^2
\]

Maximizing this expression yields the **Maximum Likelihood Estimator (MLE)**.

---

## 🔍 Likelihood Evaluation

The log-likelihood is evaluated for:
- Random parameters
- True generating parameters
- Parameters obtained from the normal equation

Results show that the likelihood is maximized near the normal-equation solution, confirming theoretical expectations.

---

## 🌄 Likelihood & Cost Landscapes

To build geometric intuition, the project visualizes:
- Log-likelihood as a function of individual parameters
- Squared error cost surface \( J(\theta) \)
- Negative log-likelihood surface \( -\ell(\theta) \)

Both surfaces reach their optimum at the same parameter values.

\[
\arg\min_\theta J(\theta)
=
\arg\max_\theta \ell(\theta)
\]

---

## ✅ Key Takeaways

- Least squares and MLE lead to the **same optimal parameters**
- The normal equation provides both the LS solution and the MLE
- Loss and likelihood surfaces offer clear geometric intuition
- Linear regression forms the foundation for more advanced models

---

## 🚀 Possible Extensions

- Estimate noise variance σ² via MLE  
- Implement Gradient Descent / SGD  
- MAP estimation with Gaussian priors  
- Higher-dimensional feature spaces  
- Logistic regression and generalized linear models  

---

## 🛠️ Requirements

- Python 3.x  
- NumPy  
- Matplotlib  

---

## 📅 Notes

This implementation is intended for:
- Learning and intuition-building  
- Academic coursework  
- Interview preparation  
- Foundations for probabilistic ML models  

---

*Linear Regression — Deterministic Optimization meets Probabilistic Modeling.*
