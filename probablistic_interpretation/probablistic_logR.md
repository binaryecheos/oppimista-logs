# Probabilistic Logistic Regression – Run Log

This file wraps up what was done in  
`generative models/probabalistic_interpretation/probablistic_logistic_regression.ipynb`.[file:68]

---

## Dataset & setup

- Toy binary classification in 2D.[file:68]  
  - Features: \(x \in \mathbb{R}^2\); bias term is added in code.  
  - Ground-truth parameters: \(\theta^\* = [-0.5,\ 2.0,\ -1.0]\).  
  - Labels: sample \(p = \sigma(\theta^{*\top} x)\), then set \(y = 1\) if \(p > 0.5\) else \(0\).[file:68]

- Model: logistic regression  
  - Hypothesis: \(h_\theta(x) = \sigma(\theta^\top x)\).[file:68]  
  - Sigmoid: \(\sigma(z) = \dfrac{1}{1 + e^{-z}}\).[file:68]

- Objective (average negative log-likelihood / cross-entropy):[file:68]

  $$
  J(\theta)
  = -\frac{1}{n} \sum_{i=1}^{n}
  \Big(
    y^{(i)} \log h_\theta(x^{(i)})
    + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))
  \Big)
  $$

This is just the Bernoulli log-likelihood with a sigmoid link, multiplied by \(-1\).[file:68]

---

## Optimizers in the arena

### Gradient Descent (GD)

- Update rule:

  $$
  \theta \leftarrow \theta - \alpha \nabla_\theta J(\theta)
  $$

- Gradient (full-batch):

  $$
  \nabla_\theta J(\theta)
  = \frac{1}{n} X^\top \big(h_\theta(X) - y\big)
  $$

  where \(h_\theta(X)\) is the vector of predicted probabilities on all samples.[file:68]

- Config used: learning rate \(\alpha = 0.1\), about \(100\) iterations.[file:68]

### Newton’s Method

- Second-order method using the Hessian.[file:68]

- Update rule:

  $$
  \theta \leftarrow \theta - H^{-1} \nabla_\theta J(\theta)
  $$

- For logistic regression:[file:68]

  $$
  \nabla_\theta J(\theta)
  = \frac{1}{n} X^\top (p - y),
  \quad p = h_\theta(X)
  $$

  $$
  H
  = \frac{1}{n} X^\top R X,
  \quad
  R = \mathrm{diag}\big(p_i (1-p_i)\big)
  $$

- Converges in roughly \(5\)–\(10\) iterations on this dataset.[file:68]

---

## Parameter estimates and accuracy

| Method         | \(\theta_0\) | \(\theta_1\) | \(\theta_2\) | Accuracy |
|----------------|-------------:|-------------:|-------------:|---------:|
| ground truth   |    -0.50     |     2.00     |    -1.00     |   1.00   |
| GD (final)     |    ≈ -0.98   |     ≈ 4.64   |    ≈ -2.62   |   0.99   |
| Newton (final) |    ≈ -17.59  |     ≈ 72.68  |    ≈ -34.44  |   1.00   |

[file:68]

Observations:

- All three parameter vectors give almost perfect classification on the synthetic data.[file:68]  
- GD and Newton point in almost the same **direction** in parameter space but with very different norms. Scaling \(\theta\) mostly makes the sigmoid steeper, leaving the decision boundary almost unchanged.[file:68]

---

## Visual insights (from the notebook)

- **Decision boundaries**  
  GD and Newton boundaries plotted on the 2D scatter almost overlap, confirming that despite very different \(\theta\), the classifiers behave the same on the data.[file:68]

- **Loss vs iteration**  

  - GD: gradual, smooth decrease in \(J(\theta)\) over many small steps.  
  - Newton: sharp drop in a handful of iterations, thanks to curvature information in the Hessian.[file:68]

- **Parameter-space trajectories**  

  In the \((\theta_1,\theta_2)\) plane:  
  - GD follows a curved path sliding along the valley of the loss surface.  
  - Newton takes a few large jumps cutting across contours towards the minimum.[file:68]

- **Loss surface contour + paths**  

  Contour plots of \(J(\theta)\) with GD and Newton paths overlaid give a visual, geometric comparison of first-order vs second-order optimization.[file:68]

---

## Takeaways

1. Logistic regression with cross-entropy is maximum likelihood estimation for a Bernoulli model with sigmoid link.[file:68]  

2. For almost linearly separable data, the solution is not unique in terms of scale: many \(\theta\) along the same ray yield effectively the same boundary and accuracy; what matters is direction, not norm.[file:68]

3. GD vs Newton trade-off:[file:68]

   - **Gradient Descent**  
     - Cheap iterations, only needs gradients.  
     - More steps to converge, but scales well to high-dimensional problems.

   - **Newton’s Method**  
     - Very few iterations once near the optimum.  
     - Each step is expensive: computing and inverting \(H\) is roughly \(O(d^3)\), so it is practical mainly for low-dimensional, “theoretical demo” settings rather than large-scale models.

---

Suggested file path: `results/probablistic_logistic_regression.md`
