# Comparison of Classic Algorithms – Results

This README summarizes the experiments and insights from  
`interpretations/comparison_of_classic_algos.ipynb`.[file:89]

---

## 1. Datasets and experimental setup

You evaluate multiple models on:

- 2D synthetic datasets (e.g. moons, Gaussian blobs) to visualize decision boundaries.[file:89]  
- Tabular continuous data (e.g. Iris) for multiclass classification.[file:89]  
- Bag‑of‑words style count data for text‑like settings.[file:89]

For each dataset you:

- Split into train and test sets.  
- Train several models with reasonable hyperparameters.  
- Compute accuracy, confusion matrices, and for 2D, decision‑boundary plots.[file:80][file:89]

---

## 2. Models and math 

### 2.1 Generalized Linear Models (GLMs)

All GLMs share a common form:

- Linear predictor:
  $$
  \eta = X\theta
  $$
- Mean response with inverse link:
  $$
  \mu = g^{-1}(\eta)
  $$

The loss is the negative log‑likelihood of an exponential‑family model.[file:64][file:73]

#### Gaussian GLM (ordinary least squares)

Assumption:
$$
y \mid x \sim \mathcal{N}(\mu, \sigma^2),
\quad
\mu = \theta^\top x
$$

Squared‑error loss:
$$
J(\theta)
= \frac{1}{2n}
\sum_{i=1}^n
\big(\theta^\top x^{(i)} - y^{(i)}\big)^2
$$
[file:64]

#### Bernoulli–logit GLM (binary logistic regression)

Assumption:
$$
y \mid x \sim \text{Bernoulli}(p),
\quad
p = \sigma(\theta^\top x)
$$

Binary cross‑entropy loss:
$$
J(\theta)
=
-\frac{1}{n}
\sum_{i=1}^n
\Big[
y^{(i)}\log p^{(i)}
+
(1 - y^{(i)})\log(1 - p^{(i)})
\Big]
$$
[file:68]

#### Softmax GLM (multiclass logistic regression)

Class probabilities:
$$
P(y = k \mid x)
=
\frac{\exp(\theta_k^\top x)}
{\sum_{j=1}^K \exp(\theta_j^\top x)}
$$

Optimized with standard multiclass cross‑entropy.[file:64][file:71]

GLMs in the notebook are trained with gradient descent/SGD, and Newton’s method for logistic is explored elsewhere.[file:64][file:68]

---

### 2.2 Generative models: Naive Bayes and GDA

#### Gaussian Naive Bayes (GNB)

Per‑feature Gaussian assumption:
$$
x_j \mid y = k
\sim
\mathcal{N}(\mu_{k,j}, \sigma_{k,j}^2)
$$

Conditional likelihood:
$$
p(x \mid y = k)
=
\prod_{j=1}^d
p(x_j \mid y = k)
$$

Posterior:
$$
P(y = k \mid x)
\propto
P(y = k)\,
\prod_{j=1}^d p(x_j \mid y = k)
$$
[file:89]

#### Multinomial Naive Bayes (MNB)

For count vectors \(x \in \mathbb{N}^d\):

Class‑conditional distribution:
$$
p(x \mid y = k)
\propto
\prod_{j=1}^d
\phi_{k,j}^{\,x_j}
$$

With Laplace smoothing:
$$
\phi_{k,j}
=
\frac{\text{count}_{k,j} + \alpha}
{\sum_{j'=1}^d \text{count}_{k,j'} + \alpha d}
$$
[file:89]

#### Gaussian Discriminant Analysis (GDA, binary)

Generative assumptions:
$$
x \mid y = k
\sim
\mathcal{N}(\mu_k, \Sigma),
\quad
k \in \{0,1\}
$$
$$
\phi = P(y = 1)
$$

The posterior has a logistic form:
$$
P(y = 1 \mid x)
=
\sigma(\theta^\top \tilde{x}),
\quad
\tilde{x}
=
\begin{bmatrix}
1 \\
x
\end{bmatrix}
$$

with \(\theta\) determined by \(\mu_0,\mu_1,\Sigma,\phi\).[file:73]

Multiclass GDA extends this to \(K\) classes with means \(\mu_k\) and shared covariance \(\Sigma\).[file:73][file:89]

---

### 2.3 Tree ensembles and neural networks

- **RandomForestClassifier**: bagging of decision trees with random feature subsampling.[file:89]  
- **XGBoost (XGBClassifier)**: gradient‑boosted trees minimizing log‑loss; many shallow trees added sequentially.[web:75][web:84][web:87][file:89]  
- **MLPClassifier**: fully connected neural network with ReLU hidden layers and Adam optimization.[file:89]

These are treated as strong non‑linear baselines on synthetic and tabular data.

---

## 3. Empirical behavior and conclusions

### 3.1 Non‑linear 2D data (moons, blobs)

On strongly non‑linear 2D datasets:

- Logistic regression (a linear boundary) underfits, giving almost straight decision lines and lower accuracy near curved boundaries.[file:80][file:89]  
- Gaussian NB does well when clusters are close to axis‑aligned Gaussians, but its independence assumption yields boxy regions and errors when features are correlated.[file:89]  
- GDA (full shared covariance) better tracks elliptical clusters and produces near‑optimal linear boundaries.[file:73][file:89]  
- RandomForest and XGBoost fit highly non‑linear, piecewise‑constant regions that wrap closely around the moons, achieving top classical performance.[file:80][file:84][file:89]  
- MLP learns smooth curved boundaries comparable to XGBoost, often matching or beating tree ensembles when tuned.[file:89]

**Key point:** Non‑linear models (trees, boosting, MLP) clearly outperform linear GLMs and simple generative models on these tasks.[file:80][file:89]

---

### 3.2 Tabular continuous data (Iris)

On Iris‑like data:

- Softmax GLM, multiclass GDA, RandomForest, and XGBoost all achieve high accuracy, with small differences in confusion matrices around the hardest classes.[file:89]  
- Gaussian NB is decent but can misclassify overlapping species; per‑class accuracy reveals which classes are most affected.[file:89]  
- MLP is competitive but not dramatically better; with small tabular datasets, trees and GLMs are often equally strong or stronger given less tuning.[file:75][file:76][file:89]

**Key point:** For “nice” tabular data, GLMs, GDA, and tree ensembles are all reliable; GLMs/GDA win on simplicity and interpretability, tree ensembles on robustness and slight performance gains.[file:75][file:76][file:89]

---

### 3.3 Bag‑of‑words / counts

On text‑like count data:

- Multinomial NB is extremely strong when data are generated from distinct multinomial word distributions, often matching or beating much more flexible discriminative models.[file:89]  
- Softmax GLM and XGBoost also perform well, but NB attains similar accuracy with far simpler training and a clear generative story.[file:89]

**Key point:** When the multinomial assumption matches the data (classic BoW), Multinomial NB is a very strong baseline.

---

### 3.4 Confusion matrices and per‑class patterns

Across datasets:

- Linear GLMs show systematic confusions near curved boundaries or for overlapping classes.  
- GDA and NB can be very balanced when assumptions hold, but exhibit characteristic failure modes when independence or Gaussianity is violated.[file:73][file:89]  
- Tree ensembles and XGBoost typically reduce these systematic errors and yield more uniform per‑class accuracy.[file:84][file:89]  
- MLP behaves similarly to XGBoost, with possible mild overfitting if regularization is weak.[file:89]

---

## 4. High‑level takeaways

1. There is no universal “best” classic algorithm; performance depends on how well model assumptions match the data and on sample size.[file:89]  
2. GLMs are simple, interpretable, and strong when relationships are (generalized) linear or when good features are available.[file:64][file:68][file:73]  
3. Generative models (NB, GDA) are extremely data‑efficient and competitive when their exponential‑family assumptions hold.[file:69][file:73][file:89]  
4. Tree ensembles (RandomForest, XGBoost) are robust, powerful defaults for tabular data and handle non‑linearities and interactions with minimal preprocessing.[web:84][web:87][file:89]  
5. MLPs are highly flexible but need more data and tuning; on small tabular datasets they are often matched or beaten by well‑tuned trees or GLMs.[file:75][file:76][file:89]

Suggested path for this file: `results/comparison_of_classic_algos.md`
