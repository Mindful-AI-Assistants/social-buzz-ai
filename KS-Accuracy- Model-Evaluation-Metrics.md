

# Complete Analysis of Model Evaluation Metrics in AI: KS, Accuracy and More

## 1. Purpose of Model Evaluation

Evaluating a machine learning (ML) model means measuring how well it performs its predictive task on new (unseen) data. This process is essential to ensure the model generalizes well and provides reliable predictions.

There are two main types of supervised ML problems:

- **Classification models:** Predict categories (e.g., spam vs. not spam, positive vs. negative).  
- **Regression models:** Predict numerical continuous values (e.g., price, temperature).

Each type requires specific metrics for performance assessment.

## 2. Evaluation Metrics for Classification Models

### Confusion Matrix

A fundamental tool showing counts of predicted vs. actual classes:

|                       | Predicted Positive | Predicted Negative |
|-----------------------|--------------------|--------------------|
| **True Positive (TP)** | Correct positive    | False negative     |
| **True Negative (TN)** | False positive      | Correct negative    |

From this matrix, we derive many metrics:

### Accuracy

- **Definition:** The ratio of correct predictions (both positive and negative) to total predictions.  
- **Formula:**  
  $$
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  $$
- **Limitation:** May be misleading on imbalanced datasets where one class dominates.

### Precision

- **Definition:** The ratio of correctly predicted positives to all predicted positives.  
- **Formula:**  
  $$
  Precision = \frac{TP}{TP + FP}
  $$
- **Use case:** Important when false positives are costly (e.g., medical diagnoses to avoid false alarms).

### Recall (Sensitivity)

- **Definition:** The ratio of correctly predicted positives to all actual positives.  
- **Formula:**  
  $$
  Recall = \frac{TP}{TP + FN}
  $$
- **Use case:** Vital when it is important to detect all positives (e.g., fraud detection).

### F1 Score

- **Definition:** Harmonic mean of Precision and Recall, balancing both.  
- **Formula:**  
  $$
  F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  $$
- **Useful when:** Classes are imbalanced and both precision and recall matter.

### AUC-ROC (Area Under ROC Curve)

- Measures the model’s ability to distinguish classes over all classification thresholds.  
- Values range from 0 to 1; closer to 1 indicates better discrimination.

## 3. Evaluation Metrics for Regression Models

These measure error between predicted and actual numerical values.

### Mean Absolute Error (MAE)

- Average of absolute differences between predicted and true values.  
- Easy to interpret as it uses the same units as the predicted variable.

### Mean Squared Error (MSE)

- Average of squared differences, penalizing larger errors.  

### Root Mean Squared Error (RMSE)

- Square root of MSE, with same units as predicted variable, useful for interpretation.

### Coefficient of Determination (R²)

- Proportion of variance in the dependent variable explained by the model.

## 4. What is KS (Kolmogorov-Smirnov) Metric in AI Models?

- KS is a metric commonly used in **binary classification** to measure the **discriminating power** of a model.  
- It calculates the maximum distance between the cumulative distribution functions (CDFs) of the model’s predicted probabilities for positive and negative classes.  
- Values range from 0 to 1 (or 0% to 100%), with higher values indicating better separation between the classes.  
- Widely used in sectors like credit scoring and fraud detection where clear separation is crucial.

## 5. Difference Between KS and Accuracy

| Metric              | What it Measures                                            | Key Characteristic                                      | Typical Use                    |
|---------------------|------------------------------------------------------------|--------------------------------------------------------|--------------------------------|
| **KS (Kolmogorov-Smirnov)** | Maximum difference between cumulative distributions of predicted probabilities for positive and negative classes | Evaluates model's overall discrimination power across *all* thresholds, independent of a fixed cutoff | Performance evaluation especially with imbalanced datasets and ranking focus |
| **Accuracy**        | Percentage of correct predictions at a specific threshold (usually 0.5) | Easy to understand but can be misleading with imbalanced data or arbitrary cutoffs | General baseline metric; limited with imbalance |

**In summary:**  
- KS focuses on how well the model scores separate positive and negative classes across all possible thresholds rather than one fixed point.  
- Accuracy simply measures the correctness of predictions using one decision threshold, which can overlook the model’s ability to rank or score examples well.

## 6. Practical Recommendations for Use

- Select metrics aligned with your business goal and problem type.  
- For imbalanced datasets, prefer metrics like Precision, Recall, F1, AUC, and KS rather than Accuracy alone.  
- Use multiple complementary metrics for a complete understanding of model performance.  
- Visual tools such as ROC and KS curves can aid interpretation and decision-making.

## 7. Summary Table of Common Metrics

| Metric        | When to Use                                          | Interpretation                                |
|---------------|-----------------------------------------------------|-----------------------------------------------|
| Accuracy      | Balanced classes, simple performance measure        | Overall correct prediction rate                |
| Precision     | High cost of false positives                         | Correctness of positive predictions            |
| Recall        | Need to identify all positives                       | Ability to capture actual positives            |
| F1 Score      | Imbalanced classes; balanced precision/recall need | Balance of precision and recall                 |
| AUC-ROC       | Measure classifier discrimination across thresholds | Area under TP rate vs FP rate curve             |
| KS            | Evaluate separation between class score distributions| Maximum difference between distributions        |
| MAE, MSE, RMSE| Regression prediction errors                          | Magnitude of prediction errors in numeric terms |
| R²            | Regression model fit                                  | Proportion of variance explained by the model   |


<br>


Here is a comprehensive response covering your request: example calculations, Python code snippets for KS and Accuracy, plus visualizations of ROC and KS curves, and detailed explanations of Precision, Recall, F1, and AUC.

# 1. Example Calculations for KS and Accuracy Metrics

### Accuracy Calculation Example

Suppose you have the following confusion matrix from your binary classifier on 100 observations:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| Actual Positive| 40 (True Positives) | 10 (False Negatives)|
| Actual Negative| 5  (False Positives)| 45 (True Negatives) |

Accuracy is:

$$
Accuracy = \frac{TP + TN}{Total} = \frac{40 + 45}{100} = 0.85 \quad (85\%)
$$

This means 85% of the predictions were correct.

### KS Statistic Calculation Example

The KS statistic measures the maximum difference between the cumulative distribution functions (CDF) of the scores (or probabilities) assigned by the model to positive and negative classes.

Suppose after sorting model prediction scores from lowest to highest, you have cumulative distributions:

| Score Threshold | CDF Positive | CDF Negative | Absolute Difference |
|-----------------|--------------|--------------|---------------------|
| 0.2             | 0.1          | 0.05         | 0.05                |
| 0.4             | 0.3          | 0.1          | 0.20                |
| 0.6             | 0.5          | 0.3          | 0.20                |
| 0.8             | 0.8          | 0.4          | 0.40                |
| 1.0             | 1.0          | 0.7          | 0.30                |

The **KS statistic = max absolute difference = 0.40**

This indicates the point at which the model best separates positive and negative classes.

# 2. Python Code Snippets to Compute KS and Accuracy

```python
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Sample true labels and predicted probabilities
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
y_scores = np.array([0.9, 0.1, 0.8, 0.75, 0.3, 0.4, 0.85, 0.2, 0.6, 0.9])

# Binarize predictions using 0.5 cutoff
y_pred = (y_scores >= 0.5).astype(int)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.2f}")

# KS statistic using scipy's two-sample KS test
# Separate scores for positive and negative classes
scores_pos = y_scores[y_true == 1]
scores_neg = y_scores[y_true == 0]

ks_statistic, ks_pvalue = ks_2samp(scores_pos, scores_neg)
print(f"KS Statistic: {ks_statistic:.3f}, p-value: {ks_pvalue:.3f}")
```

# 3. Visualizations: ROC and KS Curves

```python
# ROC curve plot
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

plt.figure(figsize=(12,5))

# ROC Curve
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# KS Curve: plot CDFs of scores
def ecdf(data):
    """Compute empirical CDF"""
    x = np.sort(data)
    y = np.arange(1, len(x)+1)/len(x)
    return x, y

x_pos, y_pos = ecdf(scores_pos)
x_neg, y_neg = ecdf(scores_neg)

plt.subplot(1,2,2)
plt.step(x_pos, y_pos, label='Positive CDF')
plt.step(x_neg, y_neg, label='Negative CDF')
plt.title('KS Curve (Empirical CDFs)')
plt.xlabel('Score')
plt.ylabel('Cumulative Probability')
plt.legend()

plt.tight_layout()
plt.show()
```

You will see the ROC curve showing the trade-off between sensitivity and specificity and the KS curve which plots the empirical CDFs of positive vs negative predicted scores. The KS statistic corresponds to the maximum vertical gap between these two CDF curves.

# 4. Explanation of Other Important Metrics: Precision, Recall, F1, AUC

| Metric     | Definition                                                                                  | Formula                                                             | Use case                                                                |
|------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------|
| **Precision** | Proportion of predicted positives that are true positives                                   | $$\frac{TP}{TP + FP}$$                                              | When false positives are costly (e.g., spam detection)                  |
| **Recall** (Sensitivity) | Proportion of actual positives that are correctly identified                        | $$\frac{TP}{TP + FN}$$                                              | When missing positives is costly (e.g., disease detection)             |
| **F1 Score** | Harmonic mean of Precision and Recall                                                      | $$2 \times \frac{Precision \times Recall}{Precision + Recall}$$    | Balances precision and recall; useful in imbalanced datasets           |
| **AUC** (Area Under ROC Curve) | Measures overall ability of model to rank positives higher than negatives              | Computed as area under ROC curve (no simple formula)                | Evaluates model discrimination independent of classification threshold |


<br>

Below are Python code snippets showing how to calculate **Precision**, **Recall**, **F1 Score**, and **AUC** using scikit-learn, including a minimal example with true labels and predicted probabilities/scores.

<br>

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Sample true labels (binary)
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]

# Predicted labels (after applying a classification threshold, e.g., 0.5)
y_pred = [1, 0, 1, 1, 0, 0, 0, 0, 1, 1]

# Predicted scores or probabilities (usually output from model.predict_proba or model.decision_function)
y_scores = [0.9, 0.1, 0.8, 0.75, 0.3, 0.4, 0.45, 0.2, 0.6, 0.85]

# Calculate Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.2f}")

# Calculate Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.2f}")

# Calculate F1 Score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.2f}")

# Calculate AUC (Area Under the ROC Curve)
auc = roc_auc_score(y_true, y_scores)
print(f"AUC: {auc:.2f}")
```

<br>

### Explanation:

- **Precision** is the proportion of predicted positive instances that are actually positive.
- **Recall** (or Sensitivity) is the proportion of actual positive instances that are correctly predicted.
- **F1 Score** is the harmonic mean of precision and recall and balances both.
- **AUC** uses the raw predicted scores/probabilities (not thresholded) and measures the model's ability to rank positives higher than negatives.


<br>




<br>


## References and Further Reading

- [Sigmoidal.ai Metrics for Classification](https://sigmoidal.ai/metricas-de-avaliacao-em-modelos-de-classificacao-em-machine-learning/)  
- [Escola DNC Metrics and Techniques](https://www.escoladnc.com.br/blog/avaliacao-de-modelos-de-machine-learning-metricas-e-tecnicas-essenciais)  
- [LinkedIn Pulse Complete Guide to ML Metrics](https://pt.linkedin.com/pulse/m%C3%A9tricas-de-avalia%C3%A7%C3%A3o-em-machine-learning-um-guia-completo-mendes-fonseca-hoxwf)  

