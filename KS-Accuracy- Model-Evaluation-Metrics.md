

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


## References and Further Reading

- [Sigmoidal.ai Metrics for Classification](https://sigmoidal.ai/metricas-de-avaliacao-em-modelos-de-classificacao-em-machine-learning/)  
- [Escola DNC Metrics and Techniques](https://www.escoladnc.com.br/blog/avaliacao-de-modelos-de-machine-learning-metricas-e-tecnicas-essenciais)  
- [LinkedIn Pulse Complete Guide to ML Metrics](https://pt.linkedin.com/pulse/m%C3%A9tricas-de-avalia%C3%A7%C3%A3o-em-machine-learning-um-guia-completo-mendes-fonseca-hoxwf)  

