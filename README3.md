# Task 4 - Logistic Regression Binary Classification

## Objective
Build a binary classifier using Logistic Regression on the Breast Cancer Wisconsin dataset, evaluate model performance, and explain key concepts like sigmoid function and ROC curve.

## Dataset
- Dataset file: `‪C:\Desktop\Elevate Labs\data (1).csv`
- Target column: `diagnosis`
   - M = Malignant (encoded as 1)
   - B = Benign (encoded as 0)
- Features: All columns except `id` and `Unnamed: 32`

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Steps Performed
1. Loaded dataset from CSV.
2. Encoded diagnosis column to binary values.
3. Dropped unnecessary columns (`id`, `Unnamed: 32`).
4. Split data into training and testing sets.
5. Standardized features using `StandardScaler`.
6. Trained Logistic Regression model.
7. Evaluated model using:
   - Confusion matrix
   - Precision
   - Recall
   - ROC-AUC score
8. Plotted:
   - ROC curve
   - Sigmoid function curve

## How to Run

```bash
pip install pandas numpy matplotlib scikit-learn
python logistic_regression_yourdata.py
