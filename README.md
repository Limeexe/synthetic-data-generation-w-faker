# synthetic-data-generation-w-faker

This project is a simple interactive application built using **Streamlit** to generate synthetic data, perform exploratory data analysis (EDA), train machine learning models, and evaluate the performance of these models. The program allows users to dynamically interact with data generation and modeling processes via an easy-to-use web interface as per subject requirement.

---

## Features

### 1. Synthetic Data Generation
- Utilizes the **Faker** library to create realistic synthetic datasets.
- Supports numeric, categorical, and temporal features.
- Users can specify the number of samples and feature configurations dynamically.

### 2. Exploratory Data Analysis (EDA)
- Visualizes relationships between features using:
  - Pairplots
  - Correlation heatmaps
- Summarizes key statistics of the dataset interactively.

### 3. Machine Learning Modeling
- Implements Random Forest Regression, Linear Regression, and Polynomial Regression models.
- Allows users to:
  - Split data into training and testing sets.
  - Perform cross-validation to evaluate model generalizability.
  - View model evaluation metrics like Mean Squared Error (MSE) and R-squared values.
  - Visualize prediction results with scatter plots.

### 4. Cross-Validation
- Uses **KFold Cross-Validation** to assess the robustness of the selected model.
- Splits the training data into 5 folds and evaluates the model's R-squared score for each fold.
- Displays:
  - R-squared scores for each fold.
  - Mean R-squared score to summarize overall performance.

### 5. Streamlit Integration
- Interactive interface for:
  - Adjusting data generation parameters.
  - Viewing and analyzing visualizations.
  - Evaluating model performance dynamically.

---

## Prerequisites

### Install the Required Libraries
Ensure you have the following Python libraries installed:

```bash
pip install streamlit pandas numpy faker scikit-learn matplotlib seaborn
```

---

## How to Run the Application

1. Clone or download this repository.
2. Navigate to the project directory.
3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open your web browser and go to the provided URL (e.g., `http://localhost:8501`).

---

## Program Workflow

1. **Data Generation**
   - Input desired number of samples and feature configurations.
   - Generate datasets containing numeric, categorical, and temporal features.
   - Save the dataset as a CSV file.

2. **Exploratory Data Analysis (EDA)**
   - Display pairplots to observe feature relationships.
   - Show correlation heatmaps to understand dependencies.
   - Summarize dataset statistics interactively.

3. **Train Machine Learning Models**
   - Select a machine learning model from:
     - Random Forest Regressor
     - Linear Regression
     - Polynomial Regression
   - Train the model on the training set and evaluate its performance.

4. **Cross-Validation**
   - Use KFold Cross-Validation (5 folds) to evaluate the selected model.
   - View R-squared scores for each fold and the mean R-squared score.

5. **Evaluation**
   - Display metrics such as MSE and R-squared on the test set.
   - Visualize predictions against true values using scatter plots.

6. **Save Results**
   - Save the synthetic dataset and model evaluation results for further analysis.

---

## File Structure

```
project-directory/
|-- app.py                  # Main Streamlit application script
|-- requirements.txt        # List of required libraries
|-- README.md               # Program documentation (this file)
|-- sample_output/          # Folder for saving visualizations and outputs
```

---

## Sample Outputs

### Pairplot
Illustrates the relationships between features and the target variable.

### Correlation Heatmap
Displays correlations between features and highlights dependencies.

### Cross-Validation Results
Shows R-squared scores for each fold and the mean R-squared score to evaluate model performance.

### Model Predictions
Scatter plot comparing true vs. predicted values of the target variable.

---

## Future Improvements

- Add more machine learning models (e.g., Linear Regression, Gradient Boosting).
- Enable hyperparameter tuning via the Streamlit interface.
- Enhance data generation by incorporating advanced synthetic data tools like GANs.
- Expand visualization options.

---

## Author
This program was developed as part of an educational project. Feel free to use, modify, and enhance it!

---
