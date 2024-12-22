# synthetic-data-generation-w-faker

This project is an interactive application built using **Streamlit** to generate synthetic data, perform exploratory data analysis (EDA), train machine learning models, and evaluate the performance of these models. The program allows users to dynamically interact with data generation and modeling processes via an easy-to-use web interface.

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
- Implements a Random Forest Regression model.
- Provides users the ability to:
  - Split data into training and testing sets.
  - View model evaluation metrics like Mean Squared Error (MSE) and R-squared values.
  - Visualize prediction results with scatter plots.

### 4. Streamlit Integration
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

3. **Modeling**
   - Train a Random Forest Regression model.
   - Test the model on synthetic data.
   - Visualize model predictions.

4. **Evaluation**
   - Display metrics such as MSE and R-squared.
   - Provide insights into the performance of the regression model.

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
