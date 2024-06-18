# Automated_Hyperparameter_Optimization_System

## Project Overview

### Aim
The aim of this project is to create an automated Hyperparameter Optimization (HPO) system using AutoML techniques. This system is designed to efficiently find the best hyperparameter settings for machine learning models, customized for specific datasets and tasks. By leveraging algorithms like Bayesian Optimization and Random Search, the system explores complex hyperparameter spaces to enhance model performance metrics such as accuracy and precision. The iterative experimentation and adaptive learning approach ensure effective and scalable optimization, tackling the difficulties of manual hyperparameter tuning and enabling practitioners to achieve higher model performance with less effort and time.

### Techniques Used
#### Bayesian Optimization
Bayesian Optimization is utilized for hyperparameter tuning. This sequential model-based optimization technique employs probabilistic models to forecast the performance of different hyperparameter settings. It operates by building a surrogate model of the objective function (such as model accuracy) that approximates actual performance based on observed hyperparameter values. Bayesian inference is then used to iteratively update this surrogate model as more evaluations are conducted.

**Advantages:**
- **Efficiency:** Requires fewer evaluations of the objective function than exhaustive grid search or random search.
- **Global Optima:** Aims to find the global optimum rather than getting trapped in local optima.
- **Adaptability:** Dynamically adjusts its search strategy based on previous evaluations.
- **Noise Handling:** Manages noisy evaluations by modeling uncertainty.
- **Flexibility:** Handles various types of hyperparameters (continuous, discrete, categorical) and can manage constraints or dependencies between them.

## Explanation of the Code

### Data Processing
The `load_and_preprocess_data` function streamlines data loading, preprocessing, and splitting to ensure datasets are ready for model training and evaluation.

1. **Loading Data:** Reads a dataset from a CSV file.
2. **Handling Target Column:** Defaults to using the last column as the target variable if not provided.
3. **Feature Extraction:** Separates the dataset into features (X) and the target (y).
4. **Identifying Feature Types:** Identifies numeric and categorical features.
5. **Defining Transformers:** Sets up preprocessing pipelines for numeric (imputes missing values and scales) and categorical (imputes and encodes) features.
6. **Applying Transformations:** Uses `ColumnTransformer` to apply the defined transformers.
7. **Train-Test Split:** Splits the preprocessed data into training and testing sets.
8. **Output:** Returns `X_train`, `X_test`, `y_train`, and `y_test`.

### Models and Hyperparameter Space Definition
The `define_model_and_hyperparameters` function sets up a specified machine learning model and its corresponding hyperparameter search space.

1. **Model Selection:** Takes an argument `model_type` which specifies the type of model to be created.
2. **Model Initialization:** Initializes the appropriate model from scikit-learn based on `model_type`.
3. **Hyperparameter Definition:** Defines a dictionary of hyperparameters and their respective ranges or values to explore.
4. **Return Values:** Returns a tuple containing the initialized model and the hyperparameter dictionary.

### Bayesian Optimization Implementation
An `objective_function` is defined to optimize hyperparameters for various machine learning models, aiming to maximize model accuracy by minimizing the negative cross-validation score.

1. **Imports:** Includes necessary libraries such as `numpy`, `scipy`, `GaussianProcessRegressor`, `RandomForestClassifier`, and `cross_val_score`.
2. **Objective Function:** Takes a dictionary `params` containing hyperparameters for a specific model and initializes the model based on `model_type`.
3. **Model Selection:** Initializes models such as `RandomForestClassifier`, `LogisticRegression`, etc., based on `model_type`.
4. **Hyperparameters:** Extracts hyperparameters from the `params` dictionary and initializes the model.
5. **Cross-Validation:** Performs k-fold cross-validation on the initialized model using training data.
6. **Score Calculation:** Computes the mean accuracy score and returns the negative of this mean score.

### Bayesian Optimization Function
Performs Bayesian optimization to find the best hyperparameters for a given machine learning model.

1. **Encoding and Decoding Functions:** Converts hyperparameter values to indices or values in a numerical array and back.
2. **Sampling Hyperparameters:** Randomly samples hyperparameter values from the defined search space.
3. **Bayesian Optimization Function:** Initializes sample points, fits a Gaussian Process model, and iteratively updates the model with new samples.

### Key Components
1. **Initialization of Sample Points:** Generates initial samples and evaluates the objective function.
2. **Gaussian Process Regression:** Provides predictions and uncertainty estimates.
3. **Expected Improvement (EI):** Decides where to sample next based on potential improvement and uncertainty.
4. **Bayesian Optimization Loop:** Iteratively fits the Gaussian Process model, selects the next point based on EI, and updates the data.
5. **Finding the Best Parameters:** Identifies and returns the best hyperparameters and their corresponding objective value.

## Comparative Analysis (Random Search vs Bayesian Optimization vs Hyperopt Scores)

### Results:
- **Random Search**
- **Hyperopt**
- **Bayesian Optimization**

### Visual Comparisons:
- Images comparing the scores and performance of different optimization techniques are provided in the report.

## How to Use This Project
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/automated-hyperparameter-optimization.git
    cd automated-hyperparameter-optimization
    ```

2. **Install Dependencies:**

3. **Run the Code:**
    ```bash
    AUTOML.ipynb
    ```

4. **Modify Parameters:**
    - Update the file to change dataset paths, model types, and hyperparameter search spaces.

## Contributing
1. Fork the repository.
2. Create a new branch.
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes.
4. Commit and push your changes.
    ```bash
    git commit -m "Description of changes"
    git push origin feature-branch
    ```
5. Create a pull request.

## Acknowledgments
- [Scikit-learn](https://scikit-learn.org/)
- [Bayesian Optimization](https://arxiv.org/abs/1807.01770)
- [HyperOpt](https://github.com/hyperopt/hyperopt)
  
## Contact
For any questions or feedback, please contact Vishal Bokhare at [vishurdx309@gmail.com].
