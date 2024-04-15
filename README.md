# Absenteeism Prediction with Logistic Regression

## Introduction

This project focuses on predicting absenteeism in the workplace using logistic regression. Absenteeism can have significant impacts on productivity and organizational performance. By leveraging machine learning techniques, we aim to develop a model that can accurately predict absenteeism and facilitate proactive interventions.

## Libraries Used

- Pandas version: 2.0.3
- NumPy version: 1.24.3
- Scikit-learn version: 1.3.0
- Pickle version: 4.0
- Matplotlib version: 3.7.2
- Seaborn version: 0.12.2

## Project Structure

### Source Files

- `PreprocessData.ipynb`: Jupyter Notebook used for processing raw data.
- `Train_model.ipynb`: Jupyter Notebook used for training the data with the logistic regression algorithm.
- `Absenteeism_module.py`: Python module used to deploy the model and scaler.
- `Test_Model_With_NewData.ipynb`: Jupyter Notebook used to test the pre-trained model with new data.

### Additional Files

- `model` and `scaler` directories: Contain the trained model and scaler for further development.
- `Project_Proposal.pdf`: Detailed proposal outlining the objectives and methodology of the project.
- `ExplainVisualization.pdf`: Presentation explaining the visualization of results using Tableau.

### Dataset Details

- Total Data: 700 samples
- Train Data: Approximately 80% of the total data (560 samples)
- Test Data: Remaining 20% of the total data (140 samples)
- New Test Data: 40 samples
- `FeatureDescription.png`: Image file describing the feature inputs for the model.
- `Absenteeism_data.csv`: CSV file containing raw data for training.
- `df_preprocessed_with_targets.csv`: CSV file containing data after preprocessing.
- `Absenteeism_Predictions.csv`: CSV file containing the predictions from the model, used for visualization.
- `Absenteeism_new_data.csv`: CSV file containing new data for predicting with the trained model to generate `Absenteeism_Predictions.csv`.

## Installation and Usage

To run the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries mentioned above using `pip`.
3. Open and run the Jupyter Notebooks (`PreprocessData.ipynb`, `Train_model.ipynb`, `Test_Model_With_NewData.ipynb`) in the specified order.
4. Utilize the `Absenteeism_module.py` for deploying the model and scaler.
5. Refer to the provided documentation (`Project_Proposal.pdf`, `ExplainVisualization.pdf`) for additional information.

By following this approach, we aim to train a logistic regression model that can accurately label excessive absenteeism based on employee attributes.
