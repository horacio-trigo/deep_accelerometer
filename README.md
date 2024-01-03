# Deep Accelerometry

## Description
Deep accelerometry is an analysis of accelerometer data using various machine learning and deep learning techniques to predict bone health. The project aims to categorize bone health into two categories: normal and osteopenia/osteoporosis. By comparing different approaches, the project seeks to identify the best classification method for this purpose.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files Description](#files-description)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
(Instructions on how to install and set up the project.)

## Usage
(Instructions on how to use the project, including example commands.)

## Files Description
Below is a description of the main Python notebooks included in this project:

### 1. download.ipynb
This notebook is responsible for the automated downloading of the National Health and Nutrition Examination Survey (NHANES) dataset from the CDC website for the year 2013. The NHANES data, spanning five categories—Demographics, Dietary, Examination, Laboratory, Questionnaire—is collected in .xpt (SAS Transport) files. The notebook handles the parsing of the CDC website's directory structure to locate these files and uses multi-threading to expedite the downloading process. It ensures all datasets are organized into their respective folders for subsequent analysis.

### 2. bug_found_in_read_sas.ipynb
This notebook documents a critical bug encountered when using Pandas' `read_sas` function, which incorrectly reads zero values in .xpt files. It provides a reference to the related GitHub issue for this bug and offers a workaround to ensure accurate data frames. The notebook demonstrates the problem with an example from the NHANES dataset and applies a quick fix by replacing the erroneous values with true zeros, ensuring the integrity of the data for further analysis.

### 3. testing_EDA_library.ipynb
This notebook showcases the use of the `ydata-profiling` library for conducting an Exploratory Data Analysis (EDA) on the Demographics dataset from NHANES. It includes the steps to convert SAS Transport files into pandas DataFrames and the subsequent generation of a profiling report. The report provides an interactive HTML document that gives insights into the data distribution, missing values, and the relationships between variables. The process outlined in the notebook facilitates a comprehensive initial assessment of the demographic variables available in the NHANES dataset for the year 2013.

### 4. building_target_variable.ipynb
This notebook focuses on constructing a target variable for predicting bone health. It starts by importing the necessary BMD measurements from the NHANES dataset, specifically femoral neck and lumbar spine BMD values. The notebook explains the calculation of the T-scores, which are a standard measure for diagnosing osteoporosis and osteopenia. These scores are computed by comparing the patient's BMD to the average BMD of a young healthy adult.

To prepare the data for machine learning, the T-scores are categorized into 'normal', 'osteopenia', or 'osteoporosis' according to established medical thresholds. The notebook includes a function to map these categories into a binary target variable (0: 'normal', 1: 'osteopenia' or 'osteoporosis'), suitable for classification algorithms. The resulting dataframe is then saved as 'target.csv' for easy access in subsequent modeling steps. Visualizations such as histograms and pie charts are provided to understand the distribution of the bone health categories within the dataset.

This script is a key step in the data preparation phase of the project, ensuring that the target variable for the machine learning models is accurately defined and ready for use.

### 5. physical_activity_feature_engineering.ipynb
This notebook is dedicated to transforming raw accelerometer data from NHANES into meaningful features that represent physical activity levels. The minute-by-minute accelerometer readings are aggregated to create daily summaries and various features indicating the intensity and duration of physical activities. 

The notebook outlines the process to:
- Load and preprocess the raw accelerometer data, addressing a known bug in the `read_sas` function.
- Engineer features from day-by-day physical activity records, calculating weighted averages for different activity intensities.
- Address data type issues that arise from the SAS to pandas conversion, ensuring that integer values are correctly processed.
- Aggregate minute-by-minute data to extract features such as total valid wear minutes and activity intensities like sedentary, light, moderate, vigorous, and very vigorous.
- Apply weighted averages to these features to account for the varying lengths of valid wear time, giving a more accurate reflection of a person's activity level.

Feature extraction is done in a way that respects the varying amounts of valid wear time per day, using custom functions to calculate weighted averages. The final features include the average minutes per day spent in various intensity levels, total accumulated MIMs (Motion Intensity Minutes) per day for each intensity level, and the number of valid days of wear time. The result is a comprehensive dataset of physical activity features saved to 'pax.csv', ready for further analysis or machine learning.

### 6. generate_time_series_matrix.ipynb 
Time Series Matrix Generation for Accelerometer Data.
This notebook presents a method for converting raw accelerometer data into a structured three-dimensional matrix. The purpose of this transformation is to prepare the data for input into deep learning models. The notebook details the following steps:

1. Loading the NHANES 2013-2014 Physical Activity Monitor data.
2. Preprocessing the data to address bugs in the `read_sas` function and converting byte literals to integers.
3. Filtering the data to ensure that only days with a full set of readings (1440 minutes) are included, and each subject has at least 7 days of such data.
4. Generating a 3D matrix where each element corresponds to a subject, a day of the week, and minute-by-minute acceleration data.
5. Saving the matrix to a `.npy` file, which can be reloaded for future analyses.

The final matrix will be saved to 'matrix_3d.npy'. The resulting matrix (`matrix_3d.npy`) is a clean and organized dataset that can be efficiently utilized in deep learning frameworks to model and predict physical activity patterns or other health-related outcomes.

### 7. generate_full_dataframe.ipynb
NHANES Data Aggregation and Processing

This script is tailored for aggregating and processing data from the NHANES (National Health and Nutrition Examination Survey) dataset. It integrates various health metrics into a unified DataFrame which includes demographics, dietary, examination, laboratory, and questionnaire data. The script also incorporates physical activity features and a binary target variable.

The script performs the following tasks:
- **Data Loading:** Loads NHANES dataset stored in .xpt files across five categories: Demographics, Dietary, Examination, Laboratory, and Questionnaire.
- **File Structure Compliance:** Follows the NHANES website's folder structure, enhancing organization and accessibility.
- **DataFrame Creation:** Converts each .xpt file into a DataFrame. These DataFrames undergo preprocessing to align each subject (identified by `SEQN`) with a single row.
- **Data Integration:** Merges individual DataFrames into a comprehensive one containing all NHANES variables.
- **Inclusion of Additional Data:** Physical activity features and a binary target variable are merged into the main DataFrame.
- **Data Cleaning:** Addresses a `read_sas` method-related bug, substituting an anomalous value (5.397605346934028e-79) with zero.
- **Output Formats:** The complete DataFrame is exported in both CSV and pickle formats.


### 8. data_preprocessing.ipynb
NHANES Data Cleaning and Preprocessing

This script focuses on cleaning and preprocessing the NHANES (National Health and Nutrition Examination Survey) data, crucial for ensuring data quality and reliability. It handles various data issues, including missing values and specific data encodings, preparing the dataset for advanced analysis or modeling.

Key functionalities of the script:
- **Data Loading:** Reads the aggregated NHANES DataFrame from 'full_df.csv'.
- **Fixing read_sas bug in pandas:** Corrects a known issue in pandas related to reading SAS files by replacing an anomalous value (5.397605346934028e-79) with zero.
- **Handling Missing Values:** Identifies and processes missing values in NHANES data, substituting specific missing codes with NaN to ensure data clarity.
- **Age Topcoding:** Addresses NHANES guidelines for individuals aged 80 and over, excluding them from the dataset to focus on a demographic with a potential for low bone density.
- **Data Storage:** Saves the cleaned and preprocessed DataFrame in both CSV ('full_df.csv') and pickle ('full_df.pkl') formats for further use.

### 9. split_dataset.ipynb
Data Splitting

- **Initial Setup:**
  - Importing necessary libraries: `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, `csv`.
  - Importing `train_test_split` from `sklearn.model_selection` for data splitting.

- **Loading Dataset:**
  - Loading the pre-processed NHANES data from 'full_df.csv' into a DataFrame.

- **Feature Selection:**
  - Selecting relevant features for analysis, including demographics, body measurements, and physical activity levels.

- **Handling Missing Values:**
  - Identifying missing values in the dataset.
  - Removing participants with missing target variables and physical activity data.

- **Dataframe Refinement:**
  - Ensuring the refined DataFrame contains no missing values.
  - Confirming the size of the DataFrame post refinement.

- **Data Visualization:**
  - Utilizing `seaborn` to create pair plots for selected features against the target variable.

- **Data Splitting:**
  - Splitting data into training and testing sets using `train_test_split`.
  - Validating the split by checking for overlaps and ensuring data consistency between the original and split datasets.

- **ID Management:**
  - Saving the IDs of the train and test datasets to CSV files for reference.
  - Testing the split by reading back the IDs and comparing them with the original dataset.


### 10. baseline_classification.ipynb
NHANES Baseline Classification Models

This Jupyter Notebook script focuses on building baseline classification models using the NHANES dataset. It incorporates a comprehensive approach, from data preparation to model evaluation, using various machine learning techniques.

The script performs the following tasks:

- **Initial Setup:**
  - Importing libraries: `numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`, and others for data manipulation, visualization, and machine learning.
  - Loading machine learning utilities from `sklearn` for preprocessing, model building, hyperparameter optimization, and performance evaluation.

- **Loading and Preparing Data:**
  - Loading split data (training and testing sets) from CSV files.
  - Converting target variables (`y_train`, `y_test`) to one-dimensional arrays.

- **Feature Scaling:**
  - Applying `StandardScaler` for feature normalization.

- **Model Building and Hyperparameter Optimization:**
  - Implementing various models: Decision Tree, Support Vector Classifier (SVC), Random Forest, Logistic Regression, and Naïve Bayes.
  - Utilizing `GridSearchCV` and `RandomizedSearchCV` for hyperparameter optimization.

- **Model Evaluation:**
  - Testing models' performance on training and testing datasets.
  - Evaluating models with metrics such as recall, cross-validation recall, classification reports, and confusion matrices.
  - Implementing functions for metrics reporting and overfitting evaluation.

- **Best Practices Discussion:**
  - Highlighting the importance of not using the test set for model comparison and the significance of validation sets for threshold settings.







## Dependencies
(List of libraries or frameworks needed.)

## Contributing
(Guidelines for how to contribute to the project.)

## License
(The type of license the project is released under.)

## Contact
(Your contact information for support or queries.)

