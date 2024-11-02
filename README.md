# Wine-Quality-Prediction
This repository contains a project aimed at predicting wine quality using the Random Forest algorithm. The model leverages various physicochemical attributes of wine to provide accurate quality assessments.
## Overview
Wine quality assessment is crucial for winemakers to ensure high standards. This project utilizes the Random Forest algorithm, an ensemble learning method, to predict wine quality based on various input features.
## Dataset
The dataset used for this project is the Red Wine Quality dataset(https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) from the Kaggle. It contains:
- **Input Features**: 
  - Fixed Acidity
  - Volatile Acidity
  - Citric Acid
  - Residual Sugar
  - Chlorides
  - Free Sulfur Dioxide
  - Total Sulfur Dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
- **Target Variable**: Quality (rated from 3 to 8)
- ## Installation
To run this project, ensure you have Python 3.x installed. Then, clone this repository and install the required packages:

```bash
git clone https://github.com/Srikanth-Reddy-01/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
or else (Another Method)
**by downloading the file,open in VS Code before opening the file you should install the VS Code in your PC/Laptop.Then open the files in Vs code ,then click on Terminal (it is located at top of the header in vs code) then type streamlit run wine.py  it will automatically redirect to web browser**.
while enterning the values you should remember the values should be seperated by commas(,) and remove whitespaces.
## Model Evaluation
The model's performance is evaluated using metrics such as:
Accuracy
Results will be printed in the console after model training.

## Feature Importance
The Random Forest algorithm provides feature importance scores, which help identify the most significant factors affecting wine quality. This information can guide winemaking processes and quality control.
