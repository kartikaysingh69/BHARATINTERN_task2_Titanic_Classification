# Titanic Survival Prediction System

## Overview

This is a classification system that predicts whether a person would have survived the sinking of the Titanic based on various factors such as socio-economic status, age, gender, and more. The Titanic dataset is a famous dataset in machine learning and is often used for classification tasks.

## Dataset

The dataset used for this project is the "Titanic: Machine Learning from Disaster" dataset, which can be obtained from the Kaggle website (https://www.kaggle.com/c/titanic/data). The dataset contains information about passengers on the Titanic, including features such as:

- `Pclass`: Passenger class (1st, 2nd, or 3rd)
- `Name`: Passenger's name
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`: Ticket number
- `Fare`: Fare paid
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- `Survived`: Survival status (0 = No, 1 = Yes)

## Dependencies

To run this project, you will need the following Python libraries:

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib
```

## Usage

1. Clone the repository to your local machine:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd TASK2
```

3. Run the Jupyter notebook or Python script to train and test the model:

```bash
jupyter notebook TASK2.ipynb
```

or

```bash
python TASK2.py
```

4. Follow the instructions in the notebook or script to load the dataset, preprocess the data, train the classification model, and make predictions.

## Model

The classification model used for this project can be any machine learning classifier such as Logistic Regression, Random Forest, or Support Vector Machine. You can choose the model that best suits your needs or experiment with different models to see which one performs best.

## Evaluation

The performance of the model can be evaluated using metrics such as accuracy, precision, recall, and F1-score. You can use the test set to evaluate the model's performance and make improvements as needed.

## Conclusion

This project demonstrates how to build a Titanic survival prediction system based on various factors. By analyzing the dataset and training a classification model, you can gain insights into which factors were most likely to lead to survival in the event of the Titanic sinking.

Feel free to modify and extend this project as needed for your specific requirements.
