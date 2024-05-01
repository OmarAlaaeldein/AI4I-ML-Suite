# AI-4I ML Suite

This repository contains a Jupyter Notebook that explores various machine learning algorithms for predicting machine failures using the AI4I 2020 Predictive Maintenance Dataset. The goal is to develop a robust model that can accurately predict machine failures based on the provided features, enabling proactive maintenance and reducing downtime.

## Dataset

The AI4I 2020 Predictive Maintenance Dataset is a synthetic dataset that reflects real predictive maintenance data encountered in industry. It consists of 10,000 data points with 14 features, including unique identifiers, product IDs, air temperature, process temperature, rotational speed, torque, tool wear, and target variables indicating machine failure and tool wear failure (TWF).

The dataset can be downloaded from the UCI Machine Learning Repository: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Machine Learning Approaches

The notebook employs a diverse set of machine learning classifiers to analyze the dataset and predict machine failures. The following classifiers are used:

- XGBClassifier
- LGBMClassifier
- CatBoostClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- LogisticRegression
- SVC
- DecisionTreeClassifier
- KNeighborsClassifier
- MLPClassifier

## Model Evaluation

To evaluate the performance of the trained machine learning models, a combination of evaluation metrics is used. In addition to common metrics such as accuracy, precision, recall, and F1-score, the Jaccard similarity coefficient is incorporated as a measure of similarity between the predicted and true labels.

The micro-average of the Jaccard similarity and the micro-average of the F1-score are computed across all classes since this problem is multi-label. These two metrics are then multiplied to create a composite metric that combines their strengths, providing a more comprehensive assessment of the models' performance.

## Results

The notebook presents the performance of each classifier on the predictive maintenance dataset, along with visualizations and analysis. The best-performing models are identified, and their strengths and weaknesses are discussed.
