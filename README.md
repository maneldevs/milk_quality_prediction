# Milk Quality Prediction

## Problem statement

In the food and dairy industry, ensuring the safety and quality of raw materials is critical. Milk is a highly 
perishable product prone to rapid bacterial growth and chemical degradation if stored or handled improperly. Relying 
solely on manual or delayed laboratory testing can lead to catastrophic waste, financial losses for producers, and 
severe public health risks. Incorporating automated machine learning systems at reception points allows for immediate 
risk mitigation.

The objective of this project is to build an automated quality control system that classifies raw milk into two 
categories: 

- **Target:** `Grade` (`1` for Acceptable Quality [High/Medium grade], `0` for Rejected Quality [Low grade]).
- **Features:** `pH`, `Temperature`, `Taste`, `Odor`, `Fat`, `Turbidity`, and `Color`.

By analyzing instant physicochemical parameters collected by internet-of-things (IoT) sensors during milk reception, 
the machine learning model will predict whether a specific batch meets the industrial standards or needs to be 
discarded before entering the production line.

- **Dairy Processing Plants:** Reduces the risk of ruining entire production batches.
- **Quality Assurance (QA) Teams:** Streamlines the inspection workflow with instant decision-making.
- **Consumers:** Ensures higher safety standards and consistency in final dairy products.

The models evaluated in this project are:

* **Logistic Regression:** A machine learning model that uses logistic regression to predict the quality of milk.
* **Decision Tree Classifier:** A machine learning model that uses decision tree to predict the quality of milk.
* **Random Forest Classifier:** A machine learning model that uses random forest to predict the quality of milk.
* **XGBoost Classifier:** A machine learning model that uses gradient boosting to predict the quality of milk.

To measure the success of our predictive models, as dataset target is balanced (60%-40%) the primary evaluation metrics selected are:

* **ROC-AUC (Primary):** Since this is a balanced classification task, Area Under the ROC Curve will be used to evaluate how well the model distinguishes between acceptable and rejected milk across all thresholds.
* **Accuracy (Secondary):** Used as a standard baseline metric for overall correctness.


## EDA

The dataset used for this project can be found in the `data` folder or can be downloaded from 
[Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). This dataset consists of 7 independent variables 
ie pH, Temperature, Taste, Odor, Fat, Turbidity, and Color and the target variable is the qualityGrade of the milk. The description of the dataset is as follows:

- **pH**: pH of the milk which ranges from 3 to 9.
- **Temperature (C)**: Temperature of the milk which ranges from 34 to 90.
- **Taste**: Taste of the milk which is a categorical variable with 2 levels: 0 for Bad and 1 for Good.
- **Odor**: Odor of the milk which is a categorical variable with 2 levels: 0 for Bad and 1 for Good.
- **Fat**: Fat of the milk which is a categorical variable with 2 levels: 0 for Low and 1 for High.
- **Turbidity**: Turbidity of the milk which is a categorical variable with 2 levels: 0 for Low and 1 for High.
- **Color**: Color of the milk which ranges from 240 to 255.
- **Grade(target)**: Represents the quality of the milk. Iss a categorical variable with 3 levels: low, medium and high.

#### Data preparation:

- Column names has been changed to lowercase and spaces have been replaced with underscores. 
- Column temprature has been renamed to temperature. 
- In column fat has been removed a final space.
- Target column grade value has been changed to 0 for low, 1 for medium and 1 for high.
- No missing values in the dataset.

#### Validation framework

The datafreame has been split into three parts:

- **Train dataset:** The dataset used to train the model. (60%)
- **Validation dataset:** The dataset used to evaluate the model's performance. (20%)
- **Test dataset:** The dataset used to evaluate the model's performance. (20%)