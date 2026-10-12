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

#### Validation framework

The datafreame has been split into three parts:

- **Train dataset:** The dataset used to train the model. (60%)
- **Validation dataset:** The dataset used to evaluate the model's performance. (20%)
- **Test dataset:** The dataset used to evaluate the model's performance. (20%)

#### Data analysis

- There are no missing values in the dataset.
- The data is considered balanced with a ratio of 60:40.
- Cols numerical are: pH, Temperature, Color.
- Cols categorical are: Taste, Odor, Fat, Turbidity.
- Turbidity has the highest impact on the quality of milk. The higher the turbidity, the more bacteria can grow in the 
milk, which can lead to spoilage. Therefore, it is important to monitor the turbidity levels in the milk to ensure that 
it is not too high.

#### Risk ratio analysis

*   **`turbidity` (Risk Ratio = 1.30):** When the milk presents high turbidity, the probability of it being 
**Rejected (Class 0)** increases by **30%** compared to the global rejection rate baseline. In a dairy production line, 
high turbidity is a physical indicator of protein destabilization or microbial coagulation, making it a critical 
"Red Flag" for quality control operators.
*   **`fat`, `taste`, and `odor` (Risk Ratio ≈ 1.00):** All these features show a Risk Ratio of 1. This means that 
grouping the data by these features results in a rejection rate that is identical to the global average (~37%). 
They do not increase or decrease the risk of milk spoilage independently.

*Operational Impact:* In an automated sensing environment, any milk batch triggering a high turbidity alert should immediately be flagged for secondary rigorous testing (like pH verification), as it significantly elevates the risk of contaminating the factory silos.


#### Mutual Information Insights

Based on the Mutual Information (MI) scores, we can observe a clear distinction in how much information each binary 
feature shares with our target variable:

*   **`turbidity` (0.073831):** This feature has, by far, the highest Mutual Information score among all categorical 
variables. It indicates that turbidity is a strong predictor and shares a significant dependency with the milk quality 
outcome. 
*   **`fat`, `taste`, and `odor` (< 0.005):** These features show extremely low MI scores, close to zero. This implies 
that, on their own, knowing whether the milk has a specific fat, taste, or odor configuration provides almost no global statistical information to determine if the milk will be accepted or rejected. 

*Conclusion for Feature Selection:* While `turbidity` is an essential driver for the model, `fat`, `taste`, and `odor` are candidates for lower importance, and the model will need to evaluate them strictly in combination with continuous 
features like `pH` and `temperature`.

#### Correlation Analysis

### Numerical Features Correlation Insights

We evaluated the linear relationship between the continuous features and our target variable using the absolute Pearson 
correlation coefficients. The results provide critical insights into how the physical parameters behave:

*   **`temperature` (0.499257):** This feature displays a **strong moderate correlation** with milk quality. 
Temperature is the single most dominant linear predictor among all continuous features. In a dairy facility, this 
aligns perfectly with industry standards: poor temperature management accelerates bacterial growth, leading directly to 
milk spoilage and immediate rejection at the factory gate.
*   **`colour` (0.177302):** It shows a **weak but noticeable correlation**. This suggests that changes in light 
reflectance (milk turning slightly translucent or yellowish) bear a relationship with quality, although it is not 
strong enough to be a standalone factor.
*   **`ph` (0.015867):** pH shows an **almost non-existent linear correlation** (close to 0). Although pH is 
biochemically the most critical indicator of milk freshness, its linear correlation score is close to zero because the 
relationship is strictly non-linear. As established in the domain description, high-quality milk lives strictly within 
a tight optimal window (6.25 to 6.90).
