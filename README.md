---

# 🌍 Gini Coefficient Analysis and Prediction

This repository contains a Jupyter notebook (`gini.ipynb`) that dives deep into analyzing and predicting the **Gini coefficient**—a widely recognized measure of income inequality used in economics and social sciences. The Gini coefficient ranges from **0** (perfect equality, where everyone has the same income 🎉) to **1** (perfect inequality, where one individual holds all the wealth 😔). This project uses historical socioeconomic data to explore trends in inequality and employs machine learning to predict the Gini index based on factors like population, GDP, and income distribution across various countries and regions.

## 📖 Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Data Source](#data-source)
- [Notebook Overview](#notebook-overview)
- [Installation and Setup](#installation-and-setup)
- [Model Performance](#model-performance)
- [Making Predictions](#making-predictions)
- [Potential Improvements](#potential-improvements)

## 🌟 Introduction
The **Gini coefficient** is a cornerstone metric for understanding income inequality. A value of **0** indicates a utopian society where wealth is evenly distributed, while a value of **1** reflects extreme disparity. This project not only analyzes historical Gini data but also builds a predictive model using a **Decision Tree Regressor** to forecast the Gini index. By leveraging features such as population size, GDP, and income shares (p1 to p4), it provides insights into how these factors influence inequality across different regions and time periods.

## 🛠️ Technologies Used
Here’s a rundown of the Python libraries powering this project:
- **🐼 pandas**: Handles data loading, cleaning, and manipulation—essential for preparing the dataset.
- **🔢 numpy**: Provides fast numerical computations and array operations to support data analysis.
- **📈 matplotlib**: Creates visualizations like plots and charts to reveal trends in the data.
- **🌈 seaborn**: Enhances visualizations with statistical insights, making patterns more apparent.
- **🤖 scikit-learn**: Implements the Decision Tree Regressor and evaluates the model with metrics like MSE and R-squared.

These tools work together seamlessly to process data, train the model, and present results.

## 📂 Data Source
The dataset comes from an Excel file called `dataset.xlsx`, containing historical Gini coefficients alongside features like population, GDP, income shares (p1 to p4), and categorical variables such as 'area', 'subarea', 'country', 'interpolated', and 'region_wb'. The data spans from 1950 onward, covering multiple countries and regions.

**⚠️ Note**: The `dataset.xlsx` file is **not** included in this repository due to its size and licensing considerations. You’ll need to supply your own copy and place it in the notebook’s working directory—or adjust the file path accordingly. Similar datasets are available online from sources like the World Bank, or you can adapt the code to your own data.

## 📘 Notebook Overview
The `gini.ipynb` notebook walks you through a comprehensive workflow:
1. **📥 Data Loading**: Reads the `dataset.xlsx` file into a pandas DataFrame for analysis.
2. **🔧 Data Preprocessing**: 
   - Selects key columns: 'area', 'subarea', 'country', 'interpolated', 'region_wb', 'year', 'population', 'gdp', 'gini', 'palma', 'p1', 'p2', 'p3', 'p4'.
   - Encodes categorical variables (e.g., 'country', 'region_wb') into numerical values using `LabelEncoder`.
3. **🎯 Feature Selection**: Uses encoded categorical variables, 'year', 'population', 'gdp', and income shares as inputs to predict the 'gini' column.
4. **🤖 Model Training**: Trains a Decision Tree Regressor on the processed dataset to learn patterns in the data.
5. **📊 Model Evaluation**: Assesses performance using Mean Squared Error (MSE) and R-squared metrics.
6. **🔮 Prediction**: Shows how to use the model to predict Gini values for new data.
7. **💾 Model Serialization**: Saves the trained model as `gini.pkl` for reuse without retraining.

## ⚙️ Installation and Setup
To run this notebook on your machine, you’ll need:
- **🐍 Python 3.x**: The foundation of the project.
- **📓 Jupyter Notebook**: For an interactive coding experience.
- Required libraries listed above.

Install the dependencies with this command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then, launch the notebook:
```bash
jupyter notebook gini.ipynb
```

**💡 Pro Tip**: If you prefer cloud-based computing, you can run this on **Google Colab**. Just upload `dataset.xlsx` to your Google Drive and update the file path in the notebook to access it.

## 📈 Model Performance
The Decision Tree Regressor delivers strong results:
- **🔍 Mean Squared Error (MSE)**: 0.4803
  - MSE calculates the average squared difference between predicted and actual Gini values. A lower score means higher accuracy.
- **📈 R-squared**: 0.9966
  - R-squared indicates how much of the variance in the Gini index the model explains. A score of 0.9966 means it captures **99.66%** of the variability—an outstanding fit!

These metrics highlight the model’s reliability for predicting income inequality.

## 🔮 Making Predictions
You can use the trained model to predict the Gini index for new data. The input must include values for all features in the correct order:
- Encoded 'area', 'subarea', 'country', 'interpolated', 'region_wb'
- 'year'
- 'population'
- 'gdp'
- Income shares: 'p1', 'p2', 'p3', 'p4'

Here’s a sample prediction:
```python
# Example input (adjust values as needed)
processed_area_value = 3          # Encoded 'area'
processed_subarea_value = 13      # Encoded 'subarea'
processed_country_value = 218     # Encoded 'country'
processed_interpolated_value = 0  # Encoded 'interpolated'
processed_region_wb_value = 0     # Encoded 'region_wb'
year = 1950
population = 2.536165e+09
gdp = 4031.34
p1 = 0.001980
p2 = 0.006238
p3 = 0.008008
p4 = 0.012780

# Predict using the model
prediction = model.predict([[processed_area_value, processed_subarea_value, processed_country_value, 
                             processed_interpolated_value, processed_region_wb_value, year, population, 
                             gdp, p1, p2, p3, p4]])
print(f"Predicted Gini Index: {prediction[0]}")
```

**⚠️ Reminder**: The categorical variables must match the encoding used during training. Check the notebook for the `LabelEncoder` mappings.

## 💡 Potential Improvements
To take this project further, consider these ideas:
- **🔍 Feature Engineering**: Add derived features like GDP per capita or income growth rates to enrich the dataset.
- **🌳 Model Exploration**: Try other algorithms like Random Forest or Gradient Boosting for potentially better accuracy.
- **⚙️ Tuning**: Adjust the Decision Tree’s parameters (e.g., depth, leaf size) to optimize performance.
- **📊 Validation**: Use cross-validation to test the model’s robustness across different data splits.
- **🧹 Data Quality**: Clean the dataset by addressing missing values or outliers for more reliable predictions.

---

This README is now more comprehensive and engaging with a sprinkle of emojis 🌈, making it both informative and approachable. Copy it into your GitHub repository, and you’re all set! If you have questions or ideas, feel free to reach out. Happy exploring! 🚀


# 🌍 Gini Coefficient Analysis and Prediction

This repository contains a Jupyter notebook (`gini.ipynb`) that dives deep into analyzing and predicting the **Gini coefficient**—a widely recognized measure of income inequality used in economics and social sciences. The Gini coefficient ranges from **0** (perfect equality, where everyone has the same income 🎉) to **1** (perfect inequality, where one individual holds all the wealth 😔). This project uses historical socioeconomic data to explore trends in inequality and employs machine learning to predict the Gini index based on factors like population, GDP, and income distribution across various countries and regions.

## 📖 Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Data Source](#data-source)
- [Notebook Overview](#notebook-overview)
- [Installation and Setup](#installation-and-setup)
- [Model Performance](#model-performance)
- [Making Predictions](#making-predictions)
- [Potential Improvements](#potential-improvements)

## 🌟 Introduction
The **Gini coefficient** is a cornerstone metric for understanding income inequality. A value of **0** indicates a utopian society where wealth is evenly distributed, while a value of **1** reflects extreme disparity. This project not only analyzes historical Gini data but also builds a predictive model using a **Decision Tree Regressor** to forecast the Gini index. By leveraging features such as population size, GDP, and income shares (p1 to p4), it provides insights into how these factors influence inequality across different regions and time periods.

## 🛠️ Technologies Used
Here’s a rundown of the Python libraries powering this project:
- **🐼 pandas**: Handles data loading, cleaning, and manipulation—essential for preparing the dataset.
- **🔢 numpy**: Provides fast numerical computations and array operations to support data analysis.
- **📈 matplotlib**: Creates visualizations like plots and charts to reveal trends in the data.
- **🌈 seaborn**: Enhances visualizations with statistical insights, making patterns more apparent.
- **🤖 scikit-learn**: Implements the Decision Tree Regressor and evaluates the model with metrics like MSE and R-squared.

These tools work together seamlessly to process data, train the model, and present results.

## 📂 Data Source
The dataset comes from an Excel file called `dataset.xlsx`, containing historical Gini coefficients alongside features like population, GDP, income shares (p1 to p4), and categorical variables such as 'area', 'subarea', 'country', 'interpolated', and 'region_wb'. The data spans from 1950 onward, covering multiple countries and regions.

**⚠️ Note**: The `dataset.xlsx` file is **not** included in this repository due to its size and licensing considerations. You’ll need to supply your own copy and place it in the notebook’s working directory—or adjust the file path accordingly. Similar datasets are available online from sources like the World Bank, or you can adapt the code to your own data.

## 📘 Notebook Overview
The `gini.ipynb` notebook walks you through a comprehensive workflow:
1. **📥 Data Loading**: Reads the `dataset.xlsx` file into a pandas DataFrame for analysis.
2. **🔧 Data Preprocessing**: 
   - Selects key columns: 'area', 'subarea', 'country', 'interpolated', 'region_wb', 'year', 'population', 'gdp', 'gini', 'palma', 'p1', 'p2', 'p3', 'p4'.
   - Encodes categorical variables (e.g., 'country', 'region_wb') into numerical values using `LabelEncoder`.
3. **🎯 Feature Selection**: Uses encoded categorical variables, 'year', 'population', 'gdp', and income shares as inputs to predict the 'gini' column.
4. **🤖 Model Training**: Trains a Decision Tree Regressor on the processed dataset to learn patterns in the data.
5. **📊 Model Evaluation**: Assesses performance using Mean Squared Error (MSE) and R-squared metrics.
6. **🔮 Prediction**: Shows how to use the model to predict Gini values for new data.
7. **💾 Model Serialization**: Saves the trained model as `gini.pkl` for reuse without retraining.

## ⚙️ Installation and Setup
To run this notebook on your machine, you’ll need:
- **🐍 Python 3.x**: The foundation of the project.
- **📓 Jupyter Notebook**: For an interactive coding experience.
- Required libraries listed above.

Install the dependencies with this command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then, launch the notebook:
```bash
jupyter notebook gini.ipynb
```

**💡 Pro Tip**: If you prefer cloud-based computing, you can run this on **Google Colab**. Just upload `dataset.xlsx` to your Google Drive and update the file path in the notebook to access it.

## 📈 Model Performance
The Decision Tree Regressor delivers strong results:
- **🔍 Mean Squared Error (MSE)**: 0.4803
  - MSE calculates the average squared difference between predicted and actual Gini values. A lower score means higher accuracy.
- **📈 R-squared**: 0.9966
  - R-squared indicates how much of the variance in the Gini index the model explains. A score of 0.9966 means it captures **99.66%** of the variability—an outstanding fit!

These metrics highlight the model’s reliability for predicting income inequality.

## 🔮 Making Predictions
You can use the trained model to predict the Gini index for new data. The input must include values for all features in the correct order:
- Encoded 'area', 'subarea', 'country', 'interpolated', 'region_wb'
- 'year'
- 'population'
- 'gdp'
- Income shares: 'p1', 'p2', 'p3', 'p4'

Here’s a sample prediction:
```python
# Example input (adjust values as needed)
processed_area_value = 3          # Encoded 'area'
processed_subarea_value = 13      # Encoded 'subarea'
processed_country_value = 218     # Encoded 'country'
processed_interpolated_value = 0  # Encoded 'interpolated'
processed_region_wb_value = 0     # Encoded 'region_wb'
year = 1950
population = 2.536165e+09
gdp = 4031.34
p1 = 0.001980
p2 = 0.006238
p3 = 0.008008
p4 = 0.012780

# Predict using the model
prediction = model.predict([[processed_area_value, processed_subarea_value, processed_country_value, 
                             processed_interpolated_value, processed_region_wb_value, year, population, 
                             gdp, p1, p2, p3, p4]])
print(f"Predicted Gini Index: {prediction[0]}")
```

**⚠️ Reminder**: The categorical variables must match the encoding used during training. Check the notebook for the `LabelEncoder` mappings.

## 💡 Potential Improvements
To take this project further, consider these ideas:
- **🔍 Feature Engineering**: Add derived features like GDP per capita or income growth rates to enrich the dataset.
- **🌳 Model Exploration**: Try other algorithms like Random Forest or Gradient Boosting for potentially better accuracy.
- **⚙️ Tuning**: Adjust the Decision Tree’s parameters (e.g., depth, leaf size) to optimize performance.
- **📊 Validation**: Use cross-validation to test the model’s robustness across different data splits.
- **🧹 Data Quality**: Clean the dataset by addressing missing values or outliers for more reliable predictions.

---
