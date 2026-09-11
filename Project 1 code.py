# Project -1 Mutual funds analysis

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Getting Data
mutual_funds_data = pd.read_csv("Mutual_Funds_project.csv")
print(mutual_funds_data.head())
print("=======================")

print(mutual_funds_data.shape)
print("====================")

print(mutual_funds_data.columns)
print("==================")
print(mutual_funds_data.dtypes)
print("=====================")

print(mutual_funds_data.isnull().sum())
print('=====================')

# Removing unnecessary columns

cols_to_remove= ['Unnamed: 0']

mutual_funds_data.drop(columns= [col for col in cols_to_remove if col in mutual_funds_data.columns],inplace=True)

print(mutual_funds_data.columns)
print("=====================")

# removing commas and double quotes

mutual_funds_data["Net_Asset_Value(Rs.)"] = mutual_funds_data["Net_Asset_Value(Rs.)"].str.replace(",", "", regex=False).str.replace('"', '', regex=False)
mutual_funds_data["Min. Invest(Rs.)"] = mutual_funds_data["Min. Invest(Rs.)"].str.replace(",", "", regex=False).str.replace('"', '', regex=False)
mutual_funds_data["SIP Min. Inv.(Rs.)"] = mutual_funds_data["SIP Min. Inv.(Rs.)"].str.replace(",", "", regex=False).str.replace('"', '', regex=False)



mutual_funds_data.replace(
    ['N/A',' N/A', 'NA', '-', '--', 'nil','null', 'None'],
    np.nan,
    inplace=True
)


# Change data types for calculations

cols_to_convert = ['Net_Asset_Value(Rs.)','CAGR% 6 Months','CAGR% 1 Year','CAGR% 3 Year','Min. Invest(Rs.)','Exp. Ratio(%)','SIP Min. Inv.(Rs.)']
mutual_funds_data[cols_to_convert] = mutual_funds_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

print(mutual_funds_data.dtypes)
print("====================")


# Check null values
print(mutual_funds_data.isnull().sum())
print('=====================')

# Fill null values
numeric_cols = mutual_funds_data.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    mutual_funds_data[col] = mutual_funds_data[col].fillna(
        mutual_funds_data[col].mean()
    )

#  for categorical columns

mutual_funds_data['Benchmark'] = mutual_funds_data['Benchmark'].fillna(
    mutual_funds_data['Benchmark'].mode()[0]
)


print(mutual_funds_data.isnull().sum())
print('=====================')

print("==========================")

# mutual_funds_data.to_csv("Mutual_Funds_Cleaned.csv", index=False)
# print("File saved as csv")


# # Data description and understanding
#
print(mutual_funds_data.describe())
print("===================")

# Mean
print("\n===== MEAN =====")
print(mutual_funds_data[numeric_cols].mean())

# Median
print("\n===== MEDIAN =====")
print(mutual_funds_data[numeric_cols].median())

# Mode
print("\n===== MODE =====")
print(mutual_funds_data[numeric_cols].mode().iloc[0])

# Standard deviation
print("\n===== STD DEVIATION =====")
print(mutual_funds_data[numeric_cols].std())
print('=========================')

# Analyze Fund distribution across different Return rates  ( 1 year or 3 year)

import matplotlib.pyplot as plt

mutual_funds_data['CAGR% 6 Months'].hist(bins=20)
plt.title("Distribution of 6-months CAGR")
plt.xlabel("CAGR %")
plt.ylabel("Number of Funds")
plt.show()

mutual_funds_data['CAGR% 1 Year'].hist(bins=20)
plt.title("Distribution of 1-Year CAGR")
plt.xlabel("CAGR %")
plt.ylabel("Number of Funds")
plt.show()

mutual_funds_data['CAGR% 3 Year'].hist(bins=20)
plt.title("Distribution of 3-Year CAGR")
plt.xlabel("CAGR %")
plt.ylabel("Number of Funds")
plt.show()
print("=====================")


##  Data Normalization

# MinMaxScaler from sklearn.preprocessing to  normalize numeric fields
# Compared returns and expense ratios on a common scale


# Applied Min-Max Scaling to normalize return metrics, expense ratios, investment amounts, and NAV values for fair comparison across mutual fund schemes.


# Before scaling
print("Before scaling data")
print("===================")
before_scaled_data = mutual_funds_data[['CAGR% 6 Months','CAGR% 1 Year','CAGR% 3 Year','Exp. Ratio(%)']]
print(before_scaled_data)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols_to_scale = [
    'CAGR% 6 Months',
    'CAGR% 1 Year',
    'CAGR% 3 Year',
    'Exp. Ratio(%)',
    'Net_Asset_Value(Rs.)',
    'Min. Invest(Rs.)',
    'SIP Min. Inv.(Rs.)'
]

mutual_funds_data[cols_to_scale] = scaler.fit_transform(
    mutual_funds_data[cols_to_scale]
)

print(mutual_funds_data[cols_to_scale].head())
print("======================")
print(mutual_funds_data)

print("Scaling is done")

print("=======================")


# # Compared returns and expense ratios on a common scale

print("After scaling")
print("======================")

After_scaling_df= mutual_funds_data[['CAGR% 6 Months','CAGR% 1 Year','CAGR% 3 Year','Exp. Ratio(%)']]
print(After_scaling_df)


import matplotlib.pyplot as plt

comparison = mutual_funds_data[['CAGR% 3 Year','Exp. Ratio(%)']].head(20)

comparison.plot(kind='bar', figsize=(12,6))

plt.title("Normalized Returns vs Expense Ratio")
plt.ylabel("Scaled Value (0-1)")
plt.show()

#==========================================================

# . Fund Scoring & Ranking
# Custom scoring formula based on:
# • High 3-Year Returns
# • Low Expense Ratio
# • Consistent 1-Year Return > 0


# Higher return = better
return_score = mutual_funds_data['CAGR% 3 Year']

# Lower expense ratio = better
expense_score = 1 - mutual_funds_data['Exp. Ratio(%)']

# Consistency bonus
mutual_funds_data['consistency_score'] = np.where(
    mutual_funds_data['CAGR% 1 Year'] > 0,
    1,
    0
)

# Final Score
mutual_funds_data['fund_score'] = (
    0.60 * return_score +
    0.30 * expense_score +
    0.10 * mutual_funds_data['consistency_score']
)


# RAnking funds
top_30_funds = mutual_funds_data.sort_values(
    by='fund_score',
    ascending=False
).head(30)

print("=============Fund score top 30 ==========")

print(top_30_funds[['Scheme',
                    'Category',
                    'Type',
                    'CAGR% 6 Months',
                    'CAGR% 1 Year',
                    'CAGR% 3 Year',
                    'Exp. Ratio(%)',
                    'fund_score']])

print("=========File saved =================")

# # Top 30 exporting
# top_30_funds.to_csv(
#     "Top_30_Mutual_Funds.csv",
#     index=False
# )




#  Custom scoring formula based on:
#
# High 3-Year CAGR Returns
# Low Expense Ratio
# Positive 1-Year Return Consistency
# Weighted scoring model for ranking mutual fund schemes

# "I created a weighted scoring model after normalizing the financial metrics. Funds received higher scores for strong 3-year returns and lower expense ratios. I also included a consistency factor by rewarding funds with positive 1-year returns. The final score was used to rank and identify the top-performing mutual funds."