# 1a Initial target investigation
import pandas as pd
import numpy as np
df = pd.read_csv('CaseStudyData.csv')
print('1.1')
print(df['IsBadBuy'].value_counts(dropna=False))
print(df['IsBadBuy'].value_counts(normalize=True))
# 1b Data Prep
print ('1.2')
## 1. Drop rows with no car details
df.dropna(axis=0, thresh=6, inplace=True)
## 2. Clear out invalid values (accross whole dataframe)
## convert '?' and '#VALUE!' to missing values ('NaN') accrosss the entire dataframe
df = df.replace('?', np.nan)
df = df.replace('#VALUE!', np.nan)

## 3. Set correct data type
df['VehYear'] = df['VehYear'].astype(int)
df['MMRAcquisitionAuctionAveragePrice'] = df['MMRAcquisitionAuctionAveragePrice'].astype(float)
df['MMRAcquisitionAuctionCleanPrice'] = df['MMRAcquisitionAuctionCleanPrice'].astype(float)
df['MMRAcquisitionRetailAveragePrice'] = df['MMRAcquisitionRetailAveragePrice'].astype(float)
df['MMRAcquisitonRetailCleanPrice'] = df['MMRAcquisitonRetailCleanPrice'].astype(float)
df['MMRCurrentAuctionAveragePrice'] = df['MMRCurrentAuctionAveragePrice'].astype(float)
df['MMRCurrentAuctionCleanPrice'] = df['MMRCurrentAuctionCleanPrice'].astype(float)
df['MMRCurrentRetailAveragePrice'] = df['MMRCurrentRetailAveragePrice'].astype(float)
df['MMRCurrentRetailCleanPrice'] = df['MMRCurrentRetailCleanPrice'].astype(float)
df['MMRCurrentRetailRatio'] = df['MMRCurrentRetailRatio'].astype(float)
df['VehBCost'] = df['VehBCost'].astype(float)

## 4. Convert selected variables to binary
Transmission_map = {'AUTO':1,'MANUAL':0,'Manual':0}
df['Transmission'] = df['Transmission'].map(Transmission_map)
IsOnlineSale_map = {0.0:0,'0':0,1.0:1,'1':1}
df['IsOnlineSale'] = df['IsOnlineSale'].map(IsOnlineSale_map)
ForSale_map = {'Yes':1,'YES':1,'yes':1,'No':0,'0':0}
df['ForSale'] = df['ForSale'].map(ForSale_map)

## 5. The WheelType columns are a special case. We will complete the 'WheelType' catogry column and drop the 'WheelTypeID' column
## This is to keep the column as an 'object' variable with user friendly names for modelling.
## To do this we'll first change zeroes in 'WheelTypeID' to missing ('NaN'), since '0' is not a recognized wheel type.
## Then we can fill all missing values with the column's mode value.
## Lastly, we map the integer values to their labels in the 'WheelType' column and apply the mapping
## (further down, we will drop the WheelTypeID column and WheelType will be 'One-hot encoded')
df['WheelTypeID'] = df['WheelTypeID'].replace('0',np.nan)
df['WheelTypeID'].fillna(df['WheelTypeID'].mode()[0], inplace=True)
WheelType_map = {'1':'Alloy','2':'Covers','3':'Special'}
df['WheelType'] = df['WheelTypeID'].map(WheelType_map)

## 6. Clearing out bad values
## 'Nationality' column we found conflicting values 'USA' and 'AMERICAN'
df['Nationality'] = df['Nationality'].replace('USA', 'AMERICAN')
## For 'MMR Price' columns we'll convert values under 5 to missing ('NaN')
mask = df['MMRAcquisitionAuctionAveragePrice'] < 5
df.loc[mask, 'MMRAcquisitionAuctionAveragePrice'] = np.nan
mask = df['MMRAcquisitionAuctionCleanPrice'] < 5
df.loc[mask, 'MMRAcquisitionAuctionCleanPrice'] = np.nan
mask = df['MMRAcquisitionRetailAveragePrice'] < 5
df.loc[mask, 'MMRAcquisitionRetailAveragePrice'] = np.nan
mask = df['MMRAcquisitonRetailCleanPrice'] < 5
df.loc[mask, 'MMRAcquisitonRetailCleanPrice'] = np.nan
mask = df['MMRCurrentAuctionAveragePrice'] < 5
df.loc[mask, 'MMRCurrentAuctionAveragePrice'] = np.nan
mask = df['MMRCurrentAuctionCleanPrice'] < 5
df.loc[mask, 'MMRCurrentAuctionCleanPrice'] = np.nan
mask = df['MMRCurrentRetailAveragePrice'] < 5
df.loc[mask, 'MMRCurrentRetailAveragePrice'] = np.nan
mask = df['MMRCurrentRetailCleanPrice'] < 5
df.loc[mask, 'MMRCurrentRetailCleanPrice'] = np.nan

## 7. Use 'fillna' to inpute missing values
df['Color'].fillna(df['Color'].mode()[0], inplace=True)
df['Transmission'].fillna(df['Transmission'].mode()[0], inplace=True)
df['Nationality'].fillna(df['Nationality'].mode()[0], inplace=True)
df['Size'].fillna(df['Size'].mode()[0], inplace=True)
df['TopThreeAmericanName'].fillna(df['TopThreeAmericanName'].mode()[0], inplace=True)
df['MMRAcquisitionAuctionAveragePrice'].fillna(df['MMRAcquisitionAuctionAveragePrice'].mean(), inplace=True)
df['MMRAcquisitionAuctionCleanPrice'].fillna(df['MMRAcquisitionAuctionCleanPrice'].mean(), inplace=True)
df['MMRAcquisitionRetailAveragePrice'].fillna(df['MMRAcquisitionRetailAveragePrice'].mean(), inplace=True)
df['MMRAcquisitonRetailCleanPrice'].fillna(df['MMRAcquisitonRetailCleanPrice'].mean(), inplace=True)
df['MMRCurrentAuctionAveragePrice'].fillna(df['MMRCurrentAuctionAveragePrice'].mean(), inplace=True)
df['MMRCurrentAuctionCleanPrice'].fillna(df['MMRCurrentAuctionCleanPrice'].mean(), inplace=True)
df['MMRCurrentRetailAveragePrice'].fillna(df['MMRCurrentRetailAveragePrice'].mean(), inplace=True)
df['MMRCurrentRetailCleanPrice'].fillna(df['MMRCurrentRetailCleanPrice'].mean(), inplace=True)
## for 'MMRCurrentRetailRatio' we can use the same calculation (CurRetailAvgPrice/CurRetailCleanPrice)
df['MMRCurrentRetailRatio'].fillna(df['MMRCurrentRetailAveragePrice'].div(df['MMRCurrentRetailCleanPrice']), inplace=True)
df['VehBCost'].fillna(df['VehBCost'].mean(), inplace=True)
df['IsOnlineSale'].fillna(df['IsOnlineSale'].mode()[0], inplace=True)
df['ForSale'].fillna(df['ForSale'].mode()[0], inplace=True)

## 8. Drop Unnecessary Variables
df.drop(['PurchaseID','PurchaseDate','WheelTypeID','PRIMEUNIT', 'AUCGUART'], axis=1, inplace=True)
print(df.info())

## 9. Formatting Categorical Variables
print('One-hot encoding')
print("Columns before:", len(df.columns))
df = pd.get_dummies(df)
print("Columns after:", len(df.columns))