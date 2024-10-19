#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# In[152]:


train1 = pd.read_csv('pre-owned cars.csv')
train2 = pd.read_csv('cardata.csv')
train4 = pd.read_csv('cardekho_dataset.csv')
train5 = pd.read_csv('cars_24_combined.csv')


# In[153]:


train1 = train1.drop(['reg_year', 'spare_key', 'reg_number', 'title', 'overall_cost', 'has_insurance'], axis=1)
train1 = train1[train1['model'].notnull()]
train1['make_year'] = 2024.0 - train1['make_year']
train1['model'] = train1['model'].apply(lambda x: x.split()[0])
train2['brand'] = train2['Name'].str.split(' ').str[0]
train2['model'] = train2['Name'].str.split(' ').str[1]
train2 = train2.drop(columns=['Name'])
train2 = train2.drop(columns=['Unnamed: 0'])
train2['Mileage'].fillna('18.9 kmpl', inplace=True)
train2['Power'].fillna('74 bhp', inplace=True)
train2['Engine'].fillna('1197 CC', inplace=True)
train2['Mileage'] = train2['Mileage'].apply(lambda x: float(x.replace('km/kg', '').strip()) * 1.39 if 'km/kg' in x else float(x.replace('kmpl', '').strip()))
train2['Power'] = pd.to_numeric(train2['Power'].str.replace('bhp', '').str.strip(), errors='coerce')
train2['Power'] = train2['Power'].astype(float)
train2['Engine'] = train2['Engine'].str.replace('CC', '').str.strip().astype(float).astype(int)
train4 = train4.drop(columns = ['Unnamed: 0', 'car_name', 'seller_type'])
train5['Location'] = train5['Location'].str.split('-').str[0]
location_mapping = {
    'MH': 'maharashtra',
    'KA': 'karnataka',
    'DL': 'delhi',
    'GJ': 'gujarat',
    'TN': 'tamil nadu',
    'TS': 'telangana',
    'HR': 'haryana',
    'UP': 'uttar pradesh',
    'WB': 'west bengal',
    'PB': 'punjab',
    'RJ': 'rajasthan',
    'KL': 'kerala',
    'MP': 'madhya pradesh',
    'BR': 'bihar',
    'AP': 'andhra pradesh',
    'CH': 'chandigarh',
    '22': 'maharashtra' 
}
train5['Location'] = train5['Location'].replace(location_mapping)
train5 = train5.drop(columns=['Unnamed: 0'])
train5['Year'] = 2024 - train5['Year']
train5['Company'] = train5['Car Name'].str.split(' ').str[0]
train5['Model'] = train5['Car Name'].str.split(n=1).str[1]
train5 = train5.drop(columns=['Car Name'])
train1.rename(columns={'engine_capacity(CC)':'engine', 'make_year':'age', 'fuel_type':'fuel', 'km_driven':'km', 'ownership':'owner'}, inplace=True)
train2.rename(columns={'Location':'location', 'Year':'age', 'Kilometers_Driven':'km', 'Fuel_Type':'fuel', 'Transmission':'transmission', 'Owner_Type':'owner', 'Mileage':'mileage', 'Engine':'engine', 'Power':'power', 'Seats':'seats', 'Price':'price'}, inplace=True)
train4.rename(columns={'vehicle_age':'age', 'km_driven':'km', 'fuel_type':'fuel', 'transmission_type':'transmission', 'max_power':'power', 'selling_price':'price'}, inplace=True)
train5.rename(columns={'Year':'age', 'Distance':'km', 'Owner':'owner', 'Fuel':'fuel', 'Location':'location', 'Drive':'transmission', 'Type':'type', 'Price':'price', 'Company':'brand', 'Model':'model'}, inplace=True)
def convert_strings_to_lowercase(df):
    string_columns = df.select_dtypes(include='object').columns
    for col in string_columns:
        df[col] = df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)
    return df

train1 = convert_strings_to_lowercase(train1)
train2 = convert_strings_to_lowercase(train2)
train4 = convert_strings_to_lowercase(train4)
train5 = convert_strings_to_lowercase(train5)
train2['price'] = train2['price'] * 100000
owner_mapping_train1 = {
    '1st owner': 1,
    '2nd owner': 2,
    '3rd owner': 3
}
owner_mapping_train2 = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth & above': 4
}
train1['owner'] = train1['owner'].map(owner_mapping_train1)
train2['owner'] = train2['owner'].map(owner_mapping_train2)
df = pd.concat([train1, train2, train4, train5], axis=0, ignore_index=True)
df['transmission'].fillna('manual', inplace=True)
df['transmission'] = df['transmission'].map({'manual': 0, 'automatic': 1}).astype(float)
label_encoder = LabelEncoder()
for column in ['fuel', 'location', 'type', 'brand', 'model']:
    non_null_values = df[column][df[column].notnull()]
    df.loc[df[column].notnull(), column] = label_encoder.fit_transform(non_null_values)
    df[column] = df[column].astype(float)
df = df.drop(df.nlargest(50, 'price').index)


# In[ ]:


params = {
    'n_estimators': 798,
    'max_depth': 8,
    'learning_rate': 0.1843536576638147,
    'subsample': 0.9222485469912514,
    'colsample_bytree': 0.5992513794410232,
    'min_child_weight': 2,
    'reg_lambda': 65.95567228051213,
    'reg_alpha': 52.333112768082394,
    'random_state': 42
}

model = xgb.XGBRegressor(**params)
X = df.drop(columns=['price'])  
y = df['price']
model.fit(X, y)

