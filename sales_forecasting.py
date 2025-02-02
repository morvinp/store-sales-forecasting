# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset = pandas.DataFrame(date, id, sales)
# dataset = dataset.drop_duplicates()

# Paste or type your script code here:
# Load your data first
dataset = dataset # Power BI will automatically make your data available as 'dataset'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

def create_train_validation_split(dataset, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)
    dataset = dataset.sort_values('date').reset_index(drop=True)
    
    for train_idx, val_idx in tss.split(dataset):
        final_train_idx = train_idx
        final_val_idx = val_idx
    
    return final_train_idx, final_val_idx

def plot_sales_forecast(dataset, model, feature_columns):
    # Create validation split
    train_idx, val_idx = create_train_validation_split(dataset)
    
    # Prepare data for visualization
    train_data = dataset.iloc[train_idx].copy()
    val_data = dataset.iloc[val_idx].copy()
    
    # Generate predictions for validation period
    val_predictions = model.predict(val_data[feature_columns])
    val_data['predictions'] = val_predictions
    
    # Calculate daily total sales
    train_daily = train_data.groupby('date')['sales'].sum().reset_index()
    val_daily = val_data.groupby('date').agg({
        'sales': 'sum',
        'predictions': 'sum'
    }).reset_index()
    
    # Create the plot
    fig = plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(train_daily['date'], train_daily['sales'], 
             label='Historical Sales', color='blue', alpha=0.6)
    
    # Plot actual validation data
    plt.plot(val_daily['date'], val_daily['sales'], 
             label='Actual Sales', color='green', linewidth=2)
    
    # Plot predictions
    plt.plot(val_daily['date'], val_daily['predictions'], 
             label='Predicted Sales', color='red', linestyle='--', linewidth=2)
    
    # Add vertical line to separate training and validation periods
    split_date = train_daily['date'].max()
    plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.5)
    plt.text(split_date, plt.ylim()[1], 'Train-Test Split', 
             rotation=90, verticalalignment='top')
    
    # Customize the plot
    plt.title('Store Sales Forecast', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Daily Sales', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

# Power BI will use this as the entry point
def main(dataset):
    # Convert dataset to pandas DataFrame if it isn't already
    dataset = pd.DataFrame(dataset)
    
    # Convert date column to datetime
    dataset['date'] = pd.to_datetime(dataset['date'])
    
    # Basic preprocessing
    # Create date features
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month
    dataset['day'] = dataset['date'].dt.day
    dataset['dayofweek'] = dataset['date'].dt.dayofweek
    dataset['quarter'] = dataset['date'].dt.quarter
    
    # Label encode categorical variables
    categorical_columns = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
    for col in categorical_columns:
        if col in dataset.columns:
            dataset[col + '_encoded'] = LabelEncoder().fit_transform(dataset[col].astype(str))
    
    # Prepare features for modeling
    feature_columns = []
    for col in ['store_nbr_encoded', 'family_encoded', 'city_encoded', 
                'state_encoded', 'type_encoded', 'cluster_encoded',
                'onpromotion', 'year', 'month', 'day', 'dayofweek',
                'quarter']:
        if col in dataset.columns:
            feature_columns.append(col)
    
    # Train model
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42
    )
    
    # Split data for visualization
    train_idx, val_idx = create_train_validation_split(dataset)
    train_data = dataset.iloc[train_idx]
    
    # Fit model
    model.fit(
        train_data[feature_columns],
        train_data['sales'],
        categorical_feature=[col for col in feature_columns if col.endswith('_encoded')]
    )
    
    # Create and return visualization
    return plot_sales_forecast(dataset, model, feature_columns)
    
# Generate the visualization
fig = main(dataset)
plt.show()
