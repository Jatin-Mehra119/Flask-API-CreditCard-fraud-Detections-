from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


# Custom transformer to preprocess date and time features
class DateTimePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.column] = pd.to_datetime(df[self.column])
        df['year'] = df[self.column].dt.year
        df['month'] = df[self.column].dt.month
        df['day'] = df[self.column].dt.day
        df['hour'] = df[self.column].dt.hour
        df['minute'] = df[self.column].dt.minute
        df['second'] = df[self.column].dt.second
        df['day_of_week'] = df[self.column].dt.dayofweek
        
        def get_part_of_day(hour):
            if hour < 6:
                return 'Night'
            elif hour < 12:
                return 'Morning'
            elif hour < 18:
                return 'Afternoon'
            else:
                return 'Evening'
        
        df['part_of_day'] = df['hour'].apply(get_part_of_day)
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df = df.drop(columns=[self.column])
        return df

# Custom transformer to calculate age
class AgeCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, dob_column):
        self.dob_column = dob_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        current_year = pd.Timestamp.now().year
        X['age'] = current_year - pd.to_datetime(X[self.dob_column]).dt.year
        return X.drop(columns=[self.dob_column])

# Custom transformer for label encoding
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {col: LabelEncoder() for col in columns}

    def fit(self, X, y=None):
        for col in self.columns:
            self.encoders[col].fit(X[col])
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = self.encoders[col].transform(X_copy[col])
        return X_copy

# Custom transformer for clustering
class LocationClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, customer_columns, merchant_columns, customer_n_clusters=10, merchant_n_clusters=10):
        self.customer_columns = customer_columns
        self.merchant_columns = merchant_columns
        self.customer_n_clusters = customer_n_clusters
        self.merchant_n_clusters = merchant_n_clusters
        self.scaler_customer = StandardScaler()
        self.scaler_merchant = StandardScaler()
        self.kmeans_customer = KMeans(n_clusters=customer_n_clusters, n_init='auto', random_state=42)
        self.kmeans_merchant = KMeans(n_clusters=merchant_n_clusters, n_init='auto', random_state=42)

    def fit(self, X, y=None):
        self.scaler_customer.fit(X[self.customer_columns])
        self.scaler_merchant.fit(X[self.merchant_columns])
        self.kmeans_customer.fit(self.scaler_customer.transform(X[self.customer_columns]))
        self.kmeans_merchant.fit(self.scaler_merchant.transform(X[self.merchant_columns]))
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['customer_location_cluster'] = self.kmeans_customer.predict(self.scaler_customer.transform(X[self.customer_columns]))
        X_copy['merchant_location_cluster'] = self.kmeans_merchant.predict(self.scaler_merchant.transform(X[self.merchant_columns]))
        return X_copy

# Custom transformer for reordering columns
class ColumnReorderer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
# Custom Transformer for dropping unnecessary rows
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


# Defining the pipeline
column_order = [
    'year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week', 'part_of_day', 'is_weekend',
    'merchant', 'category', 'amt', 'gender', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 
    'merch_long', 'age', 'customer_location_cluster', 'merchant_location_cluster'
]

preprocessing_pipeline = Pipeline([
    ('datetime_preprocessing', DateTimePreprocessor('trans_date_trans_time')),
    ('age_calculator', AgeCalculator('dob')),
    ('label_encoding', CustomLabelEncoder(columns=['gender', 'merchant', 'category', 'part_of_day'])),
    ('location_clustering', LocationClusterer(customer_columns=['lat', 'long'], merchant_columns=['merch_lat', 'merch_long'])),
    ('drop_columns', ColumnDropper(columns_to_drop=['Unnamed: 0.1', 'Unnamed: 0', 'trans_num', 'first', 'last', 'cc_num', 'street', 'city', 'state', 'zip', 'job'])),
    ('column_reordering', ColumnReorderer(columns=column_order))
])

# Fit and transform the DataFrame
# processed_df = preprocessing_pipeline.fit_transform(df)
# processed_df['is_fraud'] = df['is_fraud']


# Applying the pipeline
# processed_df = preprocessing_pipeline.fit_transform(df)
# processed_df['is_fraud'] = df['is_fraud']