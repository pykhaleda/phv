## Major Libraries
import pandas as pd 
import os


## sklearn -- Preprocessing & Tuning & Transformation
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector





## Read the CSV file using pandas
path = os.path.join(os.getcwd(), 'housing.csv')
df_housing = pd.read_csv(path)


## Replace the  (<1H OCEAN) to (1H OCEAN) -- will cause ane errors in Deploymnet
df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')

## Try to make some Feature Engineering --> Feature Extraction --> Add the new column to the main DF
df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
df_housing['bedrooms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']


## Split the whole Dataset to Feature & Target

X= df_housing.drop(columns=['median_house_value'], axis=1) ## Features
y = df_housing['median_house_value']         

## Split to Trainging and test Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.15, random_state=53)



## Separete the columns according to type (numerical or categorical)
num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float64']]
catg_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float64']]






## We can get much much easier like the following
## numerical pipeline
num_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(num_cols)),
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
])

## categorical pipeline
cat_pipeline = Pipeline(steps=[
                                ('selector', DataFrameSelector(catg_cols)),
                                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('one', OneHotEncoder(sparse_output=False))
])

## concatenate both two pipelines
total_pipeline = FeatureUnion(transformer_list=[
                                                ('num', num_pipeline),
                                                ('categ', cat_pipeline)
])

                            
## deal with (total_pipeline) as an instance -- fit and transform to train dataset and transform only to other datasets
X_train_final = total_pipeline.fit_transform(X_train)



def preprocess_new (X_new):
    return total_pipeline.transform(X_new)

































