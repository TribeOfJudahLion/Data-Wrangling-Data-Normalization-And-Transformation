import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)

# Initial Data Overview
print("Data Overview:")
print(data.head())
print("\nMissing Values:")
print(data.isna().sum())

# Debugging: Check the columns
print("\nColumns in the dataset:")
print(data.columns)

# Ensure 'income' column is present
if 'income' in data.columns:
    # Visualize data distribution for numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns

    for feature in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    # Handle missing values
    data.dropna(inplace=True)

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Visualize the effect of normalization
    for feature in numerical_features:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[feature], kde=True, bins=20)
        plt.title(f'Distribution of {feature} after Min-Max Scaling')
        plt.show()

    # Encode categorical features excluding 'income'
    categorical_features = data.select_dtypes(include=['object']).columns.difference(['income'])
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid dummy variable trap
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
    encoded_data.columns = encoder.get_feature_names_out(categorical_features)

    # Combine encoded categorical features with the rest of the dataset, excluding the original categorical features
    data_encoded = data.drop(categorical_features, axis=1)
    data_encoded = pd.concat([data_encoded, encoded_data], axis=1)

    # Check if 'income' column is still present
    if 'income' in data_encoded.columns:
        # Split the dataset into training and testing sets
        X = data_encoded.drop('income', axis=1)
        y = data_encoded['income']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Output information about the final dataset
        print("\nFinal Dataset:")
        print(X_train.head())
    else:
        raise ValueError("Column 'income' not found in the dataset after processing")
else:
    raise ValueError("Column 'income' not found in the dataset")
