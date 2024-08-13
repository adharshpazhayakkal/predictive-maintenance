import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Correct the 'Target' column based on 'Failure Type'
    data.loc[data['Failure Type'] == 'Random Failures', 'Target'] = 1
    data.loc[data['Failure Type'] == 'No Failure', 'Target'] = 0
    
    # Drop the 'Type' and 'Failure Type' columns
    data = data.drop(columns=['Failure Type','UDI','Product ID'])
    type_mapping = {'L': 1, 'M': 2, 'H': 3}
    data['Type'] = data['Type'].replace(type_mapping)
    
    # Split data into features and target
    X = data.drop(columns=['Target'])  # Replace 'Target' with your actual target column name if different
    y = data['Target']  # Ensure 'Target' is the correct column name
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

