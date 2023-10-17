import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

def load_data(file_path):
    return pd.read_csv(file_path)

def homepage(dataframe):
    st.title("Loan Prediction Data Introduction")
    st.write("This is an introduction Loan Prediction Dataset.")
    st.write(dataframe.head(10))
    st.write(dataframe.tail(10))
    st.write('Statistical Values')
    st.write(dataframe.describe())

def explore_missing_values(dataframe):
    st.subheader("1. Columns with Null Values")
    null_columns = dataframe.columns[dataframe.isnull().any()]
    if len(null_columns) > 0:
        st.write("The following columns have null values:")
        st.write(null_columns)
    else:
        st.write("No columns have null values in the DataFrame.")

def impute_data(dataframe, columns, imputation_method):
    
    if imputation_method == "Mean":
        dataframe[columns] = dataframe[columns].fillna(dataframe[columns].mean())
    elif imputation_method == "Median":
        dataframe[columns] = dataframe[columns].fillna(dataframe[columns].median())
    elif imputation_method == "Mode":
        dataframe[columns] = dataframe[columns].fillna(dataframe[columns].mode().iloc[0])

def check_imbalance(dataframe, feature):
    st.subheader("3. Imbalance Checking")
    st.write(f"Feature: {feature}")
    value_counts = dataframe[feature].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(value_counts.index, value_counts)
    plt.xlabel(feature)
    plt.ylabel("Counts")
    plt.title("Value Counts for Selected Feature")
    st.pyplot()

def visualize_and_handle_outliers(dataframe, feature):
    st.subheader("4. Outlier Detection and Handling")
    st.write(f"Visualizing Outliers for Feature: {feature}")
    plt.figure(figsize=(10, 6))
    sns.boxplot(dataframe, x=feature)
    plt.title(f"Box Plot of {feature}")
    st.pyplot()

    clean_outliers = st.checkbox("Clean Outliers (Z-score method)")
    if clean_outliers:
        z_scores = (dataframe[feature] - dataframe[feature].mean()) / dataframe[feature].std()
        z_score_threshold = st.number_input("Set a Z-score threshold to define outliers:", value=3.0)
        dataframe = dataframe[abs(z_scores) <= z_score_threshold]
        st.success("Outliers removed.")
        plt.figure(figsize=(10, 6))
        sns.boxplot(dataframe, x=feature)
        plt.title(f"Box Plot of {feature} after handling outliers")
        st.pyplot()
    else:
        st.warning("Outliers are not cleaned.")

def plot_histogram(dataframe, selected_feature):
    st.subheader("5. Histogram of features from Loan Pred Data")
    st.write(f"{selected_feature} Histogram")
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe[selected_feature], kde=True)
    st.pyplot()

def preprocess_data(dataframe, scaling_option, scaling_columns, encoding_option, encoding_columns, test_size, model_selection):
    df_processed = dataframe.copy()

    if scaling_option == "Standard Scaler":
        if scaling_columns:
            standard_scaler = StandardScaler()
            df_processed[scaling_columns] = standard_scaler.fit_transform(df_processed[scaling_columns])
    elif scaling_option == "Min-Max Scaler":
        if scaling_columns:
            min_max_scaler = MinMaxScaler()
            df_processed[scaling_columns] = min_max_scaler.fit_transform(df_processed[scaling_columns])

    if encoding_option == "Label Encoding":
        label_encoder = LabelEncoder()
        for col in encoding_columns:
            if col in df_processed.columns:
                df_processed[col] = label_encoder.fit_transform(df_processed[col])
            else:
                st.write(f"Warning: Column '{col}' not found for encoding.")

    st.write("Processed Data:")
    st.write(df_processed)


    X = df_processed.iloc[:, :-1]  # Select all columns except the last one
    y = df_processed.iloc[:, -1] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if model_selection == "XGBClassifier":
        param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
        }
        model = XGBClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    elif model_selection == "CatBoostClassifier":
        param_grid = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 4, 5],
        'l2_leaf_reg': [1, 3, 5]
        } 
        model = CatBoostClassifier()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(confusion)
    st.write('Precision Score:')
    st.write(precision)
    st.write("Recall Score:")
    st.write(recall)
    st.write('F1 Score:')
    st.write(f1)

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.image("/Users/mehrac/Desktop/DSA-Python/homepage.jpg", width=750)
    st.sidebar.image("/Users/mehrac/Desktop/DSA-Python/DSA_logo.png")

    df_loan = load_data('/Users/mehrac/Desktop/DSA-Python/loan_pred.csv')
    df_water = load_data('/Users/mehrac/Desktop/DSA-Python/water_potability.csv')

    st.sidebar.title("Navigation")
    st.title("Select a Dataset")
    selected_dataset = st.selectbox("Choose a dataset:", ('df_loan', 'df_water'))
    page_selection = st.sidebar.selectbox("Go to:", ("Homepage", "EDA", "Modeling"))

    if selected_dataset == 'df_loan':
        if page_selection == "Homepage":
            homepage(df_loan)
        elif page_selection == "EDA":
            explore_missing_values(df_loan)
            selected_imputation_columns = st.multiselect("Select columns to impute:", df_loan.columns, key="imputing_columns")
            imputation_method = st.selectbox("Select an imputation method:", ["Mean", "Median", "Mode"])
            if st.button("Impute Data"):
                impute_data(df_loan, selected_imputation_columns, imputation_method)

            imbalance_feature = st.selectbox("Select a feature to check for imbalance:", df_loan.columns)
            if st.button("Check Imbalance"):
                check_imbalance(df_loan, imbalance_feature)

            outlier_feature = st.selectbox("Select a feature to visualize and clean for outliers:", df_loan.columns)
            if st.button("Visualize and Handle Outliers"):
                visualize_and_handle_outliers(df_loan, outlier_feature)

            selected_histogram_feature = st.selectbox("Select a feature for the histogram:", df_loan.columns)
            if st.button("Plot Histogram"):
                plot_histogram(df_loan, selected_histogram_feature)

        elif page_selection == "Modeling":
            st.write("Data Preprocessing Options")
            scaling_option = st.selectbox("Select a Scaling Method:", ["No Scaling", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"], key="scaling_options")
            scaling_columns = st.multiselect("Select columns to scale:", df_loan.columns, key="scaling_columns")
            
            encoding_option = st.selectbox("Select an Encoding Method:", ["No Encoding", "Label Encoding"], key="encoding_options")
            encoding_columns = st.multiselect("Select columns to encode:", df_loan.columns, key="encoding_columns")

            test_size = st.number_input("Test Size (e.g., 0.2 for 80% train, 20% test)", 0.0, 1.0, 0.2, key="test_size")
            model_selection = st.selectbox("Select model to run:", ["XGBClassifier", "CatBoostClassifier"], key="model_selection")

            if st.button("Preprocess Data and Run Model"):
                preprocess_data(df_loan, scaling_option, scaling_columns, encoding_option, encoding_columns, test_size, model_selection)

    elif selected_dataset == 'df_water':
        if page_selection == "Homepage":
            homepage(df_water)
        elif page_selection == "EDA":
            explore_missing_values(df_water)
            selected_imputation_columns = st.multiselect("Select columns to impute:", df_water.columns, key="imputing_columns")
            imputation_method = st.selectbox("Select an imputation method:", ["Mean", "Median", "Mode"])
            if st.button("Impute Data"):
                impute_data(df_water, selected_imputation_columns, imputation_method)

            imbalance_feature = st.selectbox("Select a feature to check for imbalance:", df_water.columns)
            if st.button("Check Imbalance"):
                check_imbalance(df_water, imbalance_feature)

            outlier_feature = st.selectbox("Select a feature to visualize and clean for outliers:", df_water.columns)
            if st.button("Visualize and Handle Outliers"):
                visualize_and_handle_outliers(df_water, outlier_feature)

            selected_histogram_feature = st.selectbox("Select a feature for the histogram:", df_water.columns)
            if st.button("Plot Histogram"):
                plot_histogram(df_water, selected_histogram_feature)

        elif page_selection == "Modeling":
            st.write("Data Preprocessing Options")
            scaling_option = st.selectbox("Select a Scaling Method:", ["No Scaling", "Standard Scaler", "Min-Max Scaler", "Robust Scaler"], key="scaling_options")
            scaling_columns = st.multiselect("Select columns to scale:", df_water.columns, key="scaling_columns")
            
            encoding_option = st.selectbox("Select an Encoding Method:", ["No Encoding", "Label Encoding"], key="encoding_options")
            encoding_columns = st.multiselect("Select columns to encode:", df_water.columns, key="encoding_columns")

            test_size = st.number_input("Test Size (e.g., 0.2 for 80% train, 20% test)", 0.0, 1.0, 0.2, key="test_size")
            model_selection = st.selectbox("Select model to run:", ["XGBClassifier", "CatBoostClassifier"], key="model_selection")

            if st.button("Preprocess Data and Run Model"):
                preprocess_data(df_water, scaling_option, scaling_columns, encoding_option, encoding_columns, test_size, model_selection)

        pass

if __name__ == "__main__":
    main()
    