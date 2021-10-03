'''
This module contains functions to identify credit card customers that are
most likely to churn.

author: Richard Vlas
date: October 2021
'''
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    Return a dataframe for the csv found at pth

    input:
            pth: a path to the csv

    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def perform_eda(df, output_pth):
    '''
    Perform eda on df and save figures to images folder

    input:
            df: pandas dataframe
            output_pth: string of path to image folder

    output:
            None
    '''
    y_column = "Churn"

    cat_columns = [
        'Marital_Status',
        'Education_Level'
    ]

    quant_columns = [
        'Customer_Age',
        'Total_Trans_Amt',
        'Total_Trans_Ct'
    ]

    for col in cat_columns:
        unique_values = df[col].value_counts('normalize')
        x_pos = range(len(unique_values))
        plt.figure(figsize=(20, 10))
        plt.bar(x_pos, unique_values)
        plt.xticks(x_pos, unique_values.index, rotation=90)
        plt.ylabel("Relative Frequncy")
        plt.title(f"Bar Chart of {col}")
        plt.savefig(os.path.join(output_pth, col), bbox_inches="tight")
        plt.close()

    for col in quant_columns:
        plt.figure(figsize=(20, 10))
        plt.hist(df[col])
        plt.ylabel("Count")
        plt.title(f"Histogram of {col}")
        plt.savefig(os.path.join(output_pth, col), bbox_inches="tight")
        plt.close()

    df[y_column] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    plt.figure(figsize=(20, 10))
    plt.hist(df[y_column])
    plt.ylabel("Count")
    plt.title("Histogram of Churn")
    plt.savefig(os.path.join(output_pth, "Churn"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(15, 10))
    sns.heatmap(df.corr(), annot=False,)
    plt.savefig(os.path.join(output_pth, "Correlation"), bbox_inches="tight")
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        new_lst = []
        new_group = df.groupby(col).mean()[response]

        for val in df[col]:
            new_lst.append(new_group.loc[val])

        new_col_name = col + '_' + response
        df[new_col_name] = new_lst

    return df


def perform_feature_engineering(df, response):
    '''
    Perform feature_engineering and return train and test data subsets.
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    y = df['Churn']
    X = pd.DataFrame()
    df = encoder_helper(df, cat_columns, response)
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores report as image
    in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rfc_results.png', bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/lrc_results.png', bbox_inches="tight")
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in output_pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Store figure
    plt.savefig(output_pth, bbox_inches="tight")

    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # instantiate models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # set grid search parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # train models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # predict using rfc model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predict using lrc model
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # plot and store roc curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/roc_curve.png", bbox_inches="tight")
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # store model evaluation metrics
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # plot and store feature importances
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        "./images/results/feature_importances.png")


if __name__ == "__main__":

    PATH_TO_CSV = "./data/bank_data.csv"
    PATH_TO_EDA_IMGS = "./images/eda/"

    # import the data
    data = import_data(PATH_TO_CSV)

    # exploratory data analysis
    perform_eda(data, PATH_TO_EDA_IMGS)

    # split data into training and testing subset
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        data, 'Churn')

    # Train and store models as well as results
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
