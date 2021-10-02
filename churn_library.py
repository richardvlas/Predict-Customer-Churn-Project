'''
This module contains functions to identify credit card customers that are 
most likely to churn.

author: Richard Vlas
date: October 2021
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    perform eda on df and save figures to images folder
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


if __name__ == "__main__":
    
    path_to_csv = "./data/bank_data.csv"
    path_to_eda_imgs = "./images/eda/"
    
    # import the data
    data = import_data(path_to_csv)
    
    # exploratory data analysis
    perform_eda(data, path_to_eda_imgs)
