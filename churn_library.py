import pandas as pd


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


if __name__ == "__main__":
    # import the data
    data = import_data("./data/bank_data.csv")
    
