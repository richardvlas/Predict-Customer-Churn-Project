'''
This module contains testing functions and logging to verify if the churn_library
module works properly.

author: Richard Vlas
date: October 2021
'''
import os
import glob
import logging
import churn_library as cl

logging.basicConfig(
	filename='./logs/churn_library.log',
	level=logging.INFO,
	filemode='w',
	format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test import_data function
    '''
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_data function: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data function: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data function: The file doesn't appear to have rows or columns")
        raise err

    return df


def test_eda(perform_eda, df):
	'''
	test perform_eda function
	'''
	path_to_eda_imgs = './images/eda/'
	perform_eda(df, path_to_eda_imgs)

	try:
		img_list = [os.path.basename(file_pth) for file_pth in glob.glob(path_to_eda_imgs + "*png")]
		actual = len(img_list)
		expected = 7
		assert actual == expected
		logging.info("Testing perform_eda function: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_eda function: The perform_eda function did not create correct number of images")
		raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test test_encoder_helper function
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, cat_columns, 'Churn')

    try:
        for col in cat_columns:
            assert col in df.columns
        logging.info("Testing encoder_helper function: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper function: The dataframe did not contain the transformed features")
        raise err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering function: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering function: The train and test data objects appear to not contain data")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models function
    '''
    path_to_models = './models/'
    path_to_results = './images/results/'
    
    #train_models(X_train, X_test, y_train, y_test)

    try:
        model_lst = [os.path.basename(file_pth) for file_pth in glob.glob(path_to_models + '*pkl')]
        actual = len(model_lst)
        expected = 2
        assert actual == expected
    except AssertionError as err:
        logging.error("Testing train_models function: The function did not create correct number of models")
        raise err

    try:
        img_lst = [os.path.basename(file_pth) for file_pth in glob.glob(path_to_results + '*png')]
        actual = len(img_lst)
        expected = 4
        assert actual == expected
        logging.info("Testing train_models function: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models function: The function did not create correct number of images")
        raise err


if __name__ == "__main__":
    data = test_import(cl.import_data)
    test_eda(cl.perform_eda, data)
    data = test_encoder_helper(cl.encoder_helper, data)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, data)
    test_train_models(cl.train_models, X_train, X_test, y_train, y_test)
