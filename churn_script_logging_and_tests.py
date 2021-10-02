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


if __name__ == "__main__":
    data = test_import(cl.import_data)
    test_eda(cl.perform_eda, data)
