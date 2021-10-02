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


if __name__ == "__main__":
	data = test_import(cl.import_data)
