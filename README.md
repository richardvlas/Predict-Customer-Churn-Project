# Predict-Customer-Churn-Project
This project implements an ML model to identify credit card customers that are most likely to churn. The code is structured as a Python package and follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). To help improve the quality of the code, [autopep8](https://pypi.org/project/autopep8/) and [pylint](https://pypi.org/project/pylint/) were used to format the code.

The package can be either imported or used to run from command-line interface (CLI).

The dataset used to train the ML model is available from Kaggle and can be accessed [here](https://www.kaggle.com/sakshigoyal7/credit-card-customers/code).

## Project Structure
The project includes the following files and folders:

- `README.md` - A markdown file giving an overview of the project and explaining the project structure
- `churn_library.py` - Contains completes the process for solving the data science process
- `churn_script_logging_and_tests.py` - Contains test and logging functions
- `images` - folder with eda and result images
- `logs` - folder with logging information stored after testing the module
- `models` - folder with ML model files

## Running, testing and logging

Running the code below in the terminal starts a pipeline that contains data import, eda, feature engineering and model training
```python
python churn_library.py
```

Running the code below in the terminal should test each of the functions and provide any errors to a file stored in the `logs` folder:

```python
python churn_script_logging_and_tests.py
```

## Code formatting
Pylint scores the file and informs of any non-consistent 
