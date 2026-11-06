import pandas as pd
import numpy as np



# At grading time, I will call the below function with a dataframe of new potential customers
# The dataframe I pass in will have the same columns as customer_data.csv, but will not include any of the customers in customer_data.csv
# You will not be given any of the historical customer data or claims data at grading time, 
# so you will need to save any models you train in model_dev.py and load them here.

def set_premium(customer_data: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a dataframe of data on potential customers, identifies which customers to take on, and sets premiums for them.
    This function may load a saved model (previously trained using supervised learning) to save time.
    :param customer_data: a dataframe with the same columns as customer_data.csv
    :return: a DataFrame of the same lenght as customer_data, with the following columns:
            p_claim:  your estimated probability that each customer in customer_data will make a claim
            desired_customer:  a boolean column indicating which customers you want to take on
            premium:    the amount of the premium to charge each customer
    """