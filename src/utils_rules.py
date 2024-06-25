import numpy as np
import pandas as pd
import re
import random
from Bio import SeqIO
from tqdm import tqdm
import time
import pickle
import itertools

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split

import aaanalysis as aa
aa.options["verbose"] = False
aa.options["random_state"] = 42

from src.utils import *

def evaluate_rule(rule_string, input_values):
    """
    Evaluates a rule string based on input values and returns the corresponding label value.

    Args:
        rule_string (str): A string representing a rule, following the format "IF condition THEN label = value".
        input_values (dict): A dictionary containing input variable names and their corresponding values.

    Returns:
        float or str: The label value if the condition is met, 'NA' if an error occurs during evaluation, or False if an exception is raised.

    Example:
        input_values = {"att2": "L", "att3": "L"}
        rule_string = 'IF att2 = {L} AND att3 = {L} THEN label = {0}'
        evaluate_rule(rule_string, input_values)  # Returns 0.0
    """
    
    # remove curly brackets
    rule_string = rule_string.replace("{", "'").replace("}", "'")
    
    # Extract the label value after "THEN"
    label_value = rule_string.split("THEN")[1].split("=")[1].strip()
    rule_string = rule_string.split("THEN")[0]
    rule_string = rule_string.split("IF")[1]
    
    # Replace variable names with their corresponding values
    for var_name, value in input_values.items():
        rule_string = rule_string.replace(var_name, "'"+str(value)+"'")
        
    # Replace logical operators and range notation with Python equivalents
    rule_string = rule_string.replace("AND", "and")
    rule_string = rule_string.replace("OR", "or")
    rule_string = rule_string.replace("=", "==")
    
    # Evaluate the expression using eval()
    try:
        result = eval(rule_string)
        if label_value.strip() == "'0'":
            label_value=0
        elif label_value.strip() == "'1'":
            label_value=1
            
        if result:
            return float(label_value)
        else:
            return "NA"
    
    except Exception as e:
        print("Error evaluating rule:", e)
        return False
    
    
    
def get_rule_cover(rule_string, input_value):
    """
    Returns whether the rule string covers an example or not

    Args:
        rule_string (str): A string representing a rule, following the format "IF condition THEN label = value".
        input_values (list): peptide as a list of amino acids

    Returns:
        int: The coverage binary 1 or 0 if the rule covers example or not

    Example:
        input_values = ['T', 'I', 'G', 'N', 'Q', 'L', 'Y', 'L', 'T']
        rule_string = 'IF att2 = {L} AND att3 = !{L} THEN label = {0}'
        get_rule_cover(rule_string, input_values)  # Returns 0 or 1
    """
    
    # remove curly brackets
    rule_string = rule_string.replace("{", "'").replace("}", "'")
    
    # Extract the label value after "THEN"
    label_value = rule_string.split("THEN")[1].split("=")[1].strip()
    rule_string = rule_string.split("THEN")[0]
    rule_string = rule_string.split("IF")[1]
    
    # Replace variable names with their corresponding values
    for idx, value in enumerate(input_value):
        rule_string = rule_string.replace('att'+str(idx+1)+' ', "'"+str(value)+"' ")
        
    # Replace logical operators and range notation with Python equivalents
    rule_string = rule_string.replace("AND", "and")
    rule_string = rule_string.replace("OR", "or")
    rule_string = rule_string.replace(" = !", "!=")
    rule_string = rule_string.replace(" = ", "==")
    
    return int(eval(rule_string))

def get_ruleset_cover(rules, input_value):
    """
    Returns whether the ruleset covers an example or not

    Args:
        rules (list): list of rules
        input_values (list): peptide as a list of amino acids

    Returns:
        coverage_ls (int list): A binary list containing if rules cover the example or not
    """
    return np.array([get_rule_cover(str(rule), input_value) for rule in rules])



def get_ruleset_coverage_matrix(ruleset, input_matrix):
    """
    Returns a binary matrix of #input rows and #rules columns with each entry indicating if a rule covers an example or not

    Args:
        ruleset (str list): ruleset of rules in string format
        input_matrix (matrix): input peptide matrix

    Returns:
        coverage_mat (int matrix): A binary matrix containing if rules cover the example or not
    """
    assert len(input_matrix.shape)==2
        
    return np.array([get_ruleset_cover(ruleset.rules, input_value) for input_value in input_matrix])


def get_rule_condition(rule):
    """
    Extract and format the conditions from a rule.
    
    Args:
        rule (str): A rule in the format 'IF conditions THEN result'.
        
    Returns:
        list: A list of formatted conditions.
    """
    # Extract the part of the rule before the 'THEN'
    conditions = rule.split(' THEN ')[0]
    
    # Remove the 'IF ' prefix
    conditions = conditions.replace('IF ', '')
    
    # Replace curly braces and other unwanted characters
    conditions = conditions.replace('{', '').replace('}', '').replace(' ', '')
    
    # Split the conditions by 'AND'
    conditions_list = conditions.split('AND')
    
    # Format the conditions as attX=Y
    formatted_conditions = [cond.replace('=', '=') for cond in conditions_list]
    
    return formatted_conditions

def get_rule_condition_scale(rule):
    """
    Extract and format the conditions from a rule for scaled data.
    
    Args:
        rule (str): A rule in the format 'IF conditions THEN result'.
        
    Returns:
        list: A list of formatted conditions with scaling considerations.
    """
    # Extract the part of the rule before the 'THEN'
    conditions = rule.split(' THEN ')[0]
    
    # Remove the 'IF ' prefix
    conditions = conditions.replace('IF ', '')
    
    # Split the conditions by 'AND'
    conditions_list = conditions.split(' AND ')
    
    # Replace '=<' with '>='
    conditions = conditions.replace('=<', '>=')
    
    # Format the conditions as attX=Y
    formatted_conditions = [cond.replace(' ', '') for cond in conditions_list]
    return formatted_conditions

def parse_condition_scale(condition):
    """
    Parse a scaled condition into attribute number and range.
    
    Args:
        condition (str): A condition in the format 'attX=(lower,upper)'.
        
    Returns:
        tuple: A tuple containing attribute number (int), lower bound (float), and upper bound (float).
    """
    # Extract the attribute and the range
    attr, value_range = condition.split('=')
    
    # Remove curly braces and split the range
    value_range = value_range.strip('()')
    lower, upper = value_range.split(',')
    
    # Remove < from lower
    lower = lower.replace('<','')
    
    # Convert the range values to floats, handle -inf
    lower = float(lower) if lower.strip() != '-inf' else float('-inf')
    upper = float(upper) if upper.strip() != 'inf' else float('inf')
    
    # Extract the attribute number
    attr_num = int(attr.replace('att', ''))
    
    return attr_num, lower, upper

def check_rule_scale(rule, vector):
    """
    Check if a vector satisfies the conditions of a scaled rule.
    
    Args:
        rule (str): A rule in the format 'IF conditions THEN result'.
        vector (list): A list representing a feature vector.
        
    Returns:
        int: 1 if the vector satisfies all conditions of the rule, otherwise 0.
    """
    # Get the conditions from the rule
    conditions = get_rule_condition_scale(rule)
    
    for condition in conditions:
        attr_num, lower, upper = parse_condition_scale(condition)
        
        # Check if the vector value at the position meets the range condition
        if not (lower <= vector[attr_num - 1] <= upper):
            return 0
    return 1

def get_ruleset_cover_scale(rules, input_value):
    """
    Returns whether the ruleset covers an example or not

    Args:
        rules (list): list of rules
        input_values (list): peptide as a list of amino acids

    Returns:
        coverage_ls (int list): A binary list containing if rules cover the example or not
    """
    return np.array([check_rule_scale(str(rule), input_value) for rule in rules])



def get_ruleset_coverage_matrix_scale(ruleset, input_matrix):
    """
    Returns a binary matrix of #input rows and #rules columns with each entry indicating if a rule covers an example or not

    Args:
        ruleset (str list): ruleset of rules in string format
        input_matrix (matrix): input peptide matrix

    Returns:
        coverage_mat (int matrix): A binary matrix containing if rules cover the example or not
    """
    assert len(input_matrix.shape)==2
        
    return np.array([get_ruleset_cover_scale(ruleset.rules, input_value) for input_value in input_matrix])
