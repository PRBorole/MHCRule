# import libraries
import pandas as pd
import numpy as np
import time
import itertools
import pickle
from tqdm import tqdm

from rulekit.classification import RuleClassifier, ExpertRuleClassifier
from rulekit.params import Measures
from rulekit._helpers import *
from jpype.pickle import JPickler, JUnpickler

from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

import aaanalysis as aa
aa.options["verbose"] = False
aa.options["random_state"] = 42

from sklearn.model_selection import train_test_split
from src.utils import *
from src.utils_rules import *

class MHCRulePepHLA:
    """
    A class to handle MHC rule-based models using concatenated peptide and HLA sequences.
    
    Attributes:
        model (RuleClassifier): The rule-based classifier model.
        max_growing (int): Maximum number of growing iterations.
        minsupp_new (float): Minimum support for new rules.
        pruning_measure (str): Pruning measure for the rule classifier.
        voting_measure (str): Voting measure for the rule classifier.
        induction_measure (str): Induction measure for the rule classifier.
        complementary_conditions (bool): Whether to use complementary conditions in rule induction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MHCRulePepHLA class.
        
        Args:
            **kwargs: Arbitrary keyword arguments including 'max_growing', 'minsupp_new', 'pruning_measure',
                      'voting_measure', 'induction_measure', and 'complementary_conditions'.
        """
        self.model = None
        self.max_growing = kwargs.get('max_growing', None)
        self.minsupp_new = kwargs.get('minsupp_new', None)
        self.pruning_measure = kwargs.get('pruning_measure', None)
        self.voting_measure = kwargs.get('voting_measure', None)
        self.induction_measure = kwargs.get('induction_measure', None)
        self.complementary_conditions = kwargs.get('complementary_conditions', None)
        
    def fit(self, X, y):
        """
        Fit a rule-based classifier model using concatenated peptide and HLA sequences.
        
        Args:
            X (array-like): Feature matrix (concatenated peptide and HLA sequences).
            y (array-like): Target vector.
        """
        classifier = RuleClassifier(max_growing=self.max_growing,
                                    minsupp_new=self.minsupp_new,
                                    pruning_measure=self.pruning_measure,
                                    voting_measure=self.voting_measure,
                                    induction_measure=self.induction_measure,
                                    complementary_conditions=self.complementary_conditions)
        classifier.fit(X, y)
        self.model = classifier
        
    def predict(self, X, return_rules=False, str_rules=False):
        """
        Predict the target using the rule-based model.
        
        Args:
            X (array-like): Feature matrix (concatenated peptide and HLA sequences).
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices.
            
        Returns:
            array-like: Predicted target.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        classifier = self.model
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict(X), rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            return classifier.predict(X), coverage_matrix
        else:
            return classifier.predict(X)
    
    def predict_proba(self, X, return_rules=False, str_rules=False):
        """
        Predict the target probabilities using the rule-based model.
        
        Args:
            X (array-like): Feature matrix (concatenated peptide and HLA sequences).
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices.
            
        Returns:
            array-like: Predicted target probabilities.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        label = 1
        classifier = self.model
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], coverage_matrix
        else:
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)]
        
    def get_rules(self):
        """
        Get the rules from the rule-based model.
        
        Returns:
            list: List of rules from the model.
        """
        return [str(rule) for rule in self.model.model.rules]


class MHCRulePepOnly:
    """
    A class to handle MHC rule-based models using peptide-sequence only.
    
    Attributes:
        alleles (list): List of allele identifiers.
        models (dict): Dictionary to store trained models for each allele.
        max_growing (int): Maximum growing.
        minsupp_new (float): Minimum support for new rules.
        pruning_measure (str): Pruning measure for the rule classifier.
        voting_measure (str): Voting measure for the rule classifier.
        induction_measure (str): Induction measure for the rule classifier.
        complementary_conditions (bool): Whether to use complementary conditions in rule induction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MHCRulePepOnly class.
        
        Args:
            **kwargs: Arbitrary keyword arguments including 'max_growing', 'minsupp_new', 'pruning_measure',
                      'voting_measure', 'induction_measure', and 'complementary_conditions'.
        """
        self.alleles = []
        self.models = {}
        self.max_growing = kwargs.get('max_growing', None)
        self.minsupp_new = kwargs.get('minsupp_new', None)
        self.pruning_measure = kwargs.get('pruning_measure', None)
        self.voting_measure = kwargs.get('voting_measure', None)
        self.induction_measure = kwargs.get('induction_measure', None)
        self.complementary_conditions = kwargs.get('complementary_conditions', None)
    
    def fit(self, allele, X, y, retrain=False):
        """
        Fit a rule-based classifier model for a given allele.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            retrain (bool): If True, retrain the model even if it already exists.
        """
        # Check if model for this allele already exists and retraining is not requested
        if allele in self.alleles and not retrain:
            print("A trained model already exists for "+allele+", if you want to retrain, pass retrain==True")
        else:
            if allele not in self.alleles:
                self.alleles.append(allele)
            
            # Initialize and fit the rule classifier
            classifier = RuleClassifier(max_growing=self.max_growing,
                                        minsupp_new=self.minsupp_new,
                                        pruning_measure=self.pruning_measure,
                                        voting_measure=self.voting_measure,
                                        induction_measure=self.induction_measure,
                                        complementary_conditions=self.complementary_conditions)
            classifier.fit(X, y)
            self.models[allele] = classifier
        
    def predict(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target using the rule-based model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict(X), rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            return classifier.predict(X), coverage_matrix
        else:
            return classifier.predict(X)
    
    def predict_proba(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target probabilities using the rule-based model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target probabilities.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        label = 1
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix(classifier.model, X)
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], coverage_matrix
        else:
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)]
        
    def get_rules(self, allele):
        """
        Get the rules from the rule-based model for a given allele.
        
        Args:
            allele (str): Allele identifier.
        
        Returns:
            list: List of rules from the model.
        """
        return [str(rule) for rule in self.models[allele].model.rules]

    
class MHCRuleHydro:
    """
    A class to handle MHC rule-based models with hydrophobicity properties.
    
    Attributes:
        alleles (list): List of allele identifiers.
        models (dict): Dictionary to store trained models for each allele.
        scale (str): AA scale.
        max_growing (int): Maximum growing.
        minsupp_new (float): Minimum support for new rules.
        pruning_measure (str): Pruning measure for the rule classifier.
        voting_measure (str): Voting measure for the rule classifier.
        induction_measure (str): Induction measure for the rule classifier.
        complementary_conditions (bool): Whether to use complementary conditions in rule induction.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MHCRuleHydro class.
        
        Args:
            **kwargs: Arbitrary keyword arguments including 'scale', 'max_growing', 'minsupp_new', 'pruning_measure',
                      'voting_measure', 'induction_measure', and 'complementary_conditions'.
        """
        self.alleles = []
        self.models = {}
        self.scale = kwargs.get('scale', None)
        self.max_growing = kwargs.get('max_growing', None)
        self.minsupp_new = kwargs.get('minsupp_new', None)
        self.pruning_measure = kwargs.get('pruning_measure', None)
        self.voting_measure = kwargs.get('voting_measure', None)
        self.induction_measure = kwargs.get('induction_measure', None)
        self.complementary_conditions = kwargs.get('complementary_conditions', None)
        
    def fit(self, allele, X, y, retrain=False):
        """
        Fit a rule-based classifier model for a given allele.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            retrain (bool): If True, retrain the model even if it already exists.
        """
        X = get_sequence_encoding(X, self.scale)
        
        # Check if model for this allele already exists and retraining is not requested
        if allele in self.alleles and not retrain:
            print("A trained model already exists for "+allele+", if you want to retrain, pass retrain==True")
        else:
            if allele not in self.alleles:
                self.alleles.append(allele)
            
            # Initialize and fit the rule classifier
            classifier = RuleClassifier(max_growing=self.max_growing,
                                        minsupp_new=self.minsupp_new,
                                        pruning_measure=self.pruning_measure, 
                                        voting_measure=self.voting_measure,
                                        induction_measure=self.induction_measure, 
                                        complementary_conditions=self.complementary_conditions)
            classifier.fit(X, y)
            self.models[allele] = classifier
        
    def predict(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target using the rule-based model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        X = get_sequence_encoding(X, self.scale)
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix_scale(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict(X), rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix_scale(classifier.model, X)
            return classifier.predict(X), coverage_matrix
        else:
            return classifier.predict(X)
    
    def predict_proba(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target probabilities using the rule-based model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target probabilities.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        X = get_sequence_encoding(X, self.scale)
        label = 1
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix = get_ruleset_coverage_matrix_scale(classifier.model, X)
            rules_ls = [[str(classifier.model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix]
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], rules_ls
        elif return_rules and str_rules==False:
            coverage_matrix = get_ruleset_coverage_matrix_scale(classifier.model, X)
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)], coverage_matrix
        else:
            return classifier.predict_proba(X)[:, classifier.label_unique_values.index(label)]
        
    def get_rules(self, allele):
        """
        Get the rules from the rule-based model for a given allele.
        
        Args:
            allele (str): Allele identifier.
        
        Returns:
            list: List of rules from the model.
        """
        return [str(rule) for rule in self.models[allele].model.rules]


class MHCRuleHydroPep:
    """
    A class to stack two models, MHCRulePepOnly and MHCRuleHydro, using a logistic regression model.
    
    Attributes:
        alleles (list): List of allele identifiers.
        models (dict): Dictionary to store trained logistic regression models for each allele.
        scale (str): AA scale.
        mhcrulepeponly (object): Instance of MHCRulePepOnly.
        mhcrulehydro (object): Instance of MHCRuleHydro.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MHCRuleHydroPep class.
        
        Args:
            **kwargs: Arbitrary keyword arguments including 'scale', 'mhcrulepeponly', and 'mhcrulehydro'.
        """
        self.alleles = []
        self.models = {}
        self.scale = kwargs.get('scale', None)
        self.mhcrulepeponly = kwargs.get('mhcrulepeponly', None)
        self.mhcrulehydro = kwargs.get('mhcrulehydro', None)
        
    def fit(self, allele, X, y, retrain=False):
        """
        Fit a logistic regression model to stack the predictions from MHCRulePepOnly and MHCRuleHydro models.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            y (array-like): Target vector.
            retrain (bool): If True, retrain the model even if it already exists.
        """
        X_encoded = get_sequence_encoding(X, self.scale)
        
        # Check if model for this allele already exists and retraining is not requested
        if allele in self.alleles and not retrain:
            print("A trained model already exists for "+allele+", if you want to retrain, pass retrain==True")
        else:
            if allele not in self.alleles:
                self.alleles.append(allele)
                
            # Get predictions from individual models
            y_peponly_ls = self.mhcrulepeponly.predict_proba(allele, X)
            y_hydro_ls = self.mhcrulehydro.predict_proba(allele, X_encoded)

            # Create a DataFrame with true labels and predictions
            train_df = pd.DataFrame({'y_true': y,
                                     'y_peponly': y_peponly_ls,
                                     'y_hydro': y_hydro_ls})

            # Fit logistic regression model
            clf = LogisticRegression(random_state=42).fit(train_df.drop('y_true', axis=1), train_df['y_true'])
            self.models[allele] = clf
        
    def predict(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target using the stacked model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        X_encoded = get_sequence_encoding(X, self.scale)
        
        # Get predictions from individual models
        y_peponly_ls = self.mhcrulepeponly.predict_proba(allele, X)
        y_hydro_ls = self.mhcrulehydro.predict_proba(allele, X_encoded)
        
        # Create a DataFrame with predictions
        test_df = pd.DataFrame({'y_peponly': y_peponly_ls,
                                'y_hydro': y_hydro_ls})
        
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix_peponly = get_ruleset_coverage_matrix(self.mhcrulepeponly.models[allele].model, X)
            coverage_matrix_hydro = get_ruleset_coverage_matrix_scale(self.mhcrulehydro.models[allele].model, X_encoded)
            
            rules_peponly_ls = [[str(self.mhcrulepeponly.models[allele].model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix_peponly]
            rules_hydro_ls = [[str(self.mhcrulehydro.models[allele].model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix_hydro]
            return classifier.predict(test_df), rules_peponly_ls, rules_hydro_ls
        
        elif return_rules and str_rules==False:
            coverage_matrix_peponly = get_ruleset_coverage_matrix(self.mhcrulepeponly.models[allele].model, X)
            coverage_matrix_hydro = get_ruleset_coverage_matrix_scale(self.mhcrulehydro.models[allele].model, X_encoded)
            return classifier.predict(test_df), coverage_matrix_peponly, coverage_matrix_hydro
        
        else:
            return classifier.predict(test_df)
    
    def predict_proba(self, allele, X, return_rules=False, str_rules=False):
        """
        Predict the target probabilities using the stacked model.
        
        Args:
            allele (str): Allele identifier.
            X (array-like): Feature matrix.
            return_rules (bool): If True, return rule coverage matrices or rule strings.
            str_rules (bool): If True, return rule strings instead of coverage matrices, return_rules should be True.
            
        Returns:
            array-like: Predicted target probabilities.
            Optional: Rule coverage matrices or rule strings if return_rules is True.
        """
        X_encoded = get_sequence_encoding(X, self.scale)
        
        # Get predictions from individual models
        y_peponly_ls = self.mhcrulepeponly.predict_proba(allele, X)
        y_hydro_ls = self.mhcrulehydro.predict_proba(allele, X_encoded)
        
        # Create a DataFrame with predictions
        test_df = pd.DataFrame({'y_peponly': y_peponly_ls,
                                'y_hydro': y_hydro_ls})
        
        label = 1
        classifier = self.models[allele]
        
        if return_rules and str_rules:
            coverage_matrix_peponly = get_ruleset_coverage_matrix(self.mhcrulepeponly.models[allele].model, X)
            coverage_matrix_hydro = get_ruleset_coverage_matrix_scale(self.mhcrulehydro.models[allele].model, X_encoded)
            
            rules_peponly_ls = [[str(self.mhcrulepeponly.models[allele].model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix_peponly]
            rules_hydro_ls = [[str(self.mhcrulehydro.models[allele].model.rules[jdx]) for jdx, j in enumerate(i) if j == 1] for i in coverage_matrix_hydro]
            return classifier.predict_proba(test_df)[:, list(classifier.classes_).index(label)], rules_peponly_ls, rules_hydro_ls
        
        elif return_rules and str_rules==False:
            coverage_matrix_peponly = get_ruleset_coverage_matrix(self.mhcrulepeponly.models[allele].model, X)
            coverage_matrix_hydro = get_ruleset_coverage_matrix_scale(self.mhcrulehydro.models[allele].model, X_encoded)
            return classifier.predict_proba(test_df)[:, list(classifier.classes_).index(label)], coverage_matrix_peponly, coverage_matrix_hydro
        
        else:
            return classifier.predict_proba(test_df)[:, list(classifier.classes_).index(label)]
        
    def get_rules(self, allele):
        """
        Get the rules from both MHCRulePepOnly and MHCRuleHydro models for a given allele.
        
        Args:
            allele (str): Allele identifier.
        
        Returns:
            list: Combined list of rules from both models.
        """
        peponly_ls = [str(rule) for rule in self.mhcrulepeponly.models[allele].model.rules]
        hydro_ls = [str(rule) for rule in self.mhcrulehydro.models[allele].model.rules]
        return peponly_ls + hydro_ls
    
    
def fit_loop(model, allele_ls, df, test_size, retrain=False):
    '''
    Function to train model while looping through alleles list
    Args:
        model (MHCRule model): MHCRulePepOnly or MHCRuleHydro or MHCRuleHydroPep model to be train
        allele_ls (list): list of alleles for which training is to be done
        df (pd.DataFrame): training data
        test_size (float): fraction of training to be used for test (the test will split 50% for validation)
        retrain (boolean): If rules for exist for an allele, should it be retrained
        
    Returns:
        pd.DataFrame: result of training along with validation and test metrics
        model: trained model
    '''
    results_ls = []
    
    for allele in tqdm(allele_ls):
        
        result = {'allele':[allele]}
        
        hla_df = df[df['allele']==allele].reset_index(drop=True)
        result['peptide_count'] = [len(hla_df)]
        
        # split train, test
        train_df, test_df = train_test_split(hla_df, test_size=test_size, stratify=hla_df['y'].to_list())
        
        # Now, splitting test data into half to get test and validation sets
        test_df, valid_df = train_test_split(test_df, test_size=0.5)

        # get train
        X_train, y_train = get_vector_representation(train_df['peptide']), train_df['y'].to_numpy()

        # get valid
        X_valid, y_valid = get_vector_representation(valid_df['peptide']), valid_df['y'].to_numpy()
        
        # get test
        X_test, y_test = get_vector_representation(test_df['peptide']), test_df['y'].to_numpy()

        # Start the timer
        start_time = time.time()
        
        # fit model
        model.fit(allele, X_train, y_train, retrain=retrain)
        
        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time
        result['train_time'] = [elapsed_time]
        
        ### Train results
        prediction = model.predict(allele, X_train, return_rules=False)
        prediction_proba = model.predict_proba(allele, X_train, return_rules=False)

        acc_, f1_, auroc, auprc = get_metric_results(y_train, prediction, 
                                                     prediction_proba, 
                                                     metrics=['accuracy','f1','AUROC','AUPRC'])
        
        
        result['train_accuracy'], result['train_f1'] = acc_, f1_
        result['train_auroc'], result['train_auprc'] = auroc, auprc
        
        ### valid results
        prediction= model.predict(allele, X_valid, return_rules=False)
        prediction_proba= model.predict_proba(allele, X_valid, return_rules=False)

        acc_, f1_, auroc, auprc = get_metric_results(y_valid, prediction, 
                                                     prediction_proba, 
                                                     metrics=['accuracy','f1','AUROC','AUPRC'])
        
        
        result['valid_accuracy'], result['valid_f1'] = acc_, f1_
        result['valid_auroc'], result['valid_auprc'] = auroc, auprc
        
        ### test results
        prediction = model.predict(allele, X_test, return_rules=False)
        prediction_proba = model.predict_proba(allele, X_test, return_rules=False)

        acc_, f1_, auroc, auprc = get_metric_results(y_test, prediction, 
                                                     prediction_proba, 
                                                     metrics=['accuracy','f1','AUROC','AUPRC'])
        
        
        result['test_accuracy'], result['test_f1'] = acc_, f1_
        result['test_auroc'], result['test_auprc'] = auroc, auprc
        
        if isinstance(model, MHCRuleHydroPep):
            result['rule_count'] = [len(model.mhcrulepeponly.models[allele].model.rules) + len(model.mhcrulehydro.models[allele].model.rules)]
        else:
            result['rule_count'] = [len(model.models[allele].model.rules)]
        
        results_ls = results_ls + [pd.DataFrame(result)]
        
    return pd.concat(results_ls), model
