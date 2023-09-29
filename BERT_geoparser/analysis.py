import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from BERT_geoparser.utils import flatten


class Results:

    def __init__(self, y_true, y_pred):
        self.y_true = flatten(y_true)
        self.y_pred = flatten(y_pred)
        self.cats = set(self.y_true)

    def categorical_accuracy(self, category)->float:
        """Returns the accuracy of the results on a given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess accuracy on.
        returns
        -------
        cat_accuracy : float
            The accuracy (i.e. true/total) of the predictions on the given 
            category or sub-category.
        """
        cat_correct = []
        for yt, yp in zip(self.y_true, self.y_pred):
            if (yt == yp) and (category in yt):
                cat_correct.append(True)
            elif category in yt:
                cat_correct.append(False)
        cat_accuracy = sum(cat_correct)/len(cat_correct)
        return cat_accuracy

    def categorical_TP(self, category:str)->int:
        """The number of true positives in the given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess accuracy on.
        returns
        -------
        cat_TP : int
            The number of times the specified category was correctlyt guessed.
        """
        cat_TP = 0 
        for yt, yp in zip(self.y_true, self.y_pred):
            if (category in yp) and (category in yt):
                cat_TP += 1
        return cat_TP

    def categorical_TN(self, category:str)->int:
        """The number of true negatives in the given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess accuracy on.
        returns
        -------
        cat_TN : int
            The number of times the specified category was correctly missed.
        """
        cat_TN = 0 
        for yt, yp in zip(self.y_true, self.y_pred):
            if (category not in yp) and (category not in yt):
                cat_TN += 1
        return cat_TN
    
    def categorical_FP(self, category:str)->int:
        """The number of false positives in the given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess accuracy on.
        returns
        -------
        cat_FP : int
            The number of times the specified category was incorrectly guessed.
        """
        cat_FP = 0 
        for yt, yp in zip(self.y_true, self.y_pred):
            if (category in yp) and (category not in yt):
                cat_FP += 1
        return cat_FP
    
    def categorical_FN(self, category:str)->int:
        """The number of true positives in the given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess accuracy on.
        returns
        -------
        cat_FN : int
            The number of times the specified category was incorrectly missed.
        """
        cat_FN = 0 
        for yt, yp in zip(self.y_true, self.y_pred):
            if (category not in yp) and (category in yt):
                cat_FN += 1
        return cat_FN

    def categorical_recall(self, category:str)->float:
        """The model's recall on a given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess recall on.
        returns
        -------
        cat_recall : float
            Recall on the given category.
        """
        cat_TP = self.categorical_TP(category)
        cat_FN = self.categorical_FN(category)
        cat_recall = cat_TP/(cat_FN + cat_TP)
        return cat_recall
    
    def categorical_precision(self, category:str)->float:
        """The model's precision on a given category.
        parameters
        ----------
        category : str
            The category (e.g. 'geo') or sub-category (e.g. 'B-geo' or 'I-geo') 
            to assess precision on.
        returns
        -------
        cat_precision : float
            Precision on the given category.
        """
        cat_TP = self.categorical_TP(category)
        cat_FP = self.categorical_FP(category)
        cat_precision = cat_TP/(cat_TP + cat_FP)
        return cat_precision
    
    def macro_average_precision(self)->float:
        """Macro averaged precision across all categories.
        """
        cat_precision = [self.categorical_precision(cat) for cat in self.cats]
        macro_average = sum(cat_precision)/len(self.cats)
        return macro_average
    
    def macro_average_recall(self)->float:
        """Macro averaged recall across all categories.
        """ 
        cat_recall = [self.categorical_recall(cat) for cat in self.cats]
        macro_average = sum(cat_recall)/len(self.cats)
        return macro_average
    
    def micro_average_precision(self)->float:
        """Micro averaged precision across all categories.
        """
        cat_TP = [self.categorical_TP(cat) for cat in self.cats]
        cat_FP = [self.categorical_FP(cat) for cat in self.cats]
        precision = sum(cat_TP)/(sum(cat_FP) + sum(cat_TP))
        return precision
    
    def micro_average_recall(self)->float:
        """Micro averaged recall across all categories.
        """
        cat_TP = [self.categorical_TP(cat) for cat in self.cats]
        cat_FN = [self.categorical_FN(cat) for cat in self.cats]
        recall = sum(cat_TP)/(sum(cat_FN) + sum(cat_TP))
        return recall
    
    def micro_average_F1(self)->float:
        """Micro averaged F1 score across all categories.
        """
        mu_p = self.micro_average_precision()
        mu_r = self.micro_average_recall()
        F1 = 2*((mu_p*mu_r)/(mu_p + mu_r))
        return F1
    
    def macro_average_F1(self)->float:
        """Macro averaged F1 score across all categories.
        """
        nu_p = self.macro_average_precision()
        nu_r = self.macro_average_recall()
        F1 = 2*((nu_p*nu_r)/(nu_p + nu_r))
        return F1


