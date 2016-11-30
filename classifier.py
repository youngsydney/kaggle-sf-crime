"""
Module for all the classifiers and the ensemble classifier
"""

import numpy as np
import scipy as sp
from tabulate import tabulate
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss

import config

class Modeler(object):
    """
    Runs the K-Fold evaluation and the prediction for test data
    """

    def __init__(self, train, test, targets, strat, features):
        self.k_fold = KFold(len(train), n_folds=config.K,
                            shuffle=True, random_state=4)
        self.cm = {}
        self.train = train
        self.test = test
        self.targets = targets # Category values for the training dataset
        self.stratification = strat
        self.fts = features
        #self.produce_baseline()
        #exit()

    def run_Kfold(self):
        """
        Run the K fold validation on the training dataset
        """
        print "Entering the Testing Phase"
        x = 1
        l_loss = 0
        h_loss = 0
        loss = 0
        mrr = 0
        top3 = 0
        count = 0
        for train_idx, test_idx in self.k_fold:
            classifier = Classifier(self.train, [], self.targets, self.stratification, self.fts)
            y_pred, y_pred_detail, y_true = classifier.classify(train_idx, test_idx)
            y_pred_detail[ ~np.isfinite(y_pred_detail) ] = 0

            ag = np.average(y_pred_detail)
            vr = np.var(y_pred_detail)
            softmax = np.vectorize(lambda x: 1/(1 + np.exp(- (x - ag) / vr)))
            softmax_hyper = np.vectorize(lambda x: (1 - np.exp(- (x - ag) / vr))/(1 + np.exp(- (x - ag) / vr)))
            y_pred_detail_log = softmax(y_pred_detail)
            y_pred_detail_hyper = softmax_hyper(y_pred_detail)
            #softmax normalization on wikipedia
            if not self.cm:
                self.cm = self.build_cm(y_true)
            self.input_cm(y_pred, y_true)
            # import pdb; pdb.set_trace()
            loss += log_loss(y_true, y_pred_detail)
            l_loss += log_loss(y_true, y_pred_detail_log)
            h_loss += log_loss(y_true, y_pred_detail_hyper)
            mrr, top3, count = self.eval_MRR(y_pred_detail, y_true, mrr, top3, count)
            x += 1
            break
        metrics, averages = self.evaluate_results(loss, l_loss, h_loss, (mrr / count), top3)
        #self.display_confusion_matrix()
        self.display_metrics(metrics)
        self.display_averages(averages)

    def run_predictions(self):
        """
        Run the classifier on the test dataset
        """
        print "Running the Predictions on the test dataset"
        classifier = Classifier(self.train, self.test, self.targets, self.stratification, self.fts)
        _, y_pred_detail = classifier.classify_predict()
        self.output_file(y_pred_detail)

    def output_file(self, y_pred_detail):
        """Output the results of the classification on the test data"""
        try:
            os.remove(config.OUT_FP)
        except:
            pass
        header = 'Id'
        for type_crime in self.stratification:
            header += ',' + str(type_crime[0])
        header += '\n'
        with open(config.OUT_FP, 'w+') as out:
            out.write(header)
            x = 0
            for prob_list in y_pred_detail:
                row = str(x)
                for prob in prob_list:
                    row += ',' + str(prob)
                row += '\n'
                out.write(row)
                x += 1

    def produce_baseline(self):
        """
        Produce the baseline results which are found by predicting the prob.
        of each label with the probability of that class label
        """
        every_list = []
        sum_total = 0
        for crime in self.stratification:
            sum_total += crime[1]
        for crime in self.stratification:
            every_list.append((float(crime[1]) / sum_total))
        try:
            os.remove(config.OUT_FP)
        except:
            pass
        header = 'Id'
        for type_crime in self.stratification:
            header += ',' + str(type_crime[0])
        header += '\n'
        with open(config.OUT_FP, 'w+') as out:
            out.write(header)
            for n in range(0, 884262):
                row = str(n)
                for value in every_list:
                    row += ',' + str(value)
                row += '\n'
                out.write(row)

    def build_cm(self, targets):
        """
        Builds the confusion matrix in preparation for the runs
        """
        unique = set(targets)
        cm = {}
        for label in unique:
            cm[label] = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
        return cm

    def input_cm(self, predicted, test_y):
        """
        Builds the confusion matrix for the given model.
        """
        for label in self.cm:
            for idx, record in enumerate(predicted):
                # assigned the correct class for the current label
                if record == test_y[idx] and record == label:
                    self.cm[label]['TP'] += 1
                # falsely assigned the label of the current class
                elif record != test_y[idx] and record == label:
                    self.cm[label]['FP'] += 1
                # didn't assign the label of the current class but should have
                elif record != test_y[idx] and test_y[idx] == label:
                    self.cm[label]['FN'] += 1
                # didn't assign the label of the current class and shouldn't
                # have
                else:
                    self.cm[label]['TN'] += 1

    def display_confusion_matrix(self):
        """
        Displays the confusion matrix for each of the targets
        """
        targets = self.cm.keys()
        print '\nConfusion Matrix by Class'
        for group in targets:
            row1 = ['-', 'Predicted', '-']
            row2 = ['Actual', group, ('not ' + group)]
            row3 = [group, self.cm[group]['TP'],
                    self.cm[group]['FN']]
            row4 = [('not ' + group), self.cm[group]['FP'],
                    self.cm[group]['TN']]
            print tabulate([row1, row2, row3, row4], tablefmt="grid")

    def eval_MRR(self, detailed, true, mrr, top3, total_count):
        """
        Calculate the Mean Reciprocal Ranks
        """
        crimes =  [x[0] for x in self.stratification]
        count = 0
        for x in detailed:
            values = dict(zip(crimes, x))
            ranked = sorted(values, key=values.get, reverse=True)
            rank = ranked.index(true[count]) + 1
            count += 1
            total_count += 1
            rr = (float(1.0) / rank)
            mrr += rr
            if rank >= 3:
                top3 += 1
        return mrr, top3, total_count

    def evaluate_results(self, loss, l_loss, h_loss, mrr, top3):
        """
        Calculates the evaluation metrics
        """
        unique = set(self.targets)
        metrics = {}
        for label in unique:
            metrics[label] = {'Precision': 0, 'Recall': 0, 'F1': 0}
        averages = {'macro': {'Precision': 0, 'Recall': 0, 'F1': 0},
                    'micro': {'Precision': 0, 'Recall': 0, 'F1': 0},
                    'LogLossLog': l_loss, 'LogLossHyp': h_loss, 'MRR': mrr,
                    'Top3': top3, 'LogLoss': loss}
        count = 0
        true_postives = 0
        false_postives = 0
        false_negatives = 0

        for label in metrics:
            count += 1
            true_postives += self.cm[label]['TP']
            false_postives += self.cm[label]['FP']
            false_negatives += self.cm[label]['FN']
            try:
                precision = float(self.cm[label]['TP']) / (self.cm[label]['TP'] + self.cm[label]['FP'])
            except:
                precision = 0
            metrics[label]['Precision'] = precision
            averages['macro']['Precision'] += precision
            try:
                recall = float(self.cm[label]['TP']) / (self.cm[label]['TP'] + self.cm[label]['FN'])
            except:
                recall = 0
            metrics[label]['Recall'] = recall
            averages['macro']['Recall'] += recall
            try:
                f1 = (2 * metrics[label]['Precision'] * metrics[label]['Recall']) / (metrics[label]['Precision'] + metrics[label]['Recall'])
            except:
                f1 = 0
            metrics[label]['F1'] = f1
            averages['macro']['F1'] += f1

        averages['macro']['Precision'] = averages['macro']['Precision'] / count
        averages['macro']['Recall'] = averages['macro']['Recall'] / count
        averages['macro']['F1'] = averages['macro']['F1'] / count

        averages['micro']['Precision'] = float(true_postives) / (true_postives + false_postives)
        averages['micro']['Recall'] = float(true_postives) / (true_postives + false_negatives)
        averages['micro']['F1'] = (2 * averages['micro']['Precision'] * averages['micro']['Recall']) / (averages['micro']['Precision'] + averages['micro']['Recall'])

        return metrics, averages

    def logloss(self, act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        return ll

    def display_averages(self, averages):
        """
        Display the micro and macro averages
        """
        print '\nEvaluation Metrics Averages'
        #row1 = ['-', 'Precision', 'Recall', 'F1', 'LogLoss', 'MRR']
        row1 = ['Micro F1', 'LogLoss', 'LogLoss - logistic', 'LogLoss - hyperbolic', 'MRR', 'Top 3']
        #row2 = ['macro', averages['macro']['Precision'],
        #        averages['macro']['Recall'], averages['macro']['F1'], '-', '-']
        #row3 = ['micro', averages['micro']['Precision'],
        #        averages['micro']['Recall'], averages['micro']['F1'], '-', '-']
        row2 = [averages['micro']['F1'], averages['LogLoss'], averages['LogLossLog'],
                averages['LogLossHyp'], averages['MRR'], averages['Top3']]
        #row4 = ['Log Loss', '-', '-', '-', averages['LogLoss'], '-']
        #row5 = ['MRR', '-', '-', '-', '-', averages['MRR']]
        #print tabulate([row1, row2, row3, row4, row5], tablefmt="grid")
        print tabulate([row1, row2], tablefmt="grid")

    def display_metrics(self, metrics):
        """
        Displays the metrics by target label
        """
        print 'Classifier ' + config.classifier + '\n'
        print '\nEvaluation Metrics by Class'
        rows = []
        row1 = ['-', 'Precision', 'Recall', 'F1', 'Num. records']
        rows.append(row1)

        #unique = set(self.targets)
        #crimes = [x[0] for x in self.stratification]
        for label in self.stratification:
            row = []
            row = [label[0], metrics[label[0]]['Precision'],
                   metrics[label[0]]['Recall'], metrics[label[0]]['F1'],
                   label[1]]
            rows.append(row)
        print tabulate(rows, tablefmt="grid")


class Classifier(object):
    """
    This module builds the Naive Bayes Classifier, and the ensemble classifier
    """

    def __init__(self, data, test, targets, stratification, fts):
        self.data = data
        self.test = test
        self.targets = targets
        self.strat = stratification
        self.features = fts

    def classify(self, train_idx, test_idx):
        """
        classify the records and return the predictions and the truth
        """
        train_y = [self.targets[index] for index in train_idx]
        test_y = [self.targets[index] for index in test_idx]

        features_indices = [self.features.index(f) for f in self.features]
        train_x = self.data[train_idx].T[features_indices].T
        test_x = self.data[test_idx].T[features_indices].T
        train_x = np.matrix(train_x)
        test_x = np.matrix(test_x)

        print "Predicting..."

        #build the model by fitting the classifier with the training
        if config.classifier != 'Ensemble':
            if config.classifier == 'DecisionTree':
                print 'Decision Tree Classifier'
                clr = DecisionTreeClassifier(max_depth=4).fit(train_x, train_y)
            elif config.classifier == 'Log':
                print 'Logistic Regression'
                clr = LogisticRegression().fit(train_x, train_y)
            elif config.classifier == 'RF':
                print 'Random Forest'
                clr = RandomForestClassifier().fit(train_x, train_y)
            elif config.classifier == 'AB':
                print 'AdaBoost'
                clr = AdaBoostClassifier().fit(train_x, train_y)
            # Classifier -- run classification
            predicted = clr.predict(test_x)
            predicted_detail = clr.predict_proba(test_x)
            return predicted, predicted_detail, test_y

    def classify_predict(self):
        """
        train the classifier on the training data and then predict on test
        """
        train_y = self.targets

        print config.features
        features_indices = [config.features.index(f) for f in config.features]
        train_x = self.data.T[features_indices].T
        test_x = self.test.T[features_indices].T
        train_x = np.matrix(train_x)
        test_x = np.matrix(test_x)

        #build the model by fitting the classifier with the training
        if config.classifier != 'Ensemble':
            if config.classifier == 'DecisionTree':
                print 'Decision Tree Classifier'
                clr = DecisionTreeClassifier(max_depth=4).fit(train_x, train_y)
            elif config.classifier == 'Log':
                print 'Logistic Regression'
                clr = LogisticRegression().fit(train_x, train_y)
            elif config.classifier == 'RF':
                print 'Random Forest'
                clr = RandomForestClassifier().fit(train_x, train_y)
            elif config.classifier == 'AB':
                print 'AdaBoost'
                clr = AdaBoostClassifier().fit(train_x, train_y)
            # Classifier -- run classification
            predicted = clr.predict(test_x)
            predicted_detail = clr.predict_proba(test_x)
            return predicted, predicted_detail


    def compile_ensemble(self, lg, dt3):
        predicted_detail = lg
        predicted = []
        crimes = [x[0] for x in self.strat]

        for record_idx, record in enumerate(lg):
            max_v = 0
            max_idx = 0
            for idx, prob in enumerate(record):
                ave = (float(dt3[record_idx][idx]) + float(prob)) / 2
                predicted_detail[record_idx][idx] = ave
                if ave >= max_v:
                    max_v = ave
                    max_idx = idx
            predicted.append(crimes[max_idx])

        return predicted, predicted_detail
