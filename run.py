"""
Main script which runs the SF Crime classificatoin program
"""

import time

import load
import feature
import classifier

def main():
    """
    Controls the flow
    """
    start = time.time()

    # load the datasets
    loader = load.DataLoader()
    # return the training and the testing datasets
    train, test, strat, streets = loader.load_data()

    featurer = feature.Feature(train, test, streets)
    train, test, targets_tr, features = featurer.format()

    runner = classifier.Modeler(train, test, targets_tr, strat, features)
    runner.run_predictions()
    runner.run_Kfold()

    end = time.time()
    print "Total Time: " + str(end-start)

if __name__ == "__main__":
    """
    Runs the control.
    """
    main()
