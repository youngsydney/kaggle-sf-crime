"""
This is a module which supports the loading of the SF crime datasets.
"""

import csv
import tqdm

import config

class DataLoader(object):
    """
    Loads the data from the specified crime data file
    """

    def __init__(self):
        self.train = []
        self.test = []
        self.stratification = []

    def load_data(self):
        """
        Load the train dataset
        """
        with open(config.TRAIN_FP, 'r') as in_file:
            helper_dict = {}
            breakdown = {}
            streets = {}
            print "Loading the training dataset csv file into reader..."
            reader = csv.reader(in_file, delimiter=',', skipinitialspace=True)
            rows = list(reader)
            print "Adding the training dataset into memory..."
            for number in tqdm.tqdm(range(len(rows))):
                row = rows[number]
                # if this is the header line, grab the data field names
                if 'DayOfWeek' in row:
                    for idx, name in enumerate(row):
                        if name == 'Category':
                            category_index = idx
                        elif name == 'Address':
                            address_index = idx
                        # create a helper dict of the name of the field and
                        # the order in the list for help with data loading
                        helper_dict[str(idx)] = name
                # if line is not header then add the data to the correct lists
                else:
                    record = {}
                    # https://www.kaggle.com/eyecjay/sf-crime/vehicle-thefts-or-jerry-rice-jubilation
                    # stoppped reporting vechicle recoveries after 2006
                    if 'VEHICLE, RECOVERED' in row:
                        pass
                    else:
                        for idx, value in enumerate(row):
                            record[helper_dict[str(idx)]] = value
                            if idx == category_index:
                                if value in breakdown:
                                    breakdown[value] += 1
                                else:
                                    breakdown[value] = 1
                            elif idx == address_index:
                                streets_record = value.split()
                                for word in streets_record:
                                    if word.isupper() and len(word) > 2 and word in streets:
                                        streets[word] += 1
                                    elif word.isupper() and len(word) > 2:
                                        streets[word] = 1
                        self.train.append(record)
        self.stratification = breakdown.items()
        self.stratification.sort(key=lambda tup: tup[1], reverse=True)

        streets_thresh = [k for (k,v) in streets.items() if v > config.str_min]

        # also load the test dataset if the config file requests
        self.load_testing()

        return self.train, self.test, self.stratification, streets_thresh

    def load_testing(self):
        """
        Load the test dataset
        """
        with open(config.TEST_FP, 'r') as in_file:
            helper_dict = {}
            print "Loading the testing dataset csv file into reader..."
            reader = csv.reader(in_file, delimiter=',', skipinitialspace=True)
            rows = list(reader)
            print "Adding the testing dataset into memory..."
            for number in tqdm.tqdm(range(len(rows))):
                row = rows[number]
                # if this is the header line, grab the data field names
                if 'DayOfWeek' in row:
                    for idx, name in enumerate(row):
                        # create a helper dict of the name of the field and
                        # the order in the list for help with data loading
                        helper_dict[str(idx)] = name
                # if line is not header then add the data to the correct lists
                else:
                    record = {}
                    for idx, value in enumerate(row):
                        record[helper_dict[str(idx)]] = value
                    self.test.append(record)
