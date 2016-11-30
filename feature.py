"""
This is a module which supports the building and selection of features
"""

import numpy as np
import tqdm
import config

class Feature(object):
    """
    Handles feature extraction, selection, and formatting
    """

    def __init__(self, train, test, strts):
        self.train = train
        self.test = test
        self.data = []
        self.features = []
        self.streets = strts
        self.targets = []


    def extract_date_attributes(self):
        """
        extract the month and the hour
        """
        hours_cat = {'early_am': ['04', '05', '06', '07'],
                     'late_am': ['08', '09', '10', '11'],
                     'afternoon': ['12', '13', '14', '15'],
                     'evening': ['16', '17', '18', '19', '20'],
                     'night': ['21', '22', '23', '00', '01', '02', '03']}

        for record in self.train:
            hour = record['Dates'][11:13]
            for category in hours_cat:
                if hour in hours_cat[category]:
                    record[category] = 1
                else:
                    record[category] = 0
        for record in self.test:
            hour = record['Dates'][11:13]
            for category in hours_cat:
                if hour in hours_cat[category]:
                    record[category] = 1
                else:
                    record[category] = 0


    def fill_matrix(self):
        """
        flip the values in the matrix based on the record attributes
        """
        hours_category = {'04': 'early_am', '05': 'early_am', '06': 'early_am',
                          '07': 'early_am', '08': 'late_am', '09': 'late_am',
                          '10': 'late_am', '11': 'late_am', '12': 'afternoon',
                          '13': 'afternoon', '14': 'afternoon',
                          '15': 'afternoon', '16': 'evening', '17': 'evening',
                          '18': 'evening', '19': 'evening', '20': 'evening',
                          '21': 'night', '22': 'night', '23': 'night',
                          '00': 'night', '01': 'night', '02': 'night',
                          '03': 'night'}
        index = 0
        print "Loading training into dataframe..."
        for record in tqdm.tqdm(self.train):
            street_values = record['Address'].split()
            for street in street_values:
                if street in self.streets:
                    self.matrix_tr[index][self.features.index(street)] = 1
            self.targets.append(record['Category'])
            hour = record['Dates'][11:13]
            self.matrix_tr[index][self.features.index(hours_category[hour])] = 1
            self.matrix_tr[index][self.features.index(record['PdDistrict'])] = 1
            self.matrix_tr[index][self.features.index(record['DayOfWeek'])] = 1
            index += 1

        index = 0
        print "Loading testing into dataframe..."
        for record in tqdm.tqdm(self.test):
            street_values = record['Address'].split()
            for street in street_values:
                if street in self.streets:
                    self.matrix_te[index][self.features.index(street)] = 1
            hour = record['Dates'][11:13]
            self.matrix_te[index][self.features.index(hours_category[hour])] = 1
            self.matrix_te[index][self.features.index(record['PdDistrict'])] = 1
            self.matrix_te[index][self.features.index(record['DayOfWeek'])] = 1
            index += 1

    def format(self):
        """
        Format all the features into a pandas DataFrame
        """
        # find out how many features total I will have
        self.features.extend(['early_am', 'late_am', 'afternoon',
                              'evening', 'night'])
        self.features.extend(self.streets)

        category_options = {}
        for category in config.categorical:
            values = [x[category] for x in self.train]
            category_options[category] = set(values)
            self.features.extend(set(values))

        # create the two matrices of size of records by number of features
        self.matrix_tr = np.zeros((len(self.train), len(self.features)))
        self.matrix_te = np.zeros((len(self.test), len(self.features)))

        # fill matrix
        self.fill_matrix()

        config.features = self.features

        return self.matrix_tr, self.matrix_te, self.targets, self.features
