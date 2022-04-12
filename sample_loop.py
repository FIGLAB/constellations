import numpy as np
import os
import pickle
import gzip
import constellation
import features
from os import listdir
from os.path import isfile, join
import time

def load_data(fpgz):
    # Load
    f = gzip.open(fpgz, 'rb')
    ROOMS = pickle.load(f)
    return ROOMS

def one_hot_to_zero_index(one_hot):
    res =  np.argmax(one_hot, axis=1)
    res = res.reshape(-1,1)
    res = [val[0] for val in res]
    return res

# Start Timer
t0 = time.time()

Y_test = None
Labels = None
datasets = sorted([f for f in listdir('data/') if isfile(join('data/', f)) and f.endswith('.pklz')])

# Go through each item in the dataset
for dataset in datasets:

    # Organize Dataset by day
    all_days = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']

    # Example methods: Cross-Val Per Day
    methods = ['in-room', 'all-rooms', 'all-but-in-room']
    NORMS = dict()
    
    # Perform analaysis per dataset, per day or depending on desired method
    for method in methods:
        et = time.time()
        for day in all_days:
            testing_subset = [day]
            training_subset = []
            for other_day in all_days:
                if (other_day != day):
                    training_subset.append(other_day)
            # Load
            room = dataset.replace(".pklz", "")
            room = features.remap_location(room)
            room_dataset = load_data('data/%s' % (dataset))

            # Assemble
            identifier = "%s-%s-%s" % (room, method, day)
            if (os.path.exists('results/%s.txt' % (identifier))):
                print("Session %s-%s-%s exists... Overwriting..." % (room, method, day))
                f = open('results/%s.txt' % (identifier), 'w').write("")
                f.close()
                continue

            X_train, Y_train, Labels = constellation.create_subset(training_subset, room_dataset, method=method, room=room, feature_type='flat')
            X_test, Y_test, Labels = constellation.create_subset(testing_subset, room_dataset, method=method, room=room, feature_type='flat')

            Y_train = one_hot_to_zero_index(Y_train)
            Y_test = one_hot_to_zero_index(Y_test)

            ####################
            # Learn
            # Here's an example using a Random Forest
            ####################
            # Commented out for now
            # forest = NormedRandomForest()
            # ident = "%s-%s" % (room,day)
            # forest.train(X_train, Y_train)

            # Obtain Results
            log_entry = "%s\t%s\t%s\t%s-%s-%s\tn=%d" % (features.reverse_map_location(room), method, day, features.reverse_map_location(room), method, day, X_train.shape[1])

            # Log Room Accuracy
            print(log_entry)

        total = time.time() - et
        print("===== Execution time for %s-%s: %ds =====" % (dataset.replace(".pklz", ""),method,total))

# Stop Timer
t1 = time.time()

# Execution Time
total = t1-t0
print("Execution time: %d" % total)
