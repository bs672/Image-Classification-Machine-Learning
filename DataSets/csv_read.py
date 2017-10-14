import numpy as np
import os

"""
    Visible functions:
    load_extracted_features()
    load_graph()
    load_seed()
    save_labels(str = fname): Denote the filename you want as str
"""

def unittests():
    g = load_graph()
    e = load_extracted_features()
    s = load_seed()
    assert g is not None, 'Missing Graph.csv'
    assert e is not None, 'Missing Extracted_features.csv'
    assert s is not None, 'Missing Seed.csv'
    assert g.shape == (7064950, 2), 'Graph has wrong size: {} should be (0,0)'.format(g.shape)
    assert e.shape == (10000,1084), 'Extracted_features has wrong size: {} should be (10000,1084)'.format(e.shape)
    assert s.shape == (60,2), 'Seed has wrong size: {} should be (60,2)'.format(s.shape)
    print('Passed')
# Load-csv is the general function.
# Returns a numpy array containing the values in fname
# Returns None if fname does not exist
def load_csv(fname):
    if os.path.isfile(fname):
        return np.loadtxt(fname, delimiter=',')
    else:
        print('{} not in DataSets folder.'.format(fname))
        return None

def load_graph():
    return load_csv('Graph.csv')

def load_extracted_features():
    return load_csv('Extracted_features.csv')

def load_seed():
    return load_csv('Seed.csv')

#TODO implement save labels

if __name__ == '__main__':
    print("Visible functions")
    print("load_extracted_features():   Load Extracted_features.csv as a numpy array")
    print("load_graph():                Load Graph.csv as a numpy array")
    print("load_seed():                 Load Seed.csv into a numpy array")
    print("save_labels(str = fname):    Denote the filename you want as str.")
    print("                             Doesn't need to include .csv in fname")
    print("                             Will be saved into Labels/fname")
    print("Running Unit Tests (This will take a while): ...")
    unittests()
