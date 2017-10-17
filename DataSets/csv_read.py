import numpy as np
import os
import pandas as pd

"""
    Visible functions:
    load_extracted_features()
    load_graph()
    load_seed()
    save_labels(str = fname, vals): Denote the filename you want as str
"""

def unittests():
    g = load_graph()
    g_s = load_graph(shape_match = True)
    e = load_extracted_features()
    s = load_seed()
    ss= load_seed_similarity()
    assert g is not None, 'Missing Graph.csv'
    assert g_s is not None, 'Missing Graph.csv, shape matched'
    assert e is not None, 'Missing Extracted_features.csv'
    assert s is not None, 'Missing Seed.csv'
    assert g.shape == (7064950, 2), 'Graph has wrong size: {} should be (7064949, 2)'.format(g.shape)
    assert g_s.shape == (6000, 6000), 'G_S has wrong size: {} should be (6000,6000)'.format(g_s.shape)
    assert check_symmetric(g_s), 'G_S is not symmetric'
    assert e.shape == (10000,1084), 'Extracted_features has wrong size: {} should be (10000,1084)'.format(e.shape)
    assert s.shape == (60,2), 'Seed has wrong size: {} should be (60,2)'.format(s.shape)
    assert ss.shape == (6000,10), 'Seed Similarity has wrong size: {} should be (6000,10)'.format(ss.shape)
    print('Passed')

# Load-csv is the general function.
# Returns a numpy array containing the values in fname
# Returns None if fname does not exist
def load_csv(fname):
    if os.path.isfile(fname):
        return pd.read_csv(fname, delimiter=',', header=None).values
    elif os.path.isfile(os.path.join('DataSets',fname)):
        return pd.read_csv(os.path.join('DataSets',fname), delimiter=',',header=None).values
    else:
        print('{} not in DataSets folder.'.format(fname))
        return None

# def save_csv(fname, values):

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def load_graph(fname='Graph', shape_match=False):
    if shape_match:
        preload = load_csv(fname+'_Matrix.csv')
        if preload is not None:
            # print("Average Similarity : {}".format((preload.sum())/(6000.0**2)))
            return preload
        else:
            # Outputs a (6000, 6000) data sets with the similarity values
            # Note that this is 2.5 times the regular size of G
            g = np.array(load_csv(fname+'.csv'), dtype='int')
            lower, upper = min(min(i, j) for i, j in g), max(max(i, j) for i, j in g)
            # print("Average Similarity should be : {}".format((7064950+6000)/(6000.0**2)))
            print("Lower and Upper index {}".format((lower,upper)))
            size = upper - lower + 1
            print("Size {}".format(size))
            output = np.identity(size, dtype='int')
            # Identity used because we should consider a point to be similar to itself
            for edge in g:
                output[edge[0]-lower, edge[1]-lower] = 1
            print("Saving new similarity values as a matrix: {}_Matrix.csv".format(fname))
            # print("Average Similarity is : {}".format(output.sum()/(6000.0**2)))
            np.savetxt(fname+'_Matrix.csv',  np.asarray(output), delimiter=",", fmt='%d')
            return output
    else:
        return np.array(load_csv(fname+'.csv'), dtype='int')

def load_extracted_features():
    return load_csv('Extracted_features.csv')

def load_seed():
    return np.array(load_csv('Seed.csv'), dtype='int')

# loads (or creates) a .csv for similarities to all known labels as a [10] element array per row
# e.g [0] : [sim to 0s, sim to 1s,.... sim to 9s]
def load_seed_similarity(fname='Seed_Similarity'):
    preload = None#load_csv(fname+'.csv')
    if preload is not None:
        return preload
    else:
        s = load_seed()
        g = load_graph(shape_match=True)
        n = g.shape[0] # g.shape should be (n,n)
        output = np.zeros((n, 10), dtype='int')
        for i in range(n): # iterate though all rows in g
            for j, val in s: # iterate through all seeds
                # increase count for output[i,val] is i and j are similar
                output[i,val] += g[i,j]
        # Now to check how accurate this is for a metric
        print("Checking how much sense similarity makes")
        issues = 0
        for j, val in s:

            print('Seed {} is a {}. Similarity to seeds of same value : {}'.format(j, val, output[j,val]))
            print('Seed {} is similar to {} seeds'.format(j, np.sum(output[j])))
            print('Seed {} values : {}'.format(j,output[j]))
            if not (np.max(output[j]) == output[j,val]):
                issues += 1
                print("More Similar to a different value")
        print("A seed is more similar to other value seeds {} of them time".format(issues/60.))
        np.savetxt(fname+'.csv',  np.asarray(output), delimiter=",", fmt='%d')
        return output

#TODO implement save labels

if __name__ == '__main__':
    print("Visible functions")
    print("load_extracted_features()    : Load Extracted_features.csv as a numpy array")
    print("load_graph(shape_match=False): Load Graph.csv as a numpy array")
    print("                               shape_match will make it a 6000x6000 matrix")
    print("load_seed()                  : Load Seed.csv into a numpy array")
    print("load_seed_similarity()       : Load Seed_Similarity.csv into a numpy array")
    print("save_labels(fname,vals)      : Denote the filename you want as str.")
    print("                               Doesn't need to include .csv in fname")
    print("                               Will be saved into Labels/fname")
    print("Running Unit Tests (This will take a while): ...")
    unittests()
