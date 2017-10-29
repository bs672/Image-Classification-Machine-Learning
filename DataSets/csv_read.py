import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

"""
    Quick CSV read function for data
    Allows reads of preprocessed data, e.g PCA, SpectralEmbedding
    For analysis of variance, you should run PCA on data in a separate script
    To use this module in the main folder, get up an __init__.py file or use:

    base = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(base,'DataSets'))

    To set up the path
"""

def unittests():
    fnames = ['Seed_Similarity.csv', 'Seed.csv', 'Seed_Matrix.csv', 'Graph.csv', 'Graph_Matrix.csv',
                'Extracted_features.csv', 'SpectralEmbedding.csv']
    g = load_graph()
    g_s = load_graph(shape_match = True)
    e = load_extracted_features()
    s = load_seed()
    s_m = load_seed_matrix()
    ss= load_seed_similarity()
    e_seeds = load_extracted_features(onlyseeds=True)
    e_pca = load_extracted_features_PCA()
    e_seeds_pca = load_extracted_features_PCA(onlyseeds=True)
    spec = load_spectral_embedding()
    cca_f = load_CCA_features(k = 8)
    cca_f_seed = load_CCA_features(k = 8, onlyseeds=True)
    g_dist = load_graph(shape_match=True, g_type='dist')
    s_dist = load_spectral_embedding(k=10, g_type='dist')
    g_merged = load_merged_graph_matrix()
    s_merged = load_spectral_embedding(k=10, g_type='merged')
    for f in fnames:
        assert os.path.isfile(f), "{} should be a file".format(f)
    assert g.shape == (7064950, 2), 'Graph has wrong size: {} should be (7064950, 2)'.format(g.shape)
    assert g_s.shape == (6000, 6000), 'Graph Similarity Matrix has wrong size: {} should be (6000,6000)'.format(g_s.shape)
    assert s_m.shape == (60,60), 'Seed Matrix has wrong size: {} should be (60,60)'.format(s_m.shape)
    assert check_symmetric(g_s), 'Graph Similarity Matrix is not symmetric'
    assert check_symmetric(s_m), 'Seed Matrix is not symmetric'
    assert e.shape == (10000,1084), 'Extracted_features has wrong size: {} should be (10000,1084)'.format(e.shape)
    assert s.shape == (60,2), 'Seed has wrong size: {} should be (60,2)'.format(s.shape)
    assert ss.shape == (6000,10), 'Seed Similarity has wrong size: {} should be (6000,10)'.format(ss.shape)
    assert e_seeds.shape == (60,1084), 'Extracted_features of seeds has wrong size: {} should be (60,1084)'.format(e_seeds.shape)
    assert e_pca.shape == e.shape, 'Extracted_features_PCA1084 should match Extracted_features'
    assert e_seeds_pca.shape == e_seeds.shape, '{} should be {}'.format(e_seeds_pca.shape, e_seeds.shape)
    assert cca_f.shape == (10000,8)
    assert cca_f_seed.shape == (60,8)
    assert g_dist.shape == (10000,10000)
    assert s_dist.shape == (10000,10)
    assert g_merged.shape == (6000,6000)
    assert s_merged.shape == (6000, 10)
    # for i in range(1,101):
    #     temp = load_extracted_features_PCA(k=i)
    #     temp_seeds = load_extracted_features_PCA(k=i, onlyseeds=True)
    #     for s_i in range(s.shape[0]):
    #         assert np.allclose(temp_seeds[s_i], temp[s[s_i][0]-1], atol=0.0001), '{} vs {}'.format(temp_seeds[s_i], temp[s[s_i][0]-1])

    for i in range(60):
        for j in range(60):
            if s_m[i,j]: assert s[i][1] == s[j][1]
    # assert spec.shape == g_s.shape, 'SpectralEmbedding should match Graph Matrix'

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

def load_graph(fname='Graph', shape_match=False, g_type='adj'):
    if shape_match:
        if g_type == 'dist':
            fname = fname + '_Dist'
        preload = load_csv(fname+'_Matrix.csv')
        if preload is not None:
            return preload
        else:
            if g_type == 'dist':
                # Creates a distance matrix, ala lecture notes
                X = load_extracted_features_PCA(k=10)
                size = X.shape[0]
                output = np.zeros((size,size))
                def A(x1,x2):
                    return np.exp(-(np.linalg.norm(x1-x2)**2))
                for i in range(size):
                    for j in range(size):
                        output[i,j] = A(X[i],X[j])
                print("Saving new distance values as a matrix: {}_Matrix.csv".format(fname))
                np.savetxt(fname+'_Matrix.csv',  np.asarray(output), delimiter=",")
            else:
                g = np.array(load_csv(fname+'.csv'), dtype='int')
                lower, upper = min(min(i, j) for i, j in g), max(max(i, j) for i, j in g)
                print("Lower and Upper index {}".format((lower,upper)))
                size = upper - lower + 1
                print("Size {}".format(size))
                output = np.identity(size, dtype='int')
                # Identity used because we should consider a point to be similar to itself
                for edge in g:
                    output[edge[0]-lower, edge[1]-lower] = 1
                print("Saving new similarity values as a matrix: {}_Matrix.csv".format(fname))
                np.savetxt(fname+'_Matrix.csv',  np.asarray(output), delimiter=",", fmt='%d')
            print('Saved ' + fname+'_Matrix.csv')
            return output
    else:
        return np.array(load_csv(fname+'.csv'), dtype='int')

def load_extracted_features(onlyseeds=False):
    if onlyseeds:
        X = load_csv('Extracted_features.csv')
        S = load_seed()
        output = np.zeros((60,1084))
        i = 0;
        for index, label in S:
            output[i] = X[index-1]
            i += 1
        return output
    return load_csv('Extracted_features.csv')

# Loads extracted features with PCA selected k features
# Uses a preloaded .csv of PCA run for all components
# You want to run PCA separately of each amount of components because
# Accuracy tends to differ in lower dimensions
def load_extracted_features_PCA(k=1084, onlyseeds=False):
    fname = "Extracted_features_PCA{}.csv".format(k)
    preload = load_csv(fname)
    if preload is None:
        print('Running PCA on extracted_features{}'.format(k))
        X = StandardScaler(with_std=False).fit_transform(load_extracted_features())
        preload = PCA(n_components=k).fit_transform(X)
        np.savetxt(fname, np.asarray(preload), delimiter=",", fmt='%.5f')
        print('Saved ' + fname)

    if onlyseeds:
        s = load_seed()
        preload_seeds = np.zeros((s.shape[0],k))
        for i in range(s.shape[0]):
            # print(s[i][0])
            preload_seeds[i] = preload[s[i][0]-1]
        preload = preload_seeds

    true_k = min(k, preload.shape[1])
    output = preload[:, :true_k] # Get the k first eigens of PCA output
    assert output.shape == (preload.shape[0],true_k), "New shape {} should be {}".format(output.shape,(preload.shape[0],true_k))
    return output

def load_seed():
    return np.array(load_csv('Seed.csv'), dtype='int')

# Creates adjacency matrix for seeds, 1 if both have same label
def load_seed_matrix(fname='Seed_Matrix'):
    preload = load_csv(fname+'.csv')
    if preload is not None:
        return preload
    else:
        print('Creating Seed Matrix')
        s = load_seed()
        nums = s.shape[0]
        output = np.zeros((nums, nums), dtype='int')
        for i in range(nums):
            for j in range(nums):
                label1, label2 = s[i,1], s[j,1]
                output[i,j] = int(s[i,1]==s[j,1])
        assert check_symmetric(output), 'Seed Matrix should be symmetric'
        np.savetxt(fname+'.csv',  np.asarray(output), delimiter=",", fmt='%d')
        print('Saved ' + fname+'.csv')
        return output

def load_spectral_embedding(k=5990, g_type='adj'):
    if g_type == 'adj':
        fname = 'SpectralEmbedding.csv'
        preload = load_csv(fname)
        if preload is None:
            matrix = load_graph(shape_match=True,g_type=g_type)
            features, f_test = 1000, 500
            print('Running SpectralEmbedding for Graph_Matrix.csv with {} features'.format(features))
            preload = SpectralEmbedding(n_components=features, affinity='precomputed').fit_transform(matrix)
            print('Also Running SpectralEmbedding with {} features for sanity check'.format(f_test))
            p_test = SpectralEmbedding(n_components=f_test, affinity='precomputed').fit_transform(matrix)
            np.savetxt(fname, np.asarray(preload), delimiter=",", fmt='%.4f')
            print('Saved ' + fname)
            print('Sanity Check: Eigen Vectors are the same even if computed with different features: {}'.format('Passed' if np.allclose(preload[:,:f_test], p_test) else 'Failed'))
    elif g_type == 'dist':
        fname = 'SpectralEmbeddingDist.csv'
        preload = load_csv(fname)
        if preload is None:
            matrix = load_graph(shape_match=True, g_type=g_type)
            features, f_test = 1000, 500
            print('Running SpectralEmbedding for Graph_Dist_Matrix.csv with {} features'.format(features))
            preload = SpectralEmbedding(n_components=features, affinity='precomputed').fit_transform(matrix)
            print('Also Running SpectralEmbedding with {} features for sanity check'.format(f_test))
            p_test = SpectralEmbedding(n_components=f_test, affinity='precomputed').fit_transform(matrix)
            np.savetxt(fname, np.asarray(preload), delimiter=",", fmt='%.4f')
            # np.savetxt('SpectralEmbeddingDistTest.csv', np.asarray(p_test), delimiter=",", fmt='%.4f')
            print('Saved ' + fname)
            print('Sanity Check: Eigen Vectors are the same even if computed with different features: {}'.format('Passed' if np.allclose(preload[:,:f_test], p_test) else 'Failed'))
    elif g_type == 'merged':
        fname = 'SpectralEmbeddingMerged.csv'
        preload = load_csv(fname)
        if preload is None:
            matrix = load_merged_graph_matrix()
            features, f_test = 1000, 500
            print('Running SpectralEmbedding for Graph_Matrix_Merged.csv with {} features'.format(features))
            preload = SpectralEmbedding(n_components=features, affinity='precomputed').fit_transform(matrix)
            print('Also Running SpectralEmbedding with {} features for sanity check'.format(f_test))
            p_test = SpectralEmbedding(n_components=f_test, affinity='precomputed').fit_transform(matrix)
            np.savetxt(fname, np.asarray(preload), delimiter=",", fmt='%.4f')
            # np.savetxt('SpectralEmbeddingMergedTest.csv', np.asarray(p_test), delimiter=",", fmt='%.4f')
            print('Saved ' + fname)
            print('Sanity Check: Eigen Vectors are the same even if computed with different features: {}'.format('Passed' if np.allclose(preload[:,:f_test], p_test) else 'Failed'))

    true_k = min(k, preload.shape[1])
    output = preload[:, :true_k] # Get the k first eigens of PCA output
    assert output.shape == (preload.shape[0],true_k), "New shape {} should be {}".format(output.shape,(preload.shape[0],true_k))
    return output

# loads (or creates) a .csv for similarities to all known labels as a [10] element array per row
# e.g [0] : [sim to 0s, sim to 1s,.... sim to 9s]
def load_seed_similarity(fname='Seed_Similarity'):
    preload = load_csv(fname+'.csv')
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
        print('Saved ' + fname)
        return output

def load_CCA_features(k=8, onlyseeds=False):
    fname = "CCAPred{}.csv".format(k)
    preload = load_csv(fname)
    if preload is None:
        print(fname + 'not in DataSets')
        return None

    if onlyseeds:
        s = load_seed()
        output = np.zeros((s.shape[0],preload.shape[1]))
        for i in range(s.shape[0]):
            output[i] = preload[s[i][0]-1]
        return output
    else:
        return preload

# Produces the multiplied sum of
def load_merged_graph_matrix():
    preload = load_csv('Graph_Matrix_Merged.csv')
    if preload is None:
        D = load_graph(shape_match=True, g_type='dist')[:6000, :6000]
        G = load_graph(shape_match=True, g_type='adj')
        assert D.shape == G.shape == (6000,6000), '{} {}'.format(D.shape, G.shape)
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                D[i,j] *= G[i,j]
        np.savetxt('Graph_Matrix_Merged.csv',  np.asarray(D), delimiter=",")
        return D
    else:
        return preload




if __name__ == '__main__':
    print("Visible functions")
    print("load_extracted_features(onlyseeds=False)  : Load Extracted_features.csv as a numpy array")
    print("                                            onlyseeds will load only features of seeds")
    print("load_graph(shape_match=False, gtype='adj'): Load Graph.csv as a numpy array")
    print("                                            shape_match will make it a 6000x6000 adjacency matrix")
    print("                                            A point(i,j) = 1 if element i and j are similar")
    print("                                            g_type='dist' will make it a distance matrix using ")
    print("                                            10 features from PCA output of extracted_features ")
    print("load_seed()                               : Load Seed.csv into a numpy array")
    print("load_seed_similarity()                    : Load Seed_Similarity.csv into a numpy array")
    print("load_seed_matrix()                        : Loads an adjacency matrix of Seeds")
    print("                                            A point(i,j) = 1 if Seed i and Seed j are the same label")
    print("load_extracted_features_PCA(k, onlyseeds) : Loads the PCA output of Extracted_features.csv")
    print("Default k = 1084, onlyseeds = False         with k components. onlyseeds is also an allowed arg")
    print("load_spectral_embedding(k, gtype='adj')   : Loads the SpectralEmbedding of load_graph(shape_match=True)")
    print("                                            with k components")
    print("                                            g_type='dist' will load the SpectralEmbedding of Graph_Dist_Matrix")
    print("                                            g_type='merged' will load the SpectralEmbedding of Graph_Matrix_Merged")
    print("load_merged_graph_matrix()                : Loads the merged matrix of Graph_Matrix and Graph_Dist_Matrix")
    print("                                            such that M[i,j] = D[i,j] * G[i,j]")
    print("Running Unit Tests (This will take a while): ...")
    # load_merged_graph_matrix()
    unittests()
