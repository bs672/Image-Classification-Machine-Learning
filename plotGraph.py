
import networkx as nx
import matplotlib.pyplot as plt
import os

G_fb = nx.read_edgelist(os.path.join('DataSets','Graph.csv'), delimiter=',', create_using = nx.Graph(), nodetype = int)

print(nx.info(G_fb))

#Create network layout for visualizations
spring_pos = nx.spring_layout(G_fb)

plt.axis("off")
print('Plotting')
nx.draw_networkx(G_fb, pos = spring_pos, with_labels = False, node_size = 15)
plt.plot()

from multiprocessing import Pool
import itertools

def partitions(nodes, n):
    "Partitions the nodes into n subsets"
    nodes_iter = iter(nodes)
    while True:
        partition = tuple(itertools.islice(nodes_iter,n))
        if not partition:
            return
        yield partition

def btwn_pool(G_tuple):
    return nx.betweenness_centrality_source(*G_tuple)

# def between_parallel(G, processes = 1):
#     p = Pool(processes=processes)
#     part_generator = 4*len(p._pool)
#     node_partitions = list(partitions(G.nodes(), int(len(G)/part_generator)))
#     num_partitions = len(node_partitions)
#
#     bet_map = p.map(btwn_pool,
#                         zip([G]*num_partitions,
#                         [True]*num_partitions,
#                         [None]*num_partitions,
#                         node_partitions))
#
#     bt_c = bet_map[0]
#     for bt in bet_map[1:]:
#         for n in bt:
#             bt_c[n] += bt[n]
#     return bt_c

bt = nx.betweenness_centrality_source(G_fb)
top = 10

max_nodes =  sorted(bt.iteritems(), key = lambda v: -v[1])[:top]
bt_values = [5]*len(G_fb.nodes())
bt_colors = [0]*len(G_fb.nodes())
for max_key, max_val in max_nodes:
    bt_values[max_key] = 150
    bt_colors[max_key] = 2

plt.axis("off")
print('Plotting')
nx.draw_networkx(G_fb, pos = spring_pos, cmap = plt.get_cmap("rainbow"), node_color = bt_colors, node_size = bt_values, with_labels = False)
plt.plot()

import community

parts = community.best_partition(G_fb)
values = [parts.get(node) for node in G_fb.nodes()]

plt.axis("off")
print('Plotting')
nx.draw_networkx(G_fb, pos = spring_pos, cmap = plt.get_cmap("jet"), node_color = values, node_size = 35, with_labels = False)
plt.plot()
