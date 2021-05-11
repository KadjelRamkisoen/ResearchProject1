import numpy as np
import torch as th
import dgl
import tensorflow as tf
import math

def _send_color(edges):
  return {'feat': edges.src['feat']}


def _gen_create_multiset(num_nodes):
  def _create_multiset(nodes):
    end = nodes.mailbox['feat'].shape[1]
    multiset = th.zeros((nodes.batch_size(), num_nodes)) - 1
    multiset[:, 0] = nodes.data['feat']
    multiset[:, 1:end + 1] = nodes.mailbox['feat'].sort().values
    return {'feat': multiset}
  return _create_multiset


def _to_color(colors):
  colors = colors[colors >= 0]
  self_color = colors.astype(int).astype(str)[0]
  neighbour_color = colors[1:].astype(int).astype(str).tolist()
  return self_color + '|' + ','.join(neighbour_color)


def _update_colors(G):
  N = G.number_of_dst_nodes()
  G.update_all(message_func = _send_color, reduce_func = _gen_create_multiset(N))
  return list(map(_to_color, G.ndata.pop('feat').cpu().numpy()))


def wl_coloring(G, max_iter=10):
  """Check if the two given graphs are possibly isomorphic by the 1-Weisfeiler-Lehman algorithm.

  Arguments:
      G {networkx.classes.graph.Graph} -- Graph
   
  Keyword Arguments:
      max_iter {int} -- The max number of iterations(default: {10})

  Returns:
      G {networkx.classes.graph.Graph} -- Graph with attribute color added
  """

  # Set initial colors if the feat tensor is empty
  if (tf.equal(tf.size(G.ndata['feat']), 0)):
    G.ndata['feat'] = th.ones(G.number_of_nodes())
  
  if len(list(G.ndata['feat'].size())) > 1:
    new_features = th.ones(G.number_of_nodes())
    
    for i in range(G.number_of_nodes()):
        single_int_list = list()
        # Ceil or floor the decimal values
        for j in G.ndata['feat'][i]:
            if int(math.modf(j)[0]*10) < 5:
                single_int_list.append(int(math.floor(j)))
            elif int(math.modf(j)[0]*10) > 5:
                single_int_list.append(int(math.ceil(j)))
        
        # Create a single integer from all the values
        single_int = sum(k * 10**l for l, k in enumerate(single_int_list[::-1]))
        
    new_features[i] = single_int
    G.ndata['feat'] = new_features
    
  N = G.number_of_dst_nodes()
  current_max_color = 0
  
  # Refine colors until convergence
  for i in range(max_iter):
    #print(i)
    G_colors = _update_colors(G)

    G_unique_colors, G_counts = np.unique(G_colors, return_counts=True)
    G_multiset = dict(zip(G_unique_colors, G_counts))

    # Recoloring (str -> int)
    unique_colors = np.unique(G_unique_colors)
    recolor_map = {color: i + 1 for i, color in enumerate(unique_colors)}

    G.ndata['feat'] = th.from_numpy(np.array([recolor_map[color]
                                 for color in G_colors]) + current_max_color)
    current_max_color += len(unique_colors)
  
  return G