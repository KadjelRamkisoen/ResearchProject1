import numpy as np
import torch as th
import dgl


def _send_color(edges):
  return {'color': edges.src['color']}


def _gen_create_multiset(num_nodes):
  def _create_multiset(nodes):
    end = nodes.mailbox['color'].shape[1]
    multiset = th.zeros((nodes.batch_size(), num_nodes)) - 1
    multiset[:, 0] = nodes.data['color']
    multiset[:, 1:end + 1] = nodes.mailbox['color'].sort().values
    return {'color': multiset}
  return _create_multiset


def _to_color(colors):
  colors = colors[colors >= 0]
  self_color = colors.astype(int).astype(str)[0]
  neighbour_color = colors[1:].astype(int).astype(str).tolist()
  return self_color + '|' + ','.join(neighbour_color)


def _update_colors(G):
  N = G.number_of_dst_nodes()
  G.update_all(message_func = _send_color, reduce_func = _gen_create_multiset(N))
  return list(map(_to_color, G.ndata.pop('color').cpu().numpy()))


def wl_coloring(G, max_iter=10):
  """Check if the two given graphs are possibly isomorphic by the 1-Weisfeiler-Lehman algorithm.

  Arguments:
      G {networkx.classes.graph.Graph} -- Graph
   
  Keyword Arguments:
      max_iter {int} -- The max number of iterations(default: {10})

  Returns:
      G {networkx.classes.graph.Graph} -- Graph with attribute color added
  """

  # Set initial colors
  G.ndata['color'] = th.ones(G.number_of_nodes())
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

    G.ndata['color'] = th.from_numpy(np.array([recolor_map[color]
                                 for color in G_colors]) + current_max_color)
    current_max_color += len(unique_colors)
  
  return True