import pickle
import dgl
import torch

def get_CSL_graphs():
    adj_list = pickle.load(open('C:\\Users\\User1\\Documents\\GitHub\\ResearchProject1\\WLColorRefinement\\data\\CSL\\graphs_Kary_Deterministic_Graphs.pkl', 'rb'))
    graph_labels = torch.load('C:\\Users\\User1\\Documents\\GitHub\\ResearchProject1\\WLColorRefinement\\data\\CSL\\y_Kary_Deterministic_Graphs.pt')
    graph_lists = []

    # n_samples = len(graph_labels)
    num_node_type = 1 #41
    num_edge_type = 1 #164

    for sample in adj_list:
        _g = dgl.DGLGraph()
        _g = dgl.from_scipy(sample)
        g = dgl.transform.remove_self_loop(_g)
        g.ndata['feat'] = torch.zeros(g.number_of_nodes()).long()
        
        # adding edge features as generic requirement
        g.edata['feat'] = torch.zeros(g.number_of_edges()).long()
        
        graph_lists.append(g)
    num_node_type = graph_lists[0].ndata['feat'].size(0)
    snum_edge_type = graph_lists[0].edata['feat'].size(0)
    return (graph_lists)