import networkx as nx
import numpy as np
import pickle
from pathlib import Path
import os
import torch
import time
import dgl

"""
    super_node(dgl.DGLgraph)
    
    Function to create a supernode
    The idea is to have a function that iterates over the graph and
    finds the nodes with the same colors and then places them in one supernode
"""
def super_node(graph):
    super_node = {}
    colors = graph.ndata['feat'].numpy()
    nodes = graph.nodes().numpy()

    count = len(colors)
    i = 0
    while i < count:
        if colors[i] not in super_node:
            super_node[colors[i]] = dict()
        super_node[colors[i]][nodes[i]] = list()
        i += 1
    return super_node


"""
    Create_edge_list(dgl.DGLgraph)
    
    Function to create a list with (source node, destination node)
"""
def create_edge_list(graph):
    src_edges = graph.edges(form='uv', order='srcdst')[0].numpy()
    dst_edges = graph.edges(form='uv', order='srcdst')[1].numpy()
    
    concat_edges = list()
    for i in range(0, len(src_edges)):
        sub_list = list()
        sub_list.append(src_edges[i])
        sub_list.append(dst_edges[i])
        concat_edges.append(sub_list)
    return concat_edges  


"""
    super_edge(list, dict)
    
    Function to find the edges between supernodes.
    
    Make sure that the the src_edges of the create_edge_list function is sorted from low to high.
"""
def super_edge(edges, super_node):
    for color in super_node:
        for node in super_node[color]:
            for edge in edges:
                if node == edge[0]:
                    super_node[color][node].append(edge)
                elif node < edge[0]:
                    break
    return (super_node)


"""
    create_mapping(dict)
    @super_node - the supernode with original nodes added to it and a list of original edges
    
    Create a mapping from one nodes to super nodes.
"""
def create_mapping(super_node):
    mapping = {}
    for color in super_node:
        for node in super_node[color]:
            mapping[node] = color
    return mapping


"""
    prepare_DGLgraph(dict, dict)
    
    Prepare a DGLgraph with the following steps:
        1. Overwrite the original node with the color of the supernode
        2. Calculate and add the weights to the edges
           The weight of the edge is the total of edges that go from one node within the supernode 
           to another node in another supernode. If the number of edges from each node going out are not the same, 
           the smallest common value is taken.
"""
def prepare_DGLgraph(super_node, mapping): 
    super_node_reformat = {}
    for color, nodes in super_node.items():
        super_node_reformat[color] = {}
        for node, edges in nodes.items():
            super_node_reformat[color][node] = {}
            super_node_reformat[color][node]['edges'] = edges
            super_node_reformat[color][node]['weights'] = {}
            i = 0
            while i < len(edges):
                mapping_src = mapping[edges[i][0]]
                mapping_dst = mapping[edges[i][1]]
                super_node[color][node][i][0] = mapping_src               
                super_node[color][node][i][1] = mapping_dst
                i += 1 
            
                weights = super_node_reformat[color][node]['weights']
                
                if mapping_dst not in weights:
                    weights[mapping_dst] = 1
                else:
                    weights[mapping_dst] += 1
    weights, src_edges, dst_edges = calc_weights(super_node_reformat)
    
#     for i in super_node_reformat:
#         print('{')
#         print('Super node: ', i)
#         for j in super_node_reformat[i]:
#             print('Node: ', j)
#             print('Node data: ', super_node_reformat[i][j])
#         print('}')
#         print('\n')
#     print('super-node-reformat: ', super_node_reformat, '\n')                      
    return super_node_reformat, src_edges, dst_edges, weights


"""
    final_weights(dict)
    
    Function to calculate the final weights for the super edges
"""
def calc_weights(graph):
    weights_list = list()
    src_edges = list()
    dst_edges = list()
    
    for color in graph:
        # If the there is only one node in the supernode, then just add the weight from the 'weight' attribute
        if len(graph[color]) == 1:
            for node in graph[color]:
                for weight in graph[color][node]['weights']:
                    weights_list.append(graph[color][node]['weights'][weight])
                    src_edges.append(color)
                    dst_edges.append(weight)
        # If there are more nodes in the supernode, then check whether they all have outgoing edges to the same supernodes
        else:
            x = list()
            src_nodes = list()
            dst_nodes = list()
            for node in graph[color]:
                #Add all the weights for each node to the list x
                x.append(graph[color][node]['weights'])
                for w in graph[color][node]['weights']:
                    src_nodes.append(color)
                    dst_nodes.append(w)
            # Take the first value in the list x
            x_1 = x[0]

            i = 1
            # Check if all the edges within x_1 are also in the other lists in x
            for weight in x_1:
                while i < len(x):
                    # If the the edges from x_1 are in the next list in x, then check the weight
                    if weight in x[i]:
                        # Store the smallest weight value. There will be no edge is there is only one occurence of the edge in the list
                        if x[i][weight] >= x_1[weight]:
                            weights_list.append(x_1[weight])
                            src_edges.append(src_nodes[i])
                            dst_edges.append(dst_nodes[i])
                        elif x[i][weight] < x_1[weight]:
                            weights_list.append(x[i][weight])
                            src_edges.append(src_nodes[i])
                            dst_edges.append(dst_nodes[i])
                    i += 1
    return weights_list, src_edges, dst_edges


"""
    create_DGLGraph(list, list, list)
    
"""
def create_DGLGraph(src_nodes, dst_nodes, edata, mapping):
    src_nodes = torch.tensor(src_nodes)
    dst_nodes = torch.tensor(dst_nodes)
    edata = torch.tensor(edata)
    ndata = list()
    
    reduced_graph = dgl.graph((src_nodes, dst_nodes))
    reduced_graph.edata['feat'] = edata
    i = 0
    while i < len(reduced_graph.nodes()):
        ndata.append(mapping[i])
        i += 1
    
    reduced_graph.ndata['feat'] = torch.tensor(ndata)
    
    return reduced_graph

def reduced_graph(colored_graphs): 
    reduced_graphs_list = list()
    for graph in colored_graphs:
        superNode = super_node(graph)
        edges = create_edge_list(graph)
        superNode = super_edge(edges, superNode)
        node_color_mapping = create_mapping(superNode)
        prep_graph = prepare_DGLgraph(superNode, node_color_mapping)

        color_node_mapping = {}
        count = 0
        for i in node_color_mapping:
            if node_color_mapping[i] not in color_node_mapping:
                color_node_mapping[node_color_mapping[i]] = count
                count += 1

        src_nodes = list()
        dst_nodes = list()

        i = 0
        while i < len(prep_graph[1]):
            src_nodes.append(color_node_mapping[prep_graph[1][i]])
            dst_nodes.append(color_node_mapping[prep_graph[2][i]])
            i += 1

        reduced_graphs_list.append(
            create_DGLGraph(
                src_nodes, dst_nodes, prep_graph[3], node_color_mapping
            )
        )
    return reduced_graphs_list
