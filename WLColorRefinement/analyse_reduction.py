import csv
import pandas as pd
import matplotlib.pyplot as plt

def store_reduction_data(file, original_graphs, reduced_graphs):
    i = 0

    with open('CompressionMolecules.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        fieldnames = ['Graph nr', 
                      'Original nodes', 'Reduced nodes', 'Node compr. rate', 
                      'Original edges', 'Reduced edges', 'Edge compr. rate'
                     ]
        writer = csv.writer(csv_file)
        writer.writerow(fieldnames)

        while i < len(reduced_graphs_list):   
            x = original_graphs.train.__getitem__(i)[0]
            y = reduced_graphs[i]
            nx_x = x.to_networkx()
            nx_y = y.to_networkx()
            pos_x = nx.kamada_kawai_layout(nx_x)
            pos_y = nx.kamada_kawai_layout(nx_y)

            writer.writerow([i,x.number_of_nodes(), 
                             y.number_of_nodes(), 
                             round((((x.number_of_nodes() - y.number_of_nodes())/x.number_of_nodes())*100),2),
                             x.number_of_edges(), 
                             x.number_of_edges(),
                             round((((x.number_of_edges() - y.number_of_edges())/x.number_of_edges())*100),2)
                            ])

            i += 1
            
            
def analyse_data(df):
    print('Analysis')
    print('\n')
    
    node = df[df.columns[3]]
    edge = df[df.columns[6]]

    plt.scatter(node, edge, s = 1)
    plt.show()
    
    edge.plot.hist(grid=True, bins=20, rwidth=0.9)
    node.plot.hist(grid=True, bins=20, rwidth=0.9)