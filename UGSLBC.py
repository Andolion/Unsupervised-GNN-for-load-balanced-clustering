import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import networkx as nx
import matplotlib.pyplot as plt
import Dataset_creator
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'dropout_rate',
    0.5,
    'Dropout rate for GNN representations.',
    lower_bound=0,
    upper_bound=1)
flags.DEFINE_integer(
    'nr_servers',
    3,
    'Number of servers.',
    lower_bound=2)
flags.DEFINE_integer(
    'nr_epochs',
    50,
    'Number of epochs.',
    lower_bound=10)
flags.DEFINE_float(
    'learning_rate',
    0.001,
    'Learning rate.',
    lower_bound=0)
flags.DEFINE_float(
    'loss_weight',
    0.03,
    'The weight for the sum of losses.',
    lower_bound=0)
flags.DEFINE_integer(
    'type_dataset',
    0,
    'The dataset to run the GNN on.',
    lower_bound=0)
flags.DEFINE_integer(
    'neuron_multiplier',
    2,
    'The multiplier for the neurons of the hidden layer.',
    lower_bound=1)
flags.DEFINE_integer(
    'rng_gen_seed',
    45678,
    'The seed for the random number generator for the networks.',
    lower_bound=0)
flags.DEFINE_integer(
    'rng_gnn_seed',
    123,
    'The seed for the random number generator for the GNN.',
    lower_bound=0)
flags.DEFINE_enum(
    'activation_function',
    'relu',
    ['relu', 'selu', 'tanh', 'leaky', 'sigmoid'],
    'The activation function used in the model.')
flags.DEFINE_integer(
    'x_reduce',
    0,
    'Number of columns to redistribute.',
    lower_bound=0)
flags.DEFINE_integer(
    'y_reduce',
    0,
    'Number of rows to redistribute.',
    lower_bound=0)

class Approx_loss(torch.nn.Module):
    """
    The custom loss function
    """
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, out, env, hops, dis):
        #compute the loads of the each node
        loads = torch.empty((len(out), 1), dtype=torch.float, requires_grad=False)
        for i in range(0, len(out)):
            if dis:
                loads[i] = 5000
            else:
                loads[i] = env.node_object[str(i+env.server_number)].traffic_sent + env.node_object[str(i+env.server_number)].traffic_received

        #the load balancing loss
        cluster_load = torch.mul(out, loads)
        cluster_load = torch.sum(cluster_load, dim=0)
        cluster_mean = torch.mean(cluster_load)
        cluster_load = cluster_load-cluster_mean
        load_loss = torch.linalg.norm(cluster_load)
        #print(load_loss)

        #the hop distance loss
        hop_dist_loss = torch.mean(torch.linalg.norm(torch.mul(out, hops**2), dim=1))
        #print(hop_dist_loss)
 
        loss = hop_dist_loss + FLAGS.loss_weight*load_loss
        return loss

class Tsi_loss(torch.nn.Module):
    """
    The adjusted Tsitsulin loss function
    """
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, out, env, hops):
        cluster_sizes = torch.sum(out, dim=0)
        adjacency = nx.adjacency_matrix(env.state_G_no_sink)
        adjacency = torch.tensor(adjacency.todense(), dtype=torch.float)
        number_edges = torch.tensor(len(list(env.state_G_no_sink.edges())), dtype=torch.float)
        number_nodes = torch.tensor(env.node_number, dtype=torch.float)
        cluster_number = torch.tensor(env.server_number, dtype=torch.float)
        degree = torch.sum(adjacency, dim=1)

        graph_pool = torch.transpose(torch.matmul(adjacency, out),0,1)
        graph_pool = torch.matmul(graph_pool, out)

        norm_left = torch.matmul(torch.transpose(out,0,1), torch.transpose(degree,-1,0))
        norm_right = torch.matmul(degree, out)
        normalizer = torch.matmul(norm_left, norm_right)/2/number_edges

        mod_loss = -torch.trace(graph_pool-normalizer)/2/number_edges
        
        regul_loss = torch.linalg.norm(cluster_sizes)/number_nodes *torch.sqrt(cluster_number) - 1

        #the added element for hop distance
        hop_dist_loss = torch.mean(torch.linalg.norm(torch.mul(out, hops), dim=1))

        #print(mod_loss, regul_loss, hop_dist_loss)
        loss = mod_loss + 1*regul_loss + 0.7*hop_dist_loss
        return loss

class UGS(torch.nn.Module):
    """
    The GNN model
    """
    def __init__(self, hidden_channels):
        #choose the type of GNN layer you want
        super().__init__()
        torch.manual_seed(FLAGS.rng_gnn_seed)
        #self.conv1 = GCNConv(FLAGS.nr_servers*2, hidden_channels, aggr = 'add') 
        #self.conv2 = GCNConv(hidden_channels, FLAGS.nr_servers, aggr = 'add')      
        self.conv1 = SAGEConv(FLAGS.nr_servers*2, hidden_channels, aggr = 'add')
        self.conv2 = SAGEConv(hidden_channels, FLAGS.nr_servers, aggr = 'add') 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        if FLAGS.activation_function == 'relu':
            x = F.relu(x)
        if FLAGS.activation_function == 'leaky':
            x = F.leaky_relu(x)
        if FLAGS.activation_function == 'selu':
            x = F.selu(x)
        if FLAGS.activation_function == 'tanh':
            x = F.tanh(x)
        if FLAGS.activation_function == 'sigmoid':
            x = F.sigmoid(x)
        x = F.dropout(x, p=FLAGS.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1) 
        return x

class Tsi_model(torch.nn.Module):
    """
    The Tsitsulin model
    """
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(123)
        self.conv1 = GCNConv(FLAGS.nr_servers*2, hidden_channels, aggr = 'add') 
        self.conv2 = GCNConv(hidden_channels, FLAGS.nr_servers, aggr = 'add') 
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.selu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.softmax(x, dim=1) 
        return x

def main(argv):
    torch.set_printoptions(threshold=10000, linewidth=200, sci_mode=False)
    #create the dataset handler object
    dataset_cre = Dataset_creator.Dataset(FLAGS.nr_servers, FLAGS.type_dataset, FLAGS.rng_gen_seed, x_reduce=FLAGS.x_reduce, y_reduce=FLAGS.y_reduce)
    #create the training dataset
    dataset = dataset_cre.create_dataset()
    #create the testing dataset
    test_dataset = dataset_cre.create_test_dataset()
    #create the GNN model
    model = UGS(hidden_channels=FLAGS.nr_servers*FLAGS.neuron_multiplier)
    #create the Tsitsulin model
    tsi_model = Tsi_model(hidden_channels=64)
    #define loss criterion
    criterion = Approx_loss()  
    #define tsitsulin loss criterion
    tsi_crit = Tsi_loss()
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)  
    #define tsitsulin optimizer
    tsi_optimizer = torch.optim.Adam(tsi_model.parameters(), lr=0.001)

    #Initial print of all parameters for better viewing
    print("seed", FLAGS.rng_gen_seed)
    print("gnn seed", FLAGS.rng_gnn_seed)
    print("dataset", FLAGS.type_dataset)
    print("lr", FLAGS.learning_rate)
    print("servers:", FLAGS.nr_servers)
    print("epochs", FLAGS.nr_epochs)
    print("function", FLAGS.activation_function)
    print("dropout", FLAGS.dropout_rate)
    print("neuron multiplier", FLAGS.neuron_multiplier)
    print("beta", FLAGS.loss_weight)
    print("x reduce", FLAGS.x_reduce)
    print("y reduce", FLAGS.y_reduce)

    def train(hops, dis, k):
        optimizer.zero_grad()  # Clear gradients
        out = model(dataset[k].x, dataset[k].edge_index)
        loss = criterion(out, dataset_cre.dataset_raw[k], hops, dis)
        loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
        return loss

    def compute_config(network):
        model.eval()
        """
        Model output is of shape N x C. Where N is the number of nodes, including the server nodes.
        C is the number of cluster. Each element is a probability of that node belonging to that cluster.
        This is transformed with argmax into the new configuration of the network.
        """
        out = model(network.x, network.edge_index)
        multi = torch.argmax(out,dim=1)
        config = []
        for i in range(0, len(multi)):
            node_config = []
            for j in range(0, FLAGS.nr_servers):
                if j == multi[i]:
                    node_config.append(1)
                else:
                    node_config.append(0)
            config.append(node_config)
        return config

    def tsi_train(k, hops):
        tsi_optimizer.zero_grad()  # Clear gradients
        out = tsi_model(dataset[k].x, dataset[k].edge_index)
        loss = tsi_crit(out, dataset_cre.dataset_raw[k], hops)
        loss.backward()  # Derive gradients
        tsi_optimizer.step()  # Update parameters based on gradients
        return loss
    
    def tsi_compute_config(network):
        tsi_model.eval()
        """
        Model output is of shape N x C. Where N is the number of nodes, including the server nodes.
        C is the number of cluster. Each element is a probability of that node belonging to that cluster.
        This is transformed with argmax into the new configuration of the network.
        """
        out = tsi_model(network.x, network.edge_index)
        multi = torch.argmax(out,dim=1)
        config = []
        for i in range(0, len(multi)):
            node_config = []
            for j in range(0, FLAGS.nr_servers):
                if j == multi[i]:
                    node_config.append(1)
                else:
                    node_config.append(0)
            config.append(node_config)
        return config

    losses = []     #UGS loss counter
    losses2 = []    #Auxiliarry loss counter
    tsi_loss = []   #Tsitsulin loss counter
    tsi_loss2 = []  #Auxiliarry loss counter
    hops = []   #initial hop distances
    voronoi_load = []   #load balancing metric for Voronoi
    gnn_load = []       #load balancing metric for UGS
    tsi_load = []       #load balancing metric for Tsitsulin
    best_load = []      #best implementation for each network tested
    
    #Prepare the initial data
    for i in range(0,len(dataset)):
        hops.append(torch.tensor(dataset_cre.dataset_raw[i].hops, dtype=torch.float))
        dataset_cre.dataset_raw[i].run_network(100)
        load_mean, loads, latency = dataset_cre.dataset_raw[i].network_stats()
    #Report on Voronoi performance
    for i in range(0,len(test_dataset)):
        dataset_cre.test_dataset_raw[i].run_network(100)
        load_mean, loads, latency = dataset_cre.test_dataset_raw[i].network_stats()
        voronoi_load.append(np.linalg.norm(loads-load_mean))

    for epoch in range(FLAGS.nr_epochs):
        #Train the GNN
        for i in range(0,len(dataset)):
            configuration = compute_config(dataset[i])      #Getting the cluster configuration from the GNN
            dataset_cre.dataset_raw[i].cluster_network(configuration)  #Reconfiguring the network
            #Run simulation and check if network became disconnected
            if dataset_cre.dataset_raw[i].run_network(100) == "disconnected":
                dis = 1
            else:
                dis = 0
            #compute loss
            loss = train(hops[i], dis, i)
            print(f'Epoch: {epoch:03d}, Network: {i}, Loss: {loss:.4f}')
            losses2.append(loss.detach().numpy())
            #same process for Tsitsulin
            configuration = tsi_compute_config(dataset[i])
            dataset_cre.dataset_raw[i].cluster_network(configuration)
            dataset_cre.dataset_raw[i].run_network(100)
            loss = tsi_train(i, hops[i])
            tsi_loss2.append(loss.detach().numpy())
        losses.append(np.mean(losses2))
        tsi_loss.append(np.mean(tsi_loss2))

    for i in range(0, len(test_dataset)):
        configuration = compute_config(test_dataset[i])              #Getting the cluster configuration from the GNN
        dataset_cre.test_dataset_raw[i].cluster_network(configuration)  #Reconfiguring the network.
        dataset_cre.test_dataset_raw[i].run_network(100)
        load_mean, loads, latency = dataset_cre.test_dataset_raw[i].network_stats()
        #check if something went wrong
        if load_mean == 0:
            gnn_load.append(30000)
        else:
            gnn_load.append(np.linalg.norm(loads-load_mean))

        configuration = tsi_compute_config(test_dataset[i])              #Getting the cluster configuration from the GNN
        dataset_cre.test_dataset_raw[i].cluster_network(configuration)  #Reconfiguring the network.
        dataset_cre.test_dataset_raw[i].run_network(100)
        load_mean, loads, latency = dataset_cre.test_dataset_raw[i].network_stats()
        if load_mean == 0:
            tsi_load.append(30000)
        else:
            tsi_load.append(np.linalg.norm(loads-load_mean))

        best_load.append(np.argmin((gnn_load[i], voronoi_load[i], tsi_load[i])))

    #Again the parameters for better viewing
    print("seed", FLAGS.rng_gen_seed)
    print("gnn seed", FLAGS.rng_gnn_seed)
    print("dataset", FLAGS.type_dataset)
    print("lr", FLAGS.learning_rate)
    print("servers:", FLAGS.nr_servers)
    print("epochs", FLAGS.nr_epochs)
    print("function", FLAGS.activation_function)
    print("dropout", FLAGS.dropout_rate)
    print("neuron multiplier", FLAGS.neuron_multiplier)
    print("beta", FLAGS.loss_weight)
    print("x reduce", FLAGS.x_reduce)
    print("y reduce", FLAGS.y_reduce)

    #Print the load balancing metrics
    np.set_printoptions(precision=3)
    print("Load balancing statistics")
    gnn_load = np.array(gnn_load)
    print(gnn_load)
    voronoi_load = np.array(voronoi_load)
    print(voronoi_load)
    tsi_load = np.array(tsi_load)
    print(tsi_load)
    print(best_load)
    print()
 
if __name__ == "__main__":
    app.run(main)