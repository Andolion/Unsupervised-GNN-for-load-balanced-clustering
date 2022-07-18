import numpy as np
import torch
import random
from torch_geometric.data import Data
from Network_behaviour import NetworkGNN
import copy

class Dataset(object):
    """
    The dataset containing all the networks
    """
    def __init__(self, server_number, mode, rng_seed, x_reduce = 0, y_reduce = 0):
        random.seed(rng_seed)
        self.dataset_raw = []
        self.test_dataset_raw = []
        
        if mode == 0:
            """
            random dataset
            training: 100 random networks of varying sizes
            testing: 9 random networks of varying sizes
            """
            #create the training networks
            for i in range(0, 10):
                for j in range(0, 10):
                    #make sure networks are large enough for the number of servers
                    if i < 5:
                        i2 = i + server_number
                    else:
                        i2 = i
                    if j < 5:
                        j2 = j + server_number
                    else:
                        j2 = j
                    new_seed=random.randint(10000, 99999)
                    self.dataset_raw.append(NetworkGNN(0, i2, j2, server_number, 0, rng_seed=new_seed))
            #create the testing networks
            for i in range(server_number+1, server_number+4):
                for i in range(server_number+1, server_number+4):
                    new_seed=random.randint(10000, 99999)
                    self.test_dataset_raw.append(NetworkGNN(0, i, j, server_number, 0, rng_seed=new_seed))
        
        if mode == 1:
            """
            dynamic dataset
            training: 20 snapshots of the same network
            testing: 10 snapshots of the same network
            """
            #the initial network
            og = NetworkGNN(0, 2*server_number, 2*server_number, server_number, 0, rng_seed=rng_seed)
            self.dataset_raw.append(og)
            self.test_dataset_raw.append(og)
            #create traininig networks
            for i in range(1, 20):
                self.dataset_raw.append(copy.deepcopy(self.dataset_raw[-1]))
                prev_network = self.dataset_raw[-1]
                move_node = random.randint(prev_network.server_number, prev_network.node_number+prev_network.server_number-1)
                redo = False
                while redo == False:
                    prev_network.delete_node(move_node)
                    new_x = random.randint(0,2*server_number*10)+ random.random()
                    new_y = random.randint(0,2*server_number*10)+ random.random()
                    prev_network.add_node(move_node, new_x, new_y)
                    redo = self.dataset_raw[-1].check_connectivity()
                self.dataset_raw[-1].init_node()
            #create testing networks
            for i in range(1, 10):
                self.test_dataset_raw.append(copy.deepcopy(self.test_dataset_raw[-1]))
                prev_network = self.test_dataset_raw[-1]
                move_node = random.randint(prev_network.server_number, prev_network.node_number+prev_network.server_number-1)
                redo = False
                while redo == False:
                    prev_network.delete_node(move_node)
                    new_x = random.randint(0,2*server_number*10)+ random.random()
                    new_y = random.randint(0,2*server_number*10)+ random.random()
                    prev_network.add_node(move_node, new_x, new_y)
                    redo = self.dataset_raw[-1].check_connectivity()
                self.test_dataset_raw[-1].init_node()
                    
        if mode == 2:
            """
            same size dataset
            training: 100 random networks of the same size
            testing: 10 random networks of the same size
            Deprecated
            """
            for i in range(0, 100):
                new_seed=random.randint(10000, 99999)
                self.dataset_raw.append(NetworkGNN(0, 6, 6, server_number, 0, rng_seed=new_seed))
            for i in range(0, 10):
                new_seed=random.randint(10000, 99999)
                self.test_dataset_raw.append(NetworkGNN(0, 6, 6, server_number, 0, rng_seed=new_seed))
        
        if mode == 3:
            """
            grid dataset
            training: 50 grid networks of varying sizes
            testing: 14 grid networks of varying sizes
            """
            for i in range(server_number+1, server_number+9):
                for j in range(server_number+1, server_number+9):
                    new_seed=random.randint(10000, 99999)
                    self.dataset_raw.append(NetworkGNN(1, i, j, server_number, 0, rng_seed=new_seed))
            for i in range(0, 14):
                self.test_dataset_raw.append(self.dataset_raw.pop(random.randint(0, len(self.dataset_raw)-1)))

        if mode == 4:
            """
            transmission x2 dataset
            training: 100 random networks of varying sizes with transmission range two times the grid size
            testing: 10 random networks of varying sizes with transmission range two times the grid size
            Deprecated
            """
            for k in range(0, 100):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.dataset_raw.append(NetworkGNN(0, i, j, server_number, 0, rng_seed=new_seed, transmission = 2))
            for k in range(0, 10):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.test_dataset_raw.append(NetworkGNN(0, i, j, server_number, 0, rng_seed=new_seed, transmission = 2))
        
        if mode == 5:
            """
            sparse dense dataset
            training: 50 random networks of varying sizes with different network concentrations
            testing: 10 random networks of varying sizes with different network concentrations
            """
            for k in range(0, 50):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.dataset_raw.append(NetworkGNN(5, i, j, server_number, 0, rng_seed=new_seed))
            for k in range(0, 10):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.test_dataset_raw.append(NetworkGNN(5, i, j, server_number, 0, rng_seed=new_seed))

        if mode == 6:
            """
            net dataset
            training: 50 random networks of varying sizes under the shape of a net
            testing: 10 random networks of varying sizes under the shape of a net
            """
            for k in range(0, 50):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.dataset_raw.append(NetworkGNN(6, i, j, server_number, 0, rng_seed=new_seed))
            for k in range(0, 10):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.test_dataset_raw.append(NetworkGNN(6, i, j, server_number, 0, rng_seed=new_seed))
        
        if mode == 7:
            """
            cercles dataset
            training: 50 random networks of varying sizes under the shape of a net
            testing: 10 random networks of varying sizes under the shape of a net
            Deprecated
            """
            for k in range(0, 50):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.dataset_raw.append(NetworkGNN(7, i, j, server_number, 0, rng_seed=new_seed))
            for k in range(0, 10):
                new_seed=random.randint(10000, 99999)
                i = random.randint(2*server_number, 10)
                j = random.randint(2*server_number, 10)
                self.test_dataset_raw.append(NetworkGNN(7, i, j, server_number, 0, rng_seed=new_seed))
        
        if mode == 8:
            """
            trend dataset
            training: 50 random networks of varying sizes with different network concentrations
            testing: 10 random networks of varying sizes with different network concentrations
            This is actually the sparse dense dataset, but with variable density distribution percentage
            """
            for k in range(0, 50):
                new_seed=random.randint(10000, 99999)
                self.dataset_raw.append(NetworkGNN(8, 7, 7, server_number, 0, rng_seed=new_seed, x_reduce=x_reduce, y_reduce=y_reduce))
            for k in range(0, 10):
                new_seed=random.randint(10000, 99999)
                self.test_dataset_raw.append(NetworkGNN(8, 7, 7, server_number, 0, rng_seed=new_seed, x_reduce=x_reduce, y_reduce=y_reduce))

    def create_dataset(self):
        dataset = []
        for k in range(0, len(self.dataset_raw)):
            env = self.dataset_raw[k]
            """
            The feature set is of shape N x 2*S. Where N is the number of nodes, including server nodes.
            And S is the number of server nodes 
            First S inputs represent the clustering of the current node
            Next S inputs represent the hop distances to each server from the current node
            """
            x = []
            for i in range(env.server_number, env.server_number+env.node_number):
                xi = []
                for j in range(0, env.server_number):
                    xi.append(env.node_object[str(i)].clusters[j])
                    xi.append(env.node_object[str(i)].hop_dis[j])
                x.append(xi)
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.float)

            """
            Edge index is just the list of edges. 
            Currently in the form [v1,u1], [v2,u2].....
            When creating the dataset, it will be changed to contiguous, [v1, v2....][u1, u2...]
            """
            edge_index = list(env.state_G_no_sink.edges())
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index-env.server_number

            """Creating the dataset"""
            dataset.append(Data(x=x, edge_index=edge_index.t().contiguous()))

        return dataset

    def create_test_dataset(self):
        dataset = []
        for k in range(0, len(self.test_dataset_raw)):
            env = self.test_dataset_raw[k]
            """
            The feature set is of shape N x 2*S. Where N is the number of nodes, including server nodes.
            And S is the number of server nodes 
            First S inputs represent the clustering of the current node
            Next S inputs represent the hop distances to each server from the current node
            """
            x = []
            for i in range(env.server_number, env.server_number+env.node_number):
                xi = []
                for j in range(0, env.server_number):
                    xi.append(env.node_object[str(i)].clusters[j])
                    xi.append(env.node_object[str(i)].hop_dis[j])
                x.append(xi)
            x = np.array(x)
            x = torch.tensor(x, dtype=torch.float)

            """
            Edge index is just the list of edges. 
            Currently in the form [v1,u1], [v2,u2].....
            When creating the dataset, it will be changed to contiguous, [v1, v2....][u1, u2...]
            """
            edge_index = list(env.state_G_no_sink.edges())
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_index = edge_index-env.server_number

            """Creating the dataset"""
            dataset.append(Data(x=x, edge_index=edge_index.t().contiguous()))

        return dataset 