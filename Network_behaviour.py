import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# The network is divided into a grid, each grid cell contains one node
GRID_DISTANCE = 10          # Size of one grid cell
GRID_NODE_NUM_X = 5         # Number of cell/nodes in x axis
GRID_NODE_NUM_Y = 5         # Number of cell/nodes in y axis

class Node(object):
    """ 
    Describe node/device basic information

    ATTRIBUTES:
    --------
    node_id: An integer representing node index in the network
    receive_time: Shows when the node needs to receive a message
    send_time: Shows when the node needs to send a message
    sent_to: Which node to send the data at a specific timeslot
    traffic_sent: Number of messages sent
    traffic_received: Number of messages received
    latency: The end to end latency of messages
    sleep_mode: A boolean representing if the node is sleeping
    period: Number of timeslots between two sendings
    awake_timeslots: A list of timeslots each node is awake for
    hop_dis: Hop distance from the node to the sink
    sink: ID of closest sink node
    path: The shortest path to sink
    cluster: The current cluster of the node
    """
    def __init__(self, node_id):
        self.node_id = node_id

        """Traffic related variables"""
        self.receive_time = []
        self.send_time = []
        self.sent_to = []
        self.traffic_sent = 0
        self.traffic_received = 0
        self.latency = []

        """RDC related variables"""
        self.sleep_mode = 1
        self.period = 70
        self.awake_timeslots = []

        """Cluster related variables"""
        self.hop_dis = []
        self.sink = []
        self.path = []
        self.clusters = []

class Network(object):
    """
    Establish wireless mesh network.
    Define node parameters and mesh topology.
    """

    def __init__(self, Test, Node_x, Node_y, Sink_nr, RNG_seed, Transmission, X_reduce, Y_reduce):
        """====== Node parameters ======"""
        self.grid_distance = GRID_DISTANCE
        self.grid_xcor_node_number = Node_x
        self.grid_ycor_node_number = Node_y
        self.node_number = self.grid_xcor_node_number*self.grid_ycor_node_number        # Total number of nodes without the sinks
        self.server_number = Sink_nr      # Num of server/sink nodes. Always first number-wise

        self.transmit_range = int(Transmission*self.grid_distance)
        self.min_distance_between_nodes = int(self.grid_distance/2)
        self.inter_server_range = max(self.transmit_range+1,self.grid_xcor_node_number*self.grid_ycor_node_number/self.server_number) 

        """====== Network initialization parameters ======="""
        self.max_find_good_position_time = 20       #Number of retries to assign random position
        random.seed(RNG_seed)

        """======= Network positions storage ======="""
        self.state_G = nx.Graph()
        self.state_G_no_sink = nx.Graph()
        self.state_xcor = []    #List of x coordinates of all nodes
        self.state_ycor = []    #List of y coordinates of all nodes
        self.state_link = []    #List of links in the network

        self.test = Test    #variable for that dictates what topology will be used
        self.x_reduce = X_reduce    #variable for sparse-dense dataset. Reduces the number of nodes on x axis
        self.y_reduce = Y_reduce    #variable for sparse-dense dataset. Reduces the number of nodes on y axis

    def setup_network(self):
        """
        PURPOSE: Creates a valid network
        """
        self.set_network_topology()
        while not self.all_nodes_connected():
            self.set_network_topology()

    def all_nodes_connected(self):
        """
        PURPOSE: Check if graph is connected

        RETURN: True if connected graph, False otherwise
        """
        for i in range(0, self.server_number+self.node_number):
            for j in range(0, self.server_number+self.node_number):
                check = nx.has_path(self.state_G, i, j)
                if not check:
                    return False
                if self.state_G.degree(i) < 2:
                    return False
        return True

    def set_network_topology(self):
        """
        PURPOSE: Create network topology
        """
        self.scatter_node_random_position()
        self.set_network_connectivity()
        self.set_graph()

    def scatter_node_random_position(self):
        """
        PURPOSE: Assigns the nodes at random positions on the grid

        DESCRIPTION:

        Clear previous positions of nodes.
        Arranges nodes according to the different preset topologies
        """
        self.state_xcor.clear()
        self.state_ycor.clear()
        styles = {
            0:self.struct_random,   #the random dataset
            1:self.struct_grid,     #the grid dataset
            2:self.struct_sym,
            3:self.struct_triangle,
            4:self.struct_square,
            5:self.struct_sparse_dense, #the sparse-dense dataset. preset density
            6:self.struct_net,      #the net dataset
            7:self.struct_cercles,
            8:self.struct_trend     #the sparse-dense dataset. variable density
        }
        func = styles.get(self.test, lambda:"Invalid input")
        func()
    
    def struct_grid(self):
        """
        Creates a grid style network
        """
        self.transmit_range = self.grid_distance #ensures consistency in grid pattern
        #place the servers first
        for k in range(0, self.server_number):
            self.state_xcor.append(k*self.grid_xcor_node_number*self.grid_distance/self.server_number+2)
            self.state_ycor.append(k*self.grid_ycor_node_number*self.grid_distance/self.server_number+2)
        #place the nodes
        for i in range(0, self.grid_xcor_node_number):
            for j in range(0, self.grid_ycor_node_number): 
                self.state_xcor.append(i*self.grid_distance)
                self.state_ycor.append(j*self.grid_distance) 

    def struct_random(self):
        """
        Creates a random style network
        Place servers first and check if they respect the minimum distance
        Then place nodes and check if they respect distance
        The node will function even if minimum distance could not be achieved for all nodes
        """
        for k in range(0, self.max_find_good_position_time**2):
            self.state_xcor.clear()
            self.state_ycor.clear()
            for i in range(0, self.server_number):
                self.state_xcor.append(0)
                self.state_ycor.append(0)
                self.state_xcor[-1] = random.randint(0, self.grid_distance*self.grid_xcor_node_number)
                self.state_ycor[-1] = random.randint(0, self.grid_distance*self.grid_ycor_node_number)
            good_server = self.check_server_distance()
            if good_server:
                break
            if k == self.max_find_good_position_time**2-1 :
                print("Server distance failed")
        for i in range(0, self.grid_xcor_node_number):
            for j in range(0, self.grid_ycor_node_number): 
                self.state_xcor.append(0)
                self.state_ycor.append(0)
                for k in range(0, self.max_find_good_position_time):
                    self.state_xcor[-1] = random.random() * self.grid_distance + i*self.grid_distance
                    self.state_ycor[-1] = random.random() * self.grid_distance + j*self.grid_distance
                    good_position = self.check_neighbor_distance(i*self.grid_ycor_node_number+j+self.server_number)
                    if good_position == 1:
                        break
                    if k == self.max_find_good_position_time-1:
                        print("Node distance failed", i*self.grid_ycor_node_number+j+self.server_number)

    def struct_sym(self):
        """
        Creates a symmetric style network
        """
        self.transmit_range = self.grid_distance
        self.state_xcor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2-1)*self.grid_distance/2+5)/2-10)
        self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2-1)*self.grid_distance/2+5)/2)
        self.state_xcor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2-1)*self.grid_distance/2+5)/2+10)
        self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2-1)*self.grid_distance/2+5)/2)
        for i in range(0, int(self.grid_xcor_node_number*self.grid_ycor_node_number/4)):
            self.state_xcor.append(i*self.grid_distance/2)
            self.state_ycor.append(i*self.grid_distance/2)
        for i in range(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4), int(self.grid_xcor_node_number*self.grid_ycor_node_number/2)):
            self.state_xcor.append((i)*self.grid_distance/2+5)
            self.state_ycor.append((i)*self.grid_distance/2+5)
        for i in range(0, int(self.grid_xcor_node_number*self.grid_ycor_node_number/4)):
            self.state_xcor.append(i*self.grid_distance/2)
            self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2)-i-1)*self.grid_distance/2+5)
        for i in range(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4), int(self.grid_xcor_node_number*self.grid_ycor_node_number/2)):
            self.state_xcor.append((i)*self.grid_distance/2+5)
            self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/2)-i-1)*self.grid_distance/2)

    def struct_triangle(self):
        """
        Creates a triangle style network
        """
        #self.transmit_range = self.grid_distance
        self.state_xcor.append(2*self.grid_distance-2.5)
        self.state_ycor.append(1*self.grid_distance+5)
        self.state_xcor.append(5*self.grid_distance+2.5)
        self.state_ycor.append(1*self.grid_distance+5)
        self.state_xcor.append(3*self.grid_distance+5)
        self.state_ycor.append(3*self.grid_distance+5)
        for i in range(0, int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)):
            if i%2 == 0:
                self.state_xcor.append(i*self.grid_distance/2)
                self.state_ycor.append(i*self.grid_distance/2)
                self.state_xcor.append((2*int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)-i)*self.grid_distance/2)
                self.state_ycor.append(i*self.grid_distance/2)
                self.state_xcor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)*self.grid_distance/2-2.5)
                self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)+i)*self.grid_distance/2)
            else:
                self.state_xcor.append((i+1)*self.grid_distance/2)
                self.state_ycor.append((i-1)*self.grid_distance/2)
                self.state_xcor.append((2*int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)-i+3)*self.grid_distance/2)
                self.state_ycor.append((i-1)*self.grid_distance/2)
                self.state_xcor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)+2)*self.grid_distance/2+2.5)
                self.state_ycor.append((int(self.grid_xcor_node_number*self.grid_ycor_node_number/3)+i-1)*self.grid_distance/2)

    def struct_square(self):
        """
        Creates a square style network
        """
        self.transmit_range = self.grid_distance
        self.state_xcor.append(self.grid_distance)
        self.state_ycor.append(self.grid_distance)
        self.state_xcor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4-1)*self.grid_distance)
        self.state_ycor.append(self.grid_distance)
        self.state_xcor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4-1)*self.grid_distance)
        self.state_ycor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4-1)*self.grid_distance)
        self.state_xcor.append(self.grid_distance)
        self.state_ycor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4-1)*self.grid_distance)
        for i in range(0, int(self.grid_xcor_node_number*self.grid_ycor_node_number/4)):
            self.state_xcor.append(i*self.grid_distance)
            self.state_ycor.append(0)
            self.state_xcor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4)*self.grid_distance)
            self.state_ycor.append(i*self.grid_distance)
            self.state_xcor.append((i+1)*self.grid_distance)
            self.state_ycor.append(int(self.grid_xcor_node_number*self.grid_ycor_node_number/4)*self.grid_distance)
            self.state_xcor.append(0)
            self.state_ycor.append((i+1)*self.grid_distance)

    def struct_trend(self):
        """
        Creates a variable density topology
        Servers are placed in one of 8 areas at the edges of the network.
        The reduced number of nodes is scattered around the area
        The redistributed nodes are scattered randomly in the center of the area
        """
        for k in range(0, self.max_find_good_position_time**2):
            self.state_xcor.clear()
            self.state_ycor.clear()
            for i in range(0, self.server_number):
                pos_decider = random.random()
                if pos_decider<0.125:
                    self.state_xcor.append((self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.25:
                    self.state_xcor.append(3*(self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.375:
                    self.state_xcor.append(7*(self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                elif pos_decider<0.5:
                    self.state_xcor.append(7*(self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append(3*(self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                elif pos_decider<0.625:
                    self.state_xcor.append(3*(self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append(7*(self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.75:
                    self.state_xcor.append((self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append(7*(self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.875:
                    self.state_xcor.append((self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append(3*(self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
                else:
                    self.state_xcor.append((self.grid_xcor_node_number-self.x_reduce-1)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-self.y_reduce-1)*self.grid_distance/4 +random.randint(-2, 2))
            good_server = self.check_server_distance()
            if good_server:
                break
            if k == self.max_find_good_position_time**2-1 :
                print("Server distance failed")
        for i in range(0, self.grid_xcor_node_number-self.x_reduce):
            for j in range(0, self.grid_ycor_node_number-self.y_reduce): 
                self.state_xcor.append(i*self.grid_distance+random.randint(-2,2))
                self.state_ycor.append(j*self.grid_distance+random.randint(-2,2))
        centroidx = 0.0
        centroidy = 0.0
        for i in range(0, self.server_number):
            centroidx += self.state_xcor[i]
            centroidy += self.state_ycor[i]
        centroidx /= self.server_number
        centroidy /= self.server_number
        for i in range(0, self.node_number-(self.grid_ycor_node_number-self.x_reduce)*(self.grid_xcor_node_number-self.y_reduce)):
            self.state_xcor.append(centroidx+2*(random.random()-0.5)*self.inter_server_range)
            self.state_ycor.append(centroidy+2*(random.random()-0.5)*self.inter_server_range)
    
    def struct_net(self):
        """
        Creates a net topology
        Works like the grid topology but each nodes can be placed randomly within an area, instead of a fixed position
        """
        for k in range(0, self.max_find_good_position_time**2):
            self.state_xcor.clear()
            self.state_ycor.clear()
            for i in range(0, self.server_number):
                self.state_xcor.append(random.randint(int(i*self.grid_distance*self.grid_xcor_node_number/self.server_number), int((i+1)*self.grid_distance*self.grid_xcor_node_number/self.server_number))+random.random())
                self.state_ycor.append(random.randint(int(i*self.grid_distance*self.grid_ycor_node_number/self.server_number), int((i+1)*self.grid_distance*self.grid_ycor_node_number/self.server_number))+random.random())
            good_server = self.check_server_distance()
            if good_server:
                break
            if k == self.max_find_good_position_time**2-1 :
                print("Server distance failed")
        for i in range(0, self.grid_xcor_node_number):
            for j in range(0, self.grid_ycor_node_number): 
                self.state_xcor.append(i*self.grid_distance+random.randint(-2,2))
                self.state_ycor.append(j*self.grid_distance+random.randint(-2,2))
    
    def struct_cercles(self):
        """
        Creates a cercles topology
        """
        #self.inter_server_range = 2*self.transmit_range+1
        size_variable = int(np.sqrt(self.grid_xcor_node_number*self.grid_ycor_node_number/self.server_number))
        for k in range(0, self.max_find_good_position_time**2):
            self.state_xcor.clear()
            self.state_ycor.clear()
            for i in range(0, self.server_number):
                self.state_xcor.append(random.randint(0, 2*int(self.inter_server_range)))
                self.state_ycor.append(random.randint(0, 2*int(self.inter_server_range)))
            good_server = self.check_server_distance()
            good_max_server = self.check_max_server_distance(2)
            if good_server and good_max_server:
                break
            if k == self.max_find_good_position_time**2-1 :
                print("Server distance failed")
        nodes_left = self.node_number
        self.min_distance_between_nodes = 4
        for i in range(0, self.server_number):
            if i == self.server_number-1:
                nodes_per_server = nodes_left
            else:
                nodes_per_server = random.randint(int(self.node_number/self.server_number/3),int(2*self.node_number/self.server_number))
                nodes_left -= nodes_per_server
            for j in range(0, nodes_per_server): 
                self.state_xcor.append(0)
                self.state_ycor.append(0)
                for k in range(0, self.max_find_good_position_time):
                    self.state_xcor[-1] = self.state_xcor[i]+2*(random.random()-0.5)*self.inter_server_range
                    self.state_ycor[-1] = self.state_ycor[i]+2*(random.random()-0.5)*self.inter_server_range
                    good_position = self.check_node_distance()
                    if good_position == 1:
                        break
                    if k == self.max_find_good_position_time-1:
                        print("Node distance failed", i, j)

    def struct_sparse_dense(self):
        """
        Creates fixed sparse_dense topology
        Places the servers at in one of 8 areas at the edges of the network
        Scatters the reduced number of node around the area
        The redistributed nodes are then scattered around the center area
        """
        for k in range(0, self.max_find_good_position_time**2):
            self.state_xcor.clear()
            self.state_ycor.clear()
            for i in range(0, self.server_number):
                pos_decider = random.random()
                if pos_decider<0.125:
                    self.state_xcor.append((self.grid_xcor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.25:
                    self.state_xcor.append(3*(self.grid_xcor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.375:
                    self.state_xcor.append(7*(self.grid_xcor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                elif pos_decider<0.5:
                    self.state_xcor.append(7*(self.grid_xcor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append(3*(self.grid_ycor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                elif pos_decider<0.625:
                    self.state_xcor.append(3*(self.grid_xcor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append(7*(self.grid_ycor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.75:
                    self.state_xcor.append((self.grid_xcor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                    self.state_ycor.append(7*(self.grid_ycor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                elif pos_decider<0.875:
                    self.state_xcor.append((self.grid_xcor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append(3*(self.grid_ycor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
                else:
                    self.state_xcor.append((self.grid_xcor_node_number-2)*self.grid_distance/8 +random.randint(-2, 2))
                    self.state_ycor.append((self.grid_ycor_node_number-2)*self.grid_distance/4 +random.randint(-2, 2))
            good_server = self.check_server_distance()
            if good_server:
                break
            if k == self.max_find_good_position_time**2-1 :
                print("Server distance failed")
        for i in range(0, self.grid_xcor_node_number-1):
            for j in range(0, self.grid_ycor_node_number-1): 
                self.state_xcor.append(i*self.grid_distance+random.randint(-2,2))
                self.state_ycor.append(j*self.grid_distance+random.randint(-2,2))
        centroidx = 0.0
        centroidy = 0.0
        for i in range(0, self.server_number):
            centroidx += self.state_xcor[i]
            centroidy += self.state_ycor[i]
        centroidx /= self.server_number
        centroidy /= self.server_number
        for i in range(0, self.node_number-(self.grid_ycor_node_number-1)*(self.grid_xcor_node_number-1)):
            self.state_xcor.append(centroidx+2*(random.random()-0.5)*self.inter_server_range)
            self.state_ycor.append(centroidy+2*(random.random()-0.5)*self.inter_server_range)

    def check_neighbor_distance(self, node_id):
        """
        PURPOSE: Checks if the minimum distance between nodes is respected.

        RETURN: 1 if min distance was respected, 0 otherwise
        """
        if(node_id == self.server_number):
            return 1
        ax = self.state_xcor[node_id]
        ay = self.state_ycor[node_id]
        for k in range(0, node_id):
            bx = self.state_xcor[k]
            by = self.state_ycor[k]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance < self.min_distance_between_nodes:
                return 0
        return 1
    
    def check_node_distance(self):
        """
        PURPOSE: Checks if the minimum distance between nodes is respected. Used for a deprecated dataset

        RETURN: 1 if min distance was respected, 0 otherwise
        """
        ax = self.state_xcor[-1]
        ay = self.state_ycor[-1]
        for k in range(0, len(self.state_xcor)-1):
            bx = self.state_xcor[k]
            by = self.state_ycor[k]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if distance < self.min_distance_between_nodes:
                return 0
        return 1

    def check_server_distance(self):
        """
        PURPOSE: Checks if the minimum distance between servers is respected

        RETURN: 1 if min distance was respected, 0 otherwise
        """
        for i in range(0, self.server_number-1):
            ax = self.state_xcor[i]
            ay = self.state_ycor[i]
            for k in range(i+1, self.server_number):
                bx = self.state_xcor[k]
                by = self.state_ycor[k]
                distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if distance < self.inter_server_range:
                    return 0
        return 1

    def check_max_server_distance(self, max_size):
        """
        PURPOSE: Checks if the second round of nodes is appropriately far enough from the servers. Used for a deprecated dataset.

        RETURN: 1 if distance was respected, 0 otherwise
        """
        for i in range(0, self.server_number-1):
            ax = self.state_xcor[i]
            ay = self.state_ycor[i]
            for k in range(i+1, self.server_number):
                bx = self.state_xcor[k]
                by = self.state_ycor[k]
                distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if distance > max_size*self.inter_server_range:
                    return 0
        return 1

    def set_network_connectivity(self):
        """
        PURPOSE: Find all possible connections between nodes

        DESCRIPTION:
        Clear previous network link state list
        For each node check every other node if they are in transmission range.
        If yes append a 1, otherwise a 0.
        Append each node's list to the network one
        """
        self.state_link.clear()
        for i in range(0, self.node_number+self.server_number):
            node_link = []
            for j in range(0, self.node_number+self.server_number):
                distance = ((self.state_xcor[i]-self.state_xcor[j])**2 +
                            (self.state_ycor[i]-self.state_ycor[j])**2)**0.5
                if (i != j) and (distance <= self.transmit_range):
                    node_link.append(1)
                else:
                    node_link.append(0)
            self.state_link.append(node_link)

    def set_graph(self):
        """
        PURPOSE: Create the graph

        DESCRIPTION:
        Clear previous graph.
        Add all nodes and their positions
        If there is a bidirectional link between two nodes, add it
        Copy the graph and remove the sinks for the no sink graph
        """
        self.state_G.clear()
        for i in range(0, self.node_number+self.server_number):
            self.state_G.add_node(i, pos=(self.state_xcor[i], self.state_ycor[i]))
        for i in range(0, self.node_number+self.server_number):
            for j in range(i, self.node_number+self.server_number):
                if self.state_link[i][j] == 1 and self.state_link[j][i] == 1:
                    self.state_G.add_edge(i, j)
        self.state_G_no_sink = self.state_G.copy()
        for k in range(0, self.server_number):
            self.state_G_no_sink.remove_node(k)

    def draw_network(self):
        """
        PURPOSE: Create visualization of the network
        """
        plt.figure("Network")
        pos = nx.get_node_attributes(self.state_G, 'pos')
        nx.draw(self.state_G, pos, with_labels=True, cmap=plt.get_cmap('Accent'),
                node_color='deepskyblue', node_size=170)
        plt.show()

    def store_network(self):
        """
        PURPOSE: Store the coordinates of the nodes and the network link state list as numpy files
        """
        np.save("x_cor.npy", np.array(self.state_xcor[:]))
        np.save("y_cor.npy", np.array(self.state_ycor[:]))
        np.save("link.npy", np.array(self.state_link[:]))

    def reuse_network(self, prefix):
        """
        PURPOSE: Load the saved node coordinates and network link state list, then create the graph with these. Needs a prefix with the address on disk of files.
        """
        self.state_xcor = np.load(prefix+"x_cor.npy").tolist()
        self.state_ycor = np.load(prefix+"y_cor.npy").tolist()
        self.state_link = np.load(prefix+"link.npy").tolist()
        self.set_graph()


class NetworkGNN(Network):
    """
    IoT network environment
    """
    def __init__(self, type, nodes_x, nodes_y, sink_nr, reuse, rng_seed, prefix = None, transmission = 1.5, x_reduce = 0, y_reduce = 0):
        super().__init__(type, nodes_x, nodes_y, sink_nr, rng_seed, transmission, x_reduce, y_reduce)

        """======= Network initialization ======="""
        #checks if there is need to load a network
        if reuse:
            self.reuse_network(prefix)
        else:
            self.setup_network()
        self.node_object = {}       #dictionary that holds all nodes
        for i in range(0, self.node_number+self.server_number):
            self.node_object[str(i)] = Node(i)
            self.node_object[str(i)].clusters = np.zeros(self.server_number)
        self.hops = []  #initial hop distance of nodes
        self.initial_node_number = self.node_number
        self.time_counter = 1   #the number of the initial timeslot
        self.sim_duration = 100     #default duration of simulation in number of timeslots
        self.clustered_G = nx.Graph()
        self.clustered_G = self.state_G.copy()
        self.init_node()
        
    def run_network(self, sim_duration):
        """
        PURPOSE: Run the network for the indicated number of timeslots
        """
        self.sim_duration = sim_duration
        self.clear_traffic()
        if self.assign_timeslots() == "disconnected":
            return "disconnected"
        if not self.check_connectivity():
            return "disconnected"
        for i in range(1, self.sim_duration+1):
            #print("Timeslot ",self.time_counter,":")
            self.time_counter +=1
            self.change_awake(i)
            self.step(i)
        #prints all the statistics of the simulation
        #self.print_stats()

    def clear_traffic(self):
        """
        PURPOSE: Cleares the traffic variables of all nodes
        """
        for i in range(0, self.node_number+self.server_number):
            self.node_object[str(i)].latency.clear()
            self.node_object[str(i)].traffic_sent = 0
            self.node_object[str(i)].traffic_received = 0
            self.node_object[str(i)].sent_to.clear()
            self.node_object[str(i)].send_time.clear()
            self.node_object[str(i)].receive_time.clear()

    def step(self, timeslot):
        """
        PURPOSE: Runs one timeslot of the network
        """
        for i in range(self.server_number, self.server_number+self.node_number):
            if timeslot in self.node_object[str(i)].send_time:
                result = self.node_object[str(i)].send_time.index(timeslot)
                self.node_object[str(i)].traffic_sent += 1
                self.node_object[str(self.node_object[str(i)].sent_to[result])].traffic_received += 1
                

    def init_node(self):
        """
        PURPOSE: Initializes most of the node's information.
        Find the shortest path to a sink for each node, save the path, 
        save the hop distance and assign initial clusters
        """
        self.hops.clear()
        for i in range(self.server_number, self.server_number+self.node_number):
            self.node_object[str(i)].hop_dis.clear()
            self.node_object[str(i)].sink.clear()
            self.node_object[str(i)].path.clear()
            for k in range(0, self.server_number):
                path = self.find_route_sink(i,k)
                if len(path)>0:
                    self.node_object[str(i)].hop_dis.append(len(path) - 1)
                    self.node_object[str(i)].sink.append(k)
                    self.node_object[str(i)].path.append(path)
            self.hops.append(self.node_object[str(i)].hop_dis)
            #assign a random period to each node
            self.node_object[str(i)].period = random.randint(max(20, int(2*self.node_number/self.server_number)), self.sim_duration)
        self.assign_initial_clusters()

    def assign_initial_clusters(self):
        """
        PURPOSE: Cluster the network initially based on who is the closest node
        """
        for i in range(self.server_number, self.server_number+self.node_number):
            self.node_object[str(i)].clusters = np.zeros(self.server_number)
            self.node_object[str(i)].clusters[np.argmin(self.node_object[str(i)].hop_dis)] = 1
        for i in range(0, self.server_number):
            self.node_object[str(i)].clusters[i] = 1

    def assign_timeslots(self):
        """
        PURPOSE: Assign the communication schedule for each node on a path towards the sink.
        The assignment goes from closest to the servers to farthest
        """
        for i in range(0, self.node_number+self.server_number):
            self.node_object[str(i)].awake_timeslots = np.zeros(self.sim_duration)

        hop = 1
        hopmax = 2
        while hop <= hopmax:
            for i in range(self.server_number, self.server_number+self.node_number):
                if self.node_object[str(i)].hop_dis == []:
                    return "disconnected"
                hopmax = max(hopmax, min(self.node_object[str(i)].hop_dis))
                if min(self.node_object[str(i)].hop_dis) == hop:
                    time_sent = 0
                    time_received = 0
                    timeslot_counter = 1

                    path = self.node_object[str(i)].path[np.argmin(self.node_object[str(i)].hop_dis)]
                    while(timeslot_counter <= self.sim_duration):
                        for j in range(0, len(path)-1):
                            a = path[j]
                            b = path[j+1]
                            timeslot_counter = self.schedule_communication(a, b, timeslot_counter)
                            if timeslot_counter > self.sim_duration:
                                break
                            if a == path[0]:
                                time_sent = self.node_object[str(a)].send_time[-1]
                            if b == path[len(path)-1]:
                                time_received = self.node_object[str(b)].receive_time[-1]
                                self.node_object[str(i)].latency.append(time_received-time_sent+1)
                        if timeslot_counter <= self.sim_duration:
                            timeslot_counter = self.node_object[str(i)].send_time[-1] + self.node_object[str(i)].period
            hop +=1

    def schedule_communication(self, s, r, time):
        """
        PURPOSE: Schedule periodic communication between two nodes
        """
        local_time = time
        freq = 0
        while local_time <= self.sim_duration:
            cond1 = self.node_object[str(s)].awake_timeslots[local_time-1]  #should be 0
            cond2 = self.node_object[str(r)].awake_timeslots[local_time-1]  #should be 0
            cond3 = 1
            if not cond1 and not cond2:
                cond3 = 0
                for i in self.state_G.neighbors(s):
                    if local_time in self.node_object[str(i)].receive_time:
                        cond3 = 1
                        break
                for i in self.state_G.neighbors(r):
                    if local_time in self.node_object[str(i)].send_time:
                        cond3 = 1
                        break
            if not cond3:
                self.node_object[str(s)].awake_timeslots[local_time-1] = 1
                self.node_object[str(r)].awake_timeslots[local_time-1] = 1
                self.node_object[str(s)].send_time.append(local_time)
                self.node_object[str(s)].sent_to.append(r)
                self.node_object[str(r)].receive_time.append(local_time)
                break
            else:
                local_time += 1
        return local_time

    def change_awake(self, timeslot):
        """
        PURPOSE: Changes the awake status of nodes for the current timeslot
        """
        for i in range(self.server_number, self.server_number+self.node_number):
            if self.node_object[str(i)].awake_timeslots[timeslot-1]:
                self.node_object[str(i)].sleep_mode = 0
            else:
                self.node_object[str(i)].sleep_mode = 1

    def print_stats(self):
        """
        PURPOSE: Print the statistics of each node
        """
        for i in range(0, self.node_number+self.server_number):
            print("Node: ",i, " Period: ", self.node_object[str(i)].period)
            print("Awake timeslots", self.node_object[str(i)].awake_timeslots)
            print("Traffic sent: ",self.node_object[str(i)].traffic_sent, " Traffic received: ", self.node_object[str(i)].traffic_received)
            print("Sent times: ",self.node_object[str(i)].send_time, " Receive times ", self.node_object[str(i)].receive_time)
            print("Sent to:", self.node_object[str(i)].sent_to)
            if i >= self.server_number:
                print("Latency:", self.node_object[str(i)].latency, " Maximum latency: ", max(self.node_object[str(i)].latency))
            print()

    def find_route(self, s, t):
        """
        PURPOSE: Find the shortest route between two nodes

        RETURN: route from source to target
        """
        check = nx.has_path(self.state_G_no_sink, source=s, target=t)
        if check:
            path = nx.dijkstra_path(self.state_G_no_sink, source=s, target=t)
        else:
            path = []
        return path

    def find_route_sink(self, s, t):
        """
        PURPOSE: Find the shortest route to the sink

        RETURN: route from source to sinks
        """
        check = nx.has_path(self.clustered_G, source=s, target=t)
        if check:
            path = nx.dijkstra_path(self.clustered_G, source=s, target=t)
        else:
            path = []
        return path

    def check_connectivity(self):
        """
        PURPOSE: Check if graph is connected

        RETURN: True if connected graph, False otherwise
        """
        for i in range(0, self.server_number+self.node_number):
            for j in range(0, self.server_number+self.node_number):
                check1 = nx.has_path(self.clustered_G, i, j)
                check2 = True
                check3 = nx.has_path(self.state_G, i, j)
                if i >= self.server_number and j >= self.server_number:
                    check2 = nx.has_path(self.state_G_no_sink, i, j)
                if not check1 or not check2 or not check3:
                    return False
        return True

    def delete_node(self,k):
        """
        PURPOSE: Delete a node in the network
        """
        self.state_G_no_sink.remove_node(k)
        self.state_G.remove_node(k)
        self.clustered_G.remove_node(k)
        self.node_number = self.node_number - 1

    def add_node(self,i, x, y):
        """
        PURPOSE: Add a node in the network and initializes it
        """
        self.node_object[str(i)] = Node(i)
        self.node_object[str(i)].clusters = np.zeros(self.server_number)
        self.node_number = self.node_number + 1
        xcor = random.randint(0, self.grid_distance*self.grid_xcor_node_number)
        ycor = random.randint(0, self.grid_distance*self.grid_ycor_node_number)
        if x != -1 and y != -1:
            xcor = x
            ycor = y
        self.state_G.add_node(i, pos=(xcor, ycor))
        for j in self.state_G.nodes:
                distance = ((xcor-self.state_G.nodes[j]["pos"][0])**2 +
                            (ycor-self.state_G.nodes[j]["pos"][1])**2)**0.5
                if (i != j) and (distance <= self.transmit_range):
                    self.state_G.add_edge(i, j)

        aux_G = nx.Graph()
        aux_G.add_nodes_from(sorted(self.state_G.nodes(data=True)))
        aux_G.add_edges_from(self.state_G.edges(data=True))
        self.state_G = aux_G.copy()
        self.state_G_no_sink = self.state_G.copy()
        self.clustered_G = self.state_G.copy()
        for j in range(0, self.server_number):
            self.state_G_no_sink.remove_node(j)
        for k in range(0, self.server_number):
            path = self.find_route_sink(i,k)
            if len(path)>0:
                self.node_object[str(i)].hop_dis.append(len(path) - 1)
                self.node_object[str(i)].sink.append(k)
                self.node_object[str(i)].path.append(path)
        self.hops[i-self.server_number] = self.node_object[str(i)].hop_dis
        self.node_object[str(i)].clusters[np.argmin(self.node_object[str(i)].hop_dis)] = 1

    def cluster_network(self, configuration):
        """
        PURPOSE: Implement the clusters from the given configuration
        """
        for i in range(self.server_number, self.server_number+self.node_number):
            self.node_object[str(i)].clusters = configuration[i-self.server_number]
            self.node_object[str(i)].hop_dis.clear()
            self.node_object[str(i)].sink.clear()
            self.node_object[str(i)].path.clear()
            for j in range(0, len(configuration[i-self.server_number])):
                if configuration[i-self.server_number][j]:
                    path = self.find_route_sink(i, j)
                    if len(path)>0:
                        self.node_object[str(i)].hop_dis.append(len(path) - 1)
                        self.node_object[str(i)].sink.append(j)
                        self.node_object[str(i)].path.append(path)

    def network_stats(self):
        """
        PURPOSE: Compute and pass the load information and the max latency of each cluster
        """
        latency = np.zeros(self.server_number)
        clusters_load = []
        server_loads = []
        for i in range(0, self.server_number):
            server_loads.append(self.node_object[str(i)].traffic_sent + self.node_object[str(i)].traffic_received)
            cluster_sum = 0
            for j in range(self.server_number, self.server_number+self.node_number):
                if self.node_object[str(j)].clusters[i] == 1:
                    cluster_sum = cluster_sum + (self.node_object[str(j)].traffic_sent + self.node_object[str(j)].traffic_received)
                    if not self.node_object[str(j)].latency:
                        latency[i] = 5000
                    elif max(self.node_object[str(j)].latency)>latency[i]:
                        latency[i] = max(self.node_object[str(j)].latency)
            clusters_load.append(cluster_sum)
        load_avg = np.mean(clusters_load)
        server_avg = np.mean(server_loads)
        return server_avg, server_loads, latency

    def return_config(self):
        """
        PURPOSE: returns the configuration of the current network
        """
        config = []
        for i in range(0, self.node_number+self.server_number):
            config.append(self.node_object[str(i)].clusters)
        return config

    def draw_clusters(self):
        """
        PURPOSE: Draw the clustered network
        """
        colours = []
        for i in range(0, self.node_number+self.server_number):
            c_node=0
            for j in range(0,len(self.node_object[str(i)].clusters)):
                c_node = c_node + self.node_object[str(i)].clusters[j] * 2**j
            colours.append(c_node)
        plt.figure("Clusters")
        pos = nx.get_node_attributes(self.clustered_G, 'pos')
        nx.draw(self.clustered_G, pos, with_labels=True, cmap=plt.get_cmap('tab20'),
                node_color=colours, node_size=170)
        plt.show()

    def draw_timeslot(self, timeslot):
        """
        PURPOSE: Create visualization of the network at a specific timeslot
        """
        plt.figure("Timeslot "+str(timeslot))
        nodes = []
        edges = list(self.state_G.edges())
        edge_color = ['k']*len(edges)
        for i in range(0, self.node_number+self.server_number):
            if timeslot in self.node_object[str(i)].send_time:
                result = self.node_object[str(i)].send_time.index(timeslot)
                nodes.append(i)
                nodes.append(self.node_object[str(i)].sent_to[result])
                edges.append((i, self.node_object[str(i)].sent_to[result]))
                edge_color.append('r')
        pos = nx.get_node_attributes(self.state_G, 'pos')
        nx.draw(self.state_G, pos, with_labels=True, nodelist = nodes, edgelist = edges, edge_color=edge_color, cmap=plt.get_cmap('Accent'),
            node_color='deepskyblue', node_size=170, width = 2.0)
        plt.show()