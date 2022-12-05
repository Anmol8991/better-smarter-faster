import random
from collections import deque

class Cycle:
    def __init__(self, c):
        self._c = c
    
    def set_index(self, idx):
        self._index = idx

    def next(self, k):
        self._index += k
        if self._index >= len(self._c):
            self._index = (self._index % 49) - 1
        return self._c[self._index]

    def previous(self, k):
        i = self._index
        self._index -= k
        if self._index < 0:
            self._index = len(self._c) - (k - i)
        return self._c[self._index]

class Graph:
    def __init__(self) -> None:
        """
        Inilialize the graph dict
        """
        self.graph = {}

    def add_edge(self, edge, graph_type="undirected") -> None:
        """
        Add an Edge to self.graph
        """
        if isinstance(edge, tuple):
            if edge[0] not in self.graph:
                self.graph[edge[0]] = []
                self.graph[edge[0]].append(edge[1])
            elif edge[1] not in self.graph[edge[0]]:
                self.graph[edge[0]].append(edge[1])

            if graph_type == "undirected":
                if edge[1] not in self.graph:
                    self.graph[edge[1]] = []
                    self.graph[edge[1]].append(edge[0])
                elif edge[0] not in self.graph[edge[1]]:
                    self.graph[edge[1]].append(edge[0])
        else:
            raise TypeError("The value {} is not a tuple".format(edge))

    def add_edges(self, edges, graph_type="undirected") -> None:
        """
        Add Edges to self.graph
        """
        for edge in edges:
            if isinstance(edge, tuple):
                if edge[0] not in self.graph:
                    self.graph[edge[0]] = []
                    self.graph[edge[0]].append(edge[1])
                elif edge[1] not in self.graph[edge[0]]:
                    self.graph[edge[0]].append(edge[1])
                else:
                    print("Edge between {} and {} already exists in the graph!".format(edge[0], edge[1]))

                if graph_type == "undirected":
                    if edge[1] not in self.graph:
                        self.graph[edge[1]] = []
                        self.graph[edge[1]].append(edge[0])
                    elif edge[0] not in self.graph[edge[1]]:
                        self.graph[edge[1]].append(edge[0])
                    else:
                        print("Edge between {} and {} already exists in the graph!".format(edge[1], edge[0]))
            else:
                raise TypeError("The value {} is not a tuple".format(edge))

    def add_node(self, node) -> None:
        """
        Add a single node to the graph
        """
        if node not in self.graph:
            self.graph[node] = []

    def add_nodes(self, nodes) -> None:
        """
        Add the nodes to the graph
        """
        for node in nodes:
            if node not in self.graph:
                self.graph[node] = []
            else:
                print("Node {} already exists in the graph!".format(node))

    def node_degree(self, node=None) -> int:
        """
        Return a degree for a specific node
        """
        if node in self.graph:
            return len(self.graph[node])
        else:
            raise ValueError("Node {} not in graph".format(node))
    
    def node_degrees(self, nodes=None) -> dict:
        """
        Return the passed nodes and their respective degrees in the form of a dictionary
        """
        if self.graph:
            degree_dict = {}

            for node in nodes:
                if node not in self.graph:
                    raise ValueError("Node {} is not present in the graph".format(node))
                else:
                    degree_dict[node] = len(self.graph[node])

            return degree_dict

        else:
            raise Exception("self.graph is empty")

    def degrees(self) -> dict:
        """
        Return all the nodes and their respective degrees in the form of a dictionary
        """
        if self.graph:
            degree_dict = {}
            for key in self.graph:
                degree_dict[key] = len(self.graph[key])
            return degree_dict
        else:
            raise Exception("self.graph is empty")

    def number_of_edges(self) -> int:
        """
        Return the number of edges in the graph
        """
        count = 0
        
        for node in self.graph:
            count += len(self.graph[node])
        
        return int(count/2)
    
    def number_of_nodes(self) -> int:
        """
        Return the number of nodes in the graph
        """
        return len(self.graph)

    def neighbors(self, node=None) -> list:
        """
        Return all neighbors of the passed node
        """
        if node in self.graph:
            return self.graph[node]
        else:
            raise ValueError("Node {} not in graph".format(node))
    
    def node_list(self) -> list:
        """
        Returns a list of all nodes in the graph
        """
        return list(self.graph)

    def create_path(self, parent_dict, current_node):
        """
        Generate the actual path selected by the shortest_path algorithm
        """
        totalPath = [current_node]

        while current_node in parent_dict:
            current_node = parent_dict[current_node]
            totalPath.append(current_node)

        return totalPath[::-1]

    def shortest_path(self, start, goal) -> list:
        """
        Returns a shortest path from start to goal in an undirected graph
        """
        if start in self.graph and goal in self.graph:

            if start == goal:
                return [start]

            queue = deque()
            closed_list = []

            # For node n, parent[n] is the node immediately preceding it on the path
            parent = {}

            # Initially we start the open queue and the closed list with the start node
            queue.append(start)
            closed_list.append(start)

            # Main while loop
            while queue:

                # Pop current off open list, add to closed list
                current_node = queue.popleft()

                if current_node == goal:
                    # Create and return the path
                    return self.create_path(parent, current_node)

                for neighbor in self.neighbors(current_node):
                    # We need to make sure we are not reprocessing closed list nodes
                    if neighbor not in closed_list:
                        # We store the parent of the current_node
                        parent[neighbor] = current_node

                        # Then we want to make sure we are adding this neighbor to the open queue
                        if neighbor not in queue:
                            queue.append(neighbor)

                        # And finally, since we have touched this neighbor, we want to also add it to the closed list
                        closed_list.append(neighbor)

            # End of while

        else:
            raise ValueError("One or both of the nodes {} and {} are not in the graph".format(start, goal))

    def shortest_path_length(self, start, goal) -> int:
        """
        Returns a shortest path distance from start to goal in an undirected graph
        """
        if start in self.graph and goal in self.graph:
            # Just find the length of the shortest path
            return len(self.shortest_path(start, goal)) - 1
        else:
            raise ValueError("One or both of the nodes {} and {} are not in the graph".format(start, goal))

    def create_graph_tcol(self, size) -> None:
        """
        Create a new graph for the circle of life project
        """
        vertices = [i for i in range(size)]

        # Initialize a cyclic iterator over vertices
        cyc_point = Cycle(vertices)

        # Set its start index to 0
        cyc_point.set_index(0)

        # Set normal edges
        for i in range(size):
            self.add_edge((i, cyc_point.next(1)))

        # Reset index
        cyc_point.set_index(0)

        # Add more edges as per the logic in the write-up
        tries = 0
        max_tries = 10 ** 3

        while tries < max_tries:
            tries += 1

            # print("Try {}".format(tries))
            degree_two_nodes = {node : val for node, val in (self.degrees()).items() if val < 3}

            # print(degree_two_nodes)

            random_idx = random.randint(0, size - 1) # pulls a random index

            if random_idx not in degree_two_nodes:
                continue

            cyc_point.set_index(random_idx)

            if random.random() <= 0.5:
                positive_idx = cyc_point.next(random.randint(2, 5))
                if positive_idx in degree_two_nodes:
                    self.add_edge((random_idx, positive_idx))

            else:
                negative_idx = cyc_point.previous(random.randint(2, 5))
                if negative_idx in degree_two_nodes:
                    self.add_edge((random_idx, negative_idx))

        # End of while
