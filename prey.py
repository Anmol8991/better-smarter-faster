import random
from graph_utils import Graph
from agent import Agent

class Prey:
    
    def __init__(self, graph: Graph, agent: Agent,pos) -> None:
        """
        Spawn at some node while initializing the prey object
        """
        possible_pos = list(graph.graph)
        possible_pos.remove(agent.node) # we don't want to spawn where the agent is at

        self.pos = pos
        # random.choice(possible_pos)
    
    def move_prey(self, graph: Graph) -> None:
        """
        Randomly move the prey to one of its neighbors or stay in the same spot
        """
        possible_pos = [self.pos]
        possible_pos.extend(graph.neighbors(self.pos))

        # Just set the current position with equal random chance
        self.pos = random.choice(possible_pos)
