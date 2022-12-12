import math
import random
import numpy as np
import probability as prob
from graph_utils import Graph

class Agent:
    """
    Object that has stuff on the agent
    """
    def __init__(self, name, graph: Graph,pos) -> None:
        """
        Initialization
        """
        self.name = name

        # Spawn Agent at a random node (other than the prey and predator spawn nodes)
        self.node = pos
        # random.choice(graph.node_list())

        total_nodes = len(graph.node_list())

        # Initialize prey probability matrix to (1/49)s
        self.prey_beliefs = []

        for _ in range(total_nodes):
            self.prey_beliefs.append(1 / (total_nodes - 1))

        self.prey_beliefs[self.node] = 0.0

        # Initialize predator probability matrix to 0s
        self.pred_beliefs = []

        for _ in range(total_nodes):
            self.pred_beliefs.append(0.0)

        # For partial prey
        self.prey_transition_matrix = np.zeros((total_nodes, total_nodes), dtype=float)

        # For partial predator
        self.pred_transition_matrix = np.zeros((total_nodes, total_nodes), dtype=float)

        
        print("Initial Agent Position: " + str(self.node))

    def prey_move_sim(self, graph: Graph, prey_pos, agent_future_pos) -> int:
        """
        The function simulates the predator's movement pattern (used in Agents 2, 4, 6 and 8)
        """
        possible_positions = [prey_pos]
        possible_positions.extend(graph.neighbors(prey_pos))

        path_lengths = {}

        for prey in possible_positions:
            path_lengths[prey] = graph.shortest_path_length(prey, agent_future_pos)

        # Just set the current position to the best neighbor (and break ties randomly if there is more than one)
        return round(sum(list(path_lengths.values())) / len(possible_positions))

    def predator_move_sim(self, graph: Graph, pred_pos, agent_future_pos, distracted=False) -> int:
        """
        The function simulates the predator's movement pattern (used in Agents 2, 4, 6 and 8)
        """
        min_path_len = math.inf

        for neighbor in graph.neighbors(pred_pos):
            
            path_length = graph.shortest_path_length(neighbor, agent_future_pos)
            
            if path_length < min_path_len:
                min_path_len = path_length # this is done because we have a clear cut best neighbor

        # Just set the current position to the best neighbor (and break ties randomly if there is more than one)
        return path_length

    def update_prey_trans_matrix(self, graph: Graph) -> None:
        """
        Updates prey transition matrix P(i, j) = Probability of prey at node i at (t + 1) given prey was at node j at (t) i.e. [j -> i]
        """
        graph_nodes = graph.node_list()

        for jnode in graph_nodes:

            neighbors = list(graph.neighbors(jnode))
            prob_each_neighbor = 1 / (len(neighbors) + 1)
    
            # When prey stays at the same node
            self.prey_transition_matrix[jnode, jnode] = prob_each_neighbor

            # When prey moves to a different node
            for inode in neighbors:
                self.prey_transition_matrix[inode, jnode] = prob_each_neighbor

    def update_pred_trans_matrix(self, graph: Graph) -> None:
        """
        Updates predator transition matrix P(i, j) = Probability of predator at node i at (t + 1) given predator was at node j at (t) i.e. [j -> i]
        (for a non-distracted predator)
        """
        graph_nodes = graph.node_list()

        self.pred_transition_matrix = np.zeros((len(graph_nodes), len(graph_nodes)))

        for jnode in graph_nodes:
            min_path_length = math.inf
            likely_nodes_plan = []

            for neighbor in graph.neighbors(jnode):
                
                path_length = graph.shortest_path_length(jnode, self.node)
                
                if path_length <= min_path_length:
                    if path_length < min_path_length:
                        min_path_length = path_length
                    likely_nodes_plan.append(neighbor)

            for inode in likely_nodes_plan:
                self.pred_transition_matrix[inode, jnode] = 1 / len(likely_nodes_plan)


    def update_dist_pred_trans_matrix(self, graph: Graph) -> None:
        """
        Updates predator transition matrix P(i, j) = Probability of predator at node i at (t + 1) given predator was at node j at (t) i.e. [j -> i]
        (for a distracted predator)
        """
        graph_nodes = graph.node_list()

        self.pred_transition_matrix = np.zeros((len(graph_nodes), len(graph_nodes)))

        for jnode in graph_nodes:
            min_path_length = math.inf
            likely_nodes_plan = []
            likely_nodes_random = graph.neighbors(jnode)

            for neighbor in graph.neighbors(jnode):
                
                path_length = graph.shortest_path_length(jnode, self.node)
                
                if path_length <= min_path_length:
                    if path_length < min_path_length:
                        min_path_length = path_length
                        likely_nodes_plan = [neighbor]
                    else:
                        likely_nodes_plan.append(neighbor)

            for inode in likely_nodes_random:
                # When predator moves to a 'good' node (whether it is because of focus or distraction) 
                if inode in likely_nodes_plan:
                    self.pred_transition_matrix[inode, jnode] = 0.6 / len(likely_nodes_plan) + 0.4 / len(likely_nodes_random)
                else: # when predator is distracted and goes to a 'bad' node
                    self.pred_transition_matrix[inode, jnode] = 0.4 / len(likely_nodes_random)

    # Agent U_PARTIAL 
    def get_policy(self, graph, pred_pos, probability_vector, transition_model, optimal_state_utility,immediate_reward, partial_utility):

        
        utility = {}


        actions = graph.neighbors(self.node)
        cur_state = (self.node, pred_pos)

        # Shortest Distance from Agent to Predator
        agent_to_predator = graph.shortest_path_length(self.node , pred_pos)

        if agent_to_predator == 1:
            agent_neighbours = graph.neighbors(self.node)
            dis_to_pred = {}
            for i in agent_neighbours:
                dis_to_pred[i] = graph.shortest_path_length(i , pred_pos)

            max_val = max(dis_to_pred.values())
            best_neighbors = [neighbor for neighbor, dist in dis_to_pred.items() if dist == max_val]  
                
            return random.choice(best_neighbors) 
        

        for a in actions:
            utility[a] = 0

            for (p, s1) in transition_model[cur_state][a]:

                # Update post agent moves
                # probability_vector = prob.survey(probability_vector, agent.node, s1[1])

                # Update post prey moves
                updated_prey_prob_vector = np.matmul(self.prey_transition_matrix, np.array(probability_vector))
                updated_prey_prob_vector = updated_prey_prob_vector.tolist()

                
                expected_utility = 0
                for i in range(len(updated_prey_prob_vector)):
                    expected_utility += updated_prey_prob_vector[i]*optimal_state_utility[(a,i,s1[1])] # a - agent position, i - prey position, s1[1] - predator position
                
                utility[a] += p * expected_utility


        # UPDATE utility
        state = (self.node, tuple(probability_vector), pred_pos)

        partial_utility[state] = immediate_reward[cur_state] +  min(utility.values())


        lowest_utility_neighbors = [action for action, u in utility.items() if u == min(utility.values())]      

        # Breaking ties at random
        next_pos = random.choice(lowest_utility_neighbors)

        return next_pos




    # Agent V_MODEL
    def get_policy_model( graph, state, transition_model, predict):

        
        expected_utility = {}


        actions = graph.neighbors(state[0])

        # Shortest Distance from Agent to Predator
        agent_to_predator = graph.shortest_path_length(state[0], state[2])

        if agent_to_predator == 1:
            agent_neighbours = graph.neighbors(state[0])
            dis_to_pred = {}
            for i in agent_neighbours:
                dis_to_pred[i] = graph.shortest_path_length(i , state[2])

            max_val = max(dis_to_pred.values())
            best_neighbors = [neighbor for neighbor, dist in dis_to_pred.items() if dist == max_val]  
                
            return random.choice(best_neighbors) 

        for a in actions:
            expected_utility[a] = 0
            for (p, s1) in transition_model[state][a]:
                expected_utility[a] += p * predict([[[graph.shortest_path_length(s1[0], s1[1])/50, graph.shortest_path_length(s1[0], s1[2])/50]]])[0][0][0] * max_util
    

        # Set Minimum reward value out of all the actions as the utility of the current state
        min_val = min(expected_utility.values())

        # # UPDATE utility
        # state_utility[state] = immediate_reward[state] +   min_val


        lowest_utility_neighbors = [action for action, utility in expected_utility.items() if utility == min_val]      

        # Breaking ties at random
        next_pos = random.choice(lowest_utility_neighbors)

        return next_pos



        

    def move_1(self, graph: Graph, prey_pos, pred_pos) -> None:
        """
        This function moves the agent 1 according to the strategy mentioned in the write up
        """
        # Shortest Distance from Agent to prey
        agent_to_prey = graph.shortest_path_length(self.node , prey_pos)
        
        # Shortest Distance from Agent to Predator
        agent_to_predator = graph.shortest_path_length(self.node , pred_pos)

        # List of Neighbors of Agent
        agent_neighbours = graph.neighbors(self.node)

        # Distance of shortest path from each neighbour to prey and predator
        dist_neighbors = {}
        
        # Setting priority for every neighbor according to the order followed by Agent1 
        priority = {}

        for neighbor in agent_neighbours:

            # Distance of shortest path from neighbor to prey
            neighbor_to_prey = graph.shortest_path_length( neighbor, prey_pos)

            # Distance of shortest path from neighbor to predator
            neighbor_to_predator = graph.shortest_path_length( neighbor, pred_pos)

            dist_neighbors[neighbor] = (neighbor_to_prey,neighbor_to_predator)

            if neighbor_to_prey < agent_to_prey and neighbor_to_predator > agent_to_predator:
                priority[neighbor] = 1
                
            elif neighbor_to_prey < agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 2
                
            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 3
                
            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 4
                
            elif neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 5
                
            elif neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 6
                
            else :
                priority[neighbor] = 7
        
        # Highest priority value out of all the neighbors
        min_val = min(priority.values())

        # Neighbors with the Highest priority
        if min_val < 7:
            # List of neighbors with highest priority
            Highest_priority_neighbors = [key for key, value in priority.items() if value == min_val]      

            # Breaking ties at random
            next_pos = random.choice(Highest_priority_neighbors)
            
            # Update node for the agent
            self.node = next_pos

        else:
            self.node = self.node
    
    def move_2(self, graph: Graph, prey_pos, pred_pos, distracted=False) -> None:
        """
        Movement logic for Agent 2
        """
        # Shortest Distance from Agent to prey
        agent_to_prey = graph.shortest_path_length(self.node , prey_pos)
        
        # Shortest Distance from Agent to Predator
        agent_to_predator = graph.shortest_path_length(self.node , pred_pos)

        # List of Neighbors of Agent
        agent_neighbours = graph.neighbors(self.node)

        # Distance of shortest path from each neighbour to prey and predator
        dist_neighbors = {}

        # Setting priority for every neighbor according to the order followed by Agent1 
        priority = {}

        for neighbor in agent_neighbours:
            # Distance of shortest path from neighbor to prey
            neighbor_to_prey = graph.shortest_path_length(neighbor, prey_pos)

            # Distance of shortest path from neighbor to predator
            neighbor_to_predator = graph.shortest_path_length( neighbor, pred_pos)

            # Distance of shortest path from neighbor to predator future
            neighbor_to_predator_future = self.predator_move_sim(graph, pred_pos, neighbor, distracted) # We just need distance

            neighbor_to_prey_future = self.prey_move_sim(graph, prey_pos, neighbor) # We get an average distance

            dist_neighbors[neighbor] = (neighbor_to_prey, neighbor_to_predator)

            if neighbor_to_prey_future < agent_to_prey and neighbor_to_predator_future > agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 1

            elif neighbor_to_prey_future < agent_to_prey and neighbor_to_predator_future == agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 2
                
            elif neighbor_to_prey_future == agent_to_prey and neighbor_to_predator_future > agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 3
            
            elif neighbor_to_prey_future == agent_to_prey and neighbor_to_predator_future == agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 4

            elif neighbor_to_prey < agent_to_prey and neighbor_to_predator_future > agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 5

            elif neighbor_to_prey < agent_to_prey and neighbor_to_predator_future == agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 6
                
            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator_future > agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 7
            
            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator_future == agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 8

            elif neighbor_to_predator_future > agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 9

            elif neighbor_to_predator_future == agent_to_predator and neighbor_to_predator >= agent_to_predator :
                priority[neighbor] = 10

            elif neighbor_to_prey_future < agent_to_prey and neighbor_to_predator > agent_to_predator:
                priority[neighbor] = 11
                
            elif neighbor_to_prey_future < agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 12

            elif neighbor_to_prey_future == agent_to_prey and neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 13
                
            elif neighbor_to_prey_future == agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 14

            elif neighbor_to_prey < agent_to_prey and neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 15
                
            elif neighbor_to_prey < agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 16

            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 17
                
            elif neighbor_to_prey == agent_to_prey and neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 18

            elif neighbor_to_predator > agent_to_predator :
                priority[neighbor] = 19
                
            elif neighbor_to_predator == agent_to_predator :
                priority[neighbor] = 20

            else :
                priority[neighbor] = 21
            
        # Highest priority value out of all the neighbors
        min_val = 21
        min_val = min(priority.values())

        # Neighbors with the Highest priority
        if min_val < 21:
            # List of neighbors with highest priority
            Highest_priority_neighbors = [key for key, value in priority.items() if value == min_val]

            # Breaking ties at random
            next_pos = random.choice(Highest_priority_neighbors)

            # Update node for the agent
            self.node = next_pos
        
        else:
            self.node = self.node

    def move_3(self, graph: Graph, prey_pos: int, pred_pos: int, time_steps: int) -> bool:
        """
        Movement logic for Agents 3 and 4
        """
        error = 10 ** -5

        if time_steps == 1:
            # Initial transition matrix (not to be altered)
            self.update_prey_trans_matrix(graph)

        elif time_steps > 1:
            # Update beliefs post prey move
            """
            P ( prey at some_node now ) = SUM [ P ( prey at some_node now AND prey was at old_node then ) ]
                ... Marginalization

            P ( prey at some_node now ) = SUM [ P ( prey at old_node then ) * P ( prey at some_node now | prey at old_node then ) ]
                ... Conditional Factoring

            P ( prey at some_node now ) = SUM [ P ( prey at old_node then ) * P ( some_node | old_node ) ]
                ... Simplifying the last probability which is basically the transition probability
            
            New beilief = DOT PRODUCT [ old_belief , row in the transition matrix ]
                ... In terms of what we have
            """

            updated_beliefs_ndarray = np.matmul(self.prey_transition_matrix, np.array(self.prey_beliefs))
            self.prey_beliefs = updated_beliefs_ndarray.tolist()

            if not prob.check_sum_beliefs(self.prey_beliefs):
                raise ValueError("Sum of beliefs error (after prey move update)")

        # Find the best survey pos
        best_survey_node = prob.node_to_survey(self.prey_beliefs, "prey")

        # Perform survey on the best node
        self.prey_beliefs = prob.survey(self.prey_beliefs, best_survey_node, prey_pos)

        if not prob.check_sum_beliefs(self.prey_beliefs):
            raise ValueError("Sum of beliefs error (after node survey)")

        # Find the node for which we have highest belief
        max_prob_nodes = [node for node, belief in enumerate(self.prey_beliefs) if belief == max(self.prey_beliefs)]

        # If more than one, break ties randomly if 
        prob_prey_pos = random.choice(max_prob_nodes)

        # print("Highest probable current position of prey is [{}] with probability [{}], so we'll use that info".format(prob_prey_pos, self.prey_beliefs[prob_prey_pos]))

        if self.name == "agent3":
            # Do the actual movement to highest belief node based on agent 1 logic for agent 3
            self.move_1(graph, prob_prey_pos, pred_pos)
        
        elif self.name == "agent4":
            # Do the actual movement to highest belief node based on agent 2 logic for agent 4
            self.move_2(graph, prob_prey_pos, pred_pos)

        # Update beliefs post agent move
        prob.survey(self.prey_beliefs, self.node, prey_pos)

        if not prob.check_sum_beliefs(self.prey_beliefs):
            raise ValueError("Sum of beliefs error (after agent move)")
        
        if 1.0 - error <= max(self.prey_beliefs) <= 1.0 + error:
            return True
        else:
            return False


    def move_utility(self, graph: Graph, prey_pos, pred_pos,state_policy) -> None:
        """
        This function moves the agent according to the U* strategy mentioned in the write up
        """

        state = (self.node,prey_pos,pred_pos)
        self.node = state_policy[state] 


    def move_partial(self, graph: Graph, prey_pos: int, pred_pos: int, time_steps: int, transition_model_partial, optimal_state_utility, immediate_reward, partial_utility) -> bool:
        """
        Movement logic for Agent U_partial
        """
        error = 10 ** -5

        if time_steps == 1:
            # Initial transition matrix (not to be altered)
            self.update_prey_trans_matrix(graph)

        elif time_steps > 1:
            # Update beliefs post prey move
            """
            P ( prey at some_node now ) = SUM [ P ( prey at some_node now AND prey was at old_node then ) ]
                ... Marginalization

            P ( prey at some_node now ) = SUM [ P ( prey at old_node then ) * P ( prey at some_node now | prey at old_node then ) ]
                ... Conditional Factoring

            P ( prey at some_node now ) = SUM [ P ( prey at old_node then ) * P ( some_node | old_node ) ]
                ... Simplifying the last probability which is basically the transition probability
            
            New beilief = DOT PRODUCT [ old_belief , row in the transition matrix ]
                ... In terms of what we have
            """

            updated_beliefs_ndarray = np.matmul(self.prey_transition_matrix, np.array(self.prey_beliefs))
            self.prey_beliefs = updated_beliefs_ndarray.tolist()

            if not prob.check_sum_beliefs(self.prey_beliefs):
                raise ValueError("Sum of beliefs error (after prey move update)")

        # Find the best survey pos
        best_survey_node = prob.node_to_survey(self.prey_beliefs, "prey")

        # Perform survey on the best node
        self.prey_beliefs = prob.survey(self.prey_beliefs, best_survey_node, prey_pos)

        if not prob.check_sum_beliefs(self.prey_beliefs):
            raise ValueError("Sum of beliefs error (after node survey)")

        probability_vector = self.prey_beliefs.copy()
        next_pos = self.get_policy(graph, pred_pos, probability_vector, transition_model_partial, optimal_state_utility, immediate_reward, partial_utility)

        # state_policy[s] = next_pos
        self.node = next_pos
        

        # # Find the node for which we have highest belief
        # max_prob_nodes = [node for node, belief in enumerate(self.prey_beliefs) if belief == max(self.prey_beliefs)]

        # # If more than one, break ties randomly if 
        # prob_prey_pos = random.choice(max_prob_nodes)
        # print("Highest probable current position of prey is [{}] with probability [{}], so we'll use that info".format(prob_prey_pos, self.prey_beliefs[prob_prey_pos]))

        

        # Update beliefs post agent move
        prob.survey(self.prey_beliefs, self.node, prey_pos)

        if not prob.check_sum_beliefs(self.prey_beliefs):
            raise ValueError("Sum of beliefs error (after agent move)")
        
        if 1.0 - error <= max(self.prey_beliefs) <= 1.0 + error:
            return True
        else:
            return False
        
        
    def move_model(self, graph: Graph, prey_pos: int, pred_pos: int,  transition_model, predict) -> bool:
        """
        Movement logic for Agent V_MODEL
        """

        state = (self.node,prey_pos,pred_pos) 
        self.node = self.get_policy_model( graph, state, transition_model, predict)  
            


        