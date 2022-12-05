import math
import random
from graph_utils import Graph

def node_to_survey(current_belief: list, target: str) -> int:
    """
    Returns best node to survey for partial prey/pred
    """
    if target == "prey" or target == "predator":
        # List of nodes with the highest probability of prey/predator
        high_prob_nodes = [node for node in range(len(current_belief)) if current_belief[node] == max(current_belief)]

        return random.choice(high_prob_nodes)

    else:
        raise ValueError("The target for survey needs to be either prey or predator and not {}".format(target))

def node_to_survey_closest(graph: Graph, agent_pos, current_belief: list, target: str) -> int:
    """
    Returns best+closest node to survey for partial prey/pred
    """
    if target == "predator":
        # List of nodes with the highest probability of prey/predator
        high_prob_nodes = [node for node in range(len(current_belief)) if current_belief[node] == max(current_belief)]

        sp_len = math.inf
        nodes_to_survey = []

    elif target == "prey":
        # List of nodes with the highest probability of prey/predator
        high_prob_nodes = [node for node in range(len(current_belief)) if current_belief[node] == max(current_belief)]

    else:
        raise ValueError("The target for survey needs to be either prey or predator and not {}".format(target))

    for node in high_prob_nodes:

        node_len = graph.shortest_path_length(node, agent_pos)            

        if node_len < sp_len:
            sp_len = node_len
            nodes_to_survey = [node]
        elif node_len == sp_len:
            nodes_to_survey.append(node)

    return random.choice(nodes_to_survey)

def node_to_survey_proximity(graph: Graph, agent_pos, current_belief: list, target: str) -> int:
    """
    Returns best+proximity node to survey for partial prey/pred
    """
    if target == "predator":
        l1_neighbors = graph.neighbors(agent_pos)

        # l2_neighbors = []
        # for neighbor in l1_neighbors:
        #     l2_neighbors.extend(graph.neighbors(neighbor))

        # l1_neighbors.extend(l2_neighbors)

        # list(set(l1_neighbors)) # remove all duplicates

        region = l1_neighbors

        beliefs = {}

        for node in region:
            beliefs[node] = current_belief[node]

        max_prob_in_region = max(list(beliefs.values()))

        if max_prob_in_region > 0:
            # List of nodes with the highest probability of prey/predator
            high_prob_nodes = [node for node, prob in beliefs.items() if prob == max_prob_in_region]
            print("In nodes_to_survey_proximity")
            print(high_prob_nodes)

        else: # follow old closest logic
            # List of nodes with the highest probability of prey/predator
            high_prob_nodes = [node for node in range(len(current_belief)) if current_belief[node] == max(current_belief)]

        sp_len = math.inf
        node_to_survey = 0

        for node in high_prob_nodes:

            node_len = graph.shortest_path_length(node, agent_pos)            

            if graph.shortest_path_length(node, agent_pos) < sp_len:
                sp_len = node_len
                node_to_survey = node

        return node_to_survey

    elif target == "prey":
        # List of nodes with the highest probability of prey/predator
        high_prob_nodes = [node for node in range(len(current_belief)) if current_belief[node] == max(current_belief)]

        return random.choice(high_prob_nodes)

    else:
        raise ValueError("The target for survey needs to be either prey or predator and not {}".format(target))

def node_to_survey_proximity2(graph: Graph, agent_pos, current_belief: list) -> int:
    """
    Returns best+proximity node to survey for partial prey/pred
    """
    l1_neighbors = graph.neighbors(agent_pos)

    l1_neighbors = list(set(l1_neighbors))

    print(l1_neighbors)

    l2_neighbors = []

    for neighbor in l1_neighbors:
        for l2_neighbor in graph.neighbors(neighbor):
            l2_neighbors.append(l2_neighbor)

    l2_neighbors = list(set(l2_neighbors))

    l1_neighbors.extend(l2_neighbors)

    region = list(set(l1_neighbors)) # remove all duplicates

    print("Region")
    print(region)

    beliefs = {}

    for node in region:
        beliefs[node] = current_belief[node]

    max_prob_in_region = max(list(beliefs.values()))

    print("max_prob_in_region")
    print(max_prob_in_region)

    high_prob_nodes = [node for node, prob in beliefs.items() if prob == max_prob_in_region]
    sp_len = math.inf
    nodes_to_survey = []

    for node in high_prob_nodes:
        node_len = graph.shortest_path_length(node, agent_pos)

        if node_len < sp_len:
            sp_len = node_len
            nodes_to_survey = [node]
        elif node_len == sp_len:
            nodes_to_survey.append(node)
    
    print("nodes_to_survey")
    print(nodes_to_survey)

    cutoff = 0.0

    for node in nodes_to_survey:
        if node in l2_neighbors:
            cutoff = 0.5
        else:
            cutoff = 0.02

    to_survey = True

    if max_prob_in_region > 0 and max_prob_in_region < cutoff:
        # List of nodes with the highest probability of prey/predator
        to_survey = True

    elif max_prob_in_region >= cutoff:
        to_survey = False
        # List of nodes with the highest probability of prey/predator in region
    
    print("to_survey")
    print(to_survey)

    return (to_survey, random.choice(nodes_to_survey))

def node_to_survey_proximity9(graph: Graph, agent_pos, prey_belief: list, pred_belief: list, target: str) -> int:
    """
    Returns a cheat node to survey to almost always guarantee a success
    """
    """
    Returns best+proximity node to survey for partial prey/pred
    """
    if target == "predator":
        l1_neighbors = graph.neighbors(agent_pos)

        l1_neighbors = list(set(l1_neighbors))

        l2_neighbors = []
        for node in l1_neighbors:
            for neighbor in graph.neighbors(node):
                l2_neighbors.append(neighbor)

        l1_neighbors.extend(l2_neighbors)

        region = list(set(l1_neighbors)) # remove all duplicates

        beliefs = {}

        for node in region:
            beliefs[node] = pred_belief[node]

        max_prob_in_region = max(list(beliefs.values()))

        # List of nodes with the highest probability of prey/predator
        if max_prob_in_region > 0:
            high_prob_nodes = [node for node, prob in beliefs.items() if prob == max_prob_in_region]
            print("In nodes_to_survey_proximity")
            print(high_prob_nodes)

        else:
            high_prob_nodes = [node for node in range(len(pred_belief)) if pred_belief[node] == max(pred_belief)]

        sp_len = math.inf
        nodes_to_survey = []

        for node in high_prob_nodes:

            node_len = graph.shortest_path_length(node, agent_pos)            

            if node_len < sp_len:
                sp_len = node_len
                nodes_to_survey = [node]
            elif node_len == sp_len:
                nodes_to_survey.append(node)

        return random.choice(nodes_to_survey)

    elif target == "prey":
        # List of nodes with the highest probability of prey/predator
        high_prob_nodes = [node for node in range(len(prey_belief)) if prey_belief[node] == max(prey_belief)]

        return random.choice(high_prob_nodes)

    else:
        raise ValueError("The target for survey needs to be either prey or predator and not {}".format(target))


def survey(current_belief: list, node_surveyed: int, target_pos: int) -> list:
    """
    The function performs a "survey" and subsequent belief updates on the node_surveyed
    Every time an agent moves to a node it essentially surveys its new position (i.e. node_surveyed = agent.node)
    """
    # Check if prey/pred found at node_surveyed or not 
    if node_surveyed == target_pos:
        found = True
    else:
        found = False

    # If prey/pred is NOT at node surveyed
    if not found:
        """
        Formula used:
            P ( prey/pred at node | survey did not find prey/pred at node_surveyed )
                = P ( prey/pred at node AND survey did not find prey/pred at node_surveyed ) / P ( survey did not find prey/pred at node_surveyed )
                    ... Definition of Conditional Probability

                = P ( prey/pred at node ) * P ( survey did not find prey/pred at node_surveyed | prey/pred at node ) / P ( survey did not find prey/pred at node_surveyed )
                    ... Conditional Factoring

                P ( survey did not find prey/pred at node_surveyed | prey/pred at node ) = 1 ; if node_surveyed != actual position of prey/pred(i.e target_pos ); since we know prey/pred was at some other node(i.e. target_pos)than node_surveyed
                P ( survey did not find prey/pred at node_surveyed | prey/pred at node ) = 0 ; if node_surveyed == actual position of prey/pred(i.e target_pos ); since we know prey/pred was at target_pos which is equal to node_surveyed

                Case 1: if node_surveyed != actual position of prey/pred(i.e target_pos )
                = P ( prey/pred at node ) * 1 / P ( prey/pred not at node_surveyed )
                
                Case 2: if node_surveyed == actual position of prey/pred(i.e target_pos )
                = P ( prey/pred at node ) * 0 / P ( prey/pred not at node_surveyed )
                = 0

        """
        denominator = (1 - current_belief[node_surveyed])

        # Update probabilities
        for node in range(len(current_belief)):  
            if node == node_surveyed: # set probability of surveyed node = 0 since prey/pred is not there
                current_belief[node_surveyed] = 0 
            else: # update all the other probabilities
                current_belief[node] = current_belief[node] / denominator 

    else: # prey/pred is at node surveyed
        for node in range(len(current_belief)):  
            if node == node_surveyed: # set probability of surveyed node to 1 since prey/pred is there
                current_belief[node_surveyed] = 1
            else: # update all the other probabilities
                current_belief[node] = 0

    return current_belief

def survey_defective(current_belief: list, node_surveyed: int, target_pos: int, rand: float) -> list:
    """
    The function performs a "defective survey" and subsequent belief updates on the node_surveyed
    This is only used for agents 7b and 8b
    """
    # Check if prey/pred found at node_surveyed or not 
    if node_surveyed == target_pos:
        if rand <= 0.9: # 90% of the time the survey correctly finds a prey/pred
            found = True
        else: # 10% of the time there is a false negative
            found = False
    else:
        found = False

    # If prey/pred is NOT at node surveyed
    if not found:
        """
        Formula used:
            P ( prey/pred at node | survey did NOT FIND prey/pred at node_surveyed )
                = P ( prey/pred at node AND survey did NOT FIND prey/pred at node_surveyed ) / P ( survey did NOT FIND prey/pred at node_surveyed )
                    ... Definition of Conditional Probability

                = P ( prey/pred at node ) * P ( survey did NOT FIND prey/pred at node_surveyed | prey/pred at node ) / P ( survey did NOT FIND prey/pred at node_surveyed )
                    ... Conditional Factoring
                

            Where - 
            P ( prey/pred NOT at node_surveyed ) = P ( prey/pred NOT FOUND at node_surveyed AND prey/pred at node_surveyed ) + P ( prey/pred NOT FOUND at node_surveyed AND prey/pred NOT at node_surveyed )
                = P ( prey/pred at node_surveyed ) * P ( survey did NOT FIND prey/pred at node_surveyed | prey/pred at node_surveyed ) + P ( prey/pred NOT at node_surveyed )* P ( survey did NOT FIND prey/pred NOT at node_surveyed | prey/pred NOT at node_surveyed)
                    ... Conditional Factoring / Simplification of last probability
            
                P ( survey did NOT FIND prey/pred NOT at node_surveyed | prey/pred NOT at node_surveyed)  = 1 ;since we know prey/pred was not found at node_surveyed AND it was NOT at node_surveyed
                        ... (and the defect does not result in any false positives)

                = P ( prey/pred at node_surveyed ) * P ( survey did NOT FIND prey/pred at node_surveyed | prey/pred at node_surveyed ) + P ( prey/pred NOT at node_surveyed )* 1

                = P ( prey/pred at node_surveyed ) * 0.1 + ( 1 - P ( prey/pred at node_surveyed ) )

                =  1 - 0.9 * P ( prey/pred at node_surveyed ) )
                    ... Values known to us
        """

        denominator = 1 - current_belief[node_surveyed] * 0.9 

        # Update probabilities
        for node in range(len(current_belief)):  
            if node == node_surveyed:
                # set probability of surveyed node P ( prey/pred at node_surveyed ) * P ( survey did not find prey/pred at node_surveyed | prey/pred at node ) / denominator
                # since prey/pred is not found there
                current_belief[node_surveyed] = current_belief[node_surveyed] * 0.1 / denominator
            else:
                # update all the other probabilities
                current_belief[node] = current_belief[node] / denominator 

    else: # prey/pred is at node surveyed
        for node in range(len(current_belief)):  
            if node == node_surveyed: # set probability of surveyed node to 1 since prey/pred is there
                current_belief[node_surveyed] = 1
            else: # update all the other probabilities
                current_belief[node] = 0

    return current_belief

def check_sum_beliefs(beliefs: list) -> bool:
    """
    Checks if beliefs add up to 1
    """
    error = 10 ** -5

    if 1 - error <= sum(beliefs) <= 1 + error:
        return True
    else:
        print("Oops! Sum of beliefs not equal to 1. Instead they are: " + str(sum(beliefs)))
        return False
