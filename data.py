import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import statistics

import matplotlib.pyplot as plt
import numpy as np


# labels = ['G1', 'G2', 'G3', 'G4', 'G5']
# men_means = [20, 34, 30, 35, 27]
# women_means = [25, 32, 34, 20, 25]

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Men')
# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x, labels)
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()

# plt.show()

def plotting(l1, l2, num_runs, num_trials, agent_num, graph_number):


    x = np.arange(len(agent_num))
    fig, ax = plt.subplots()

    rects1 = ax.bar(x + 0.00, l1, width = 0.25,label='Success')
    rects2 = ax.bar(x + 0.25, l2, width = 0.25,label='Timesteps')
    

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Rate')
    ax.set_xlabel('Agents')
    ax.set_title("Comparing AGENTS {} for Graph {} (Tested on {} different starting states with {} trials on each state)".format(agent_num, graph_number, num_runs, num_trials ))
    ax.set_xticks(x, agent_num)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # Just save a PNG for the plot
    plt.gcf().set_size_inches(11.2775330396, 7.04845814978) # The MacbookPro 13.3 inches size
    file_name = "./Graphs/graph" + str(graph_number)  +'/comparingAgents {}.png'.format(agent_num)
    plt.savefig(file_name, dpi=227) # The MacbookPro 13.3 inches dpi

    plt.show()


    
# def plotting2(l1, l2, num_runs, num_trials, agent_num, agent_char):

#     # X
#     x = [i for i in range(num_runs)]
#     plt.xticks(x)

#     # Plots for success/failure rates
#     plt.plot(x, l1, marker='o', color='b', label='prey_known_rate')
#     for i, j in zip([x[i] for i in range(0,len(x),5)],[l1[i] for i in range(0,len(l1),5)] ):
#         plt.text(i+ 0.050, j + 0.0010, "({}, {})".format(i, round(j, 2)))

#     plt.plot(x, l2, marker='o', color='r', label='predator_known_rate')
#     for i, j in zip([x[i] for i in range(0,len(x),5)],[l2[i] for i in range(0,len(l2),5)] ):
#         plt.text(i+ 0.050, j + 0.010, "({}, {})".format(i, round(j, 2)))

#     # Labels and legend
#     plt.legend(loc="upper right")
#     plt.xlabel('Run Numbers')
#     plt.ylabel('accurate_prediction')
#     plt.title("AGENT {} (Tested on {} trials for each run)".format(agent_num, num_trials))
#     plt.legend()
    

#     # Just save a PNG for the plot
#     plt.gcf().set_size_inches(11.2775330396, 7.04845814978) # The MacbookPro 13.3 inches size
#     plt.savefig("./know_graphs/agent{}.png".format(str(agent_num) + agent_char), dpi=227) # The MacbookPro 13.3 inches dpi
#     plt.show()
#     plt.show()

# def weighted_avg_and_std(values, weights):

#     average = np.average(values, weights=weights)
#     variance = np.average((values-average)**2, weights=weights)

#     return (average, math.sqrt(variance))




graph_number = 0
# int(input("Enter the Graph number: "))
input_string = input('Enter the agents you want to compare: ')
agent_num = input_string.split()

# convert each item to int type
for i in range(len(agent_num)):
    agent_num[i] = int(agent_num[i])

print('Comparing Agents: ', agent_num)

num_runs = 1000
# int(input("Enter the number of different starting states for which you want to cram data: "))


num_trials = 0

# prey_known = []
# pred_known = []

mean_success_agent = []
mean_timesteps_agent = []


for agent in agent_num:
    total_timesteps = 0
    total_success = 0
    for run in range(num_runs):

        file_name = "./Graphs/graph" + str(graph_number)  + "/agent_csv/agent" + str(agent) +  "/data/Run" + str(run) + '.csv' 
        df = pd.read_csv(file_name)

        mean_success_run = df["success"].mean()
        mean_timesteps_run = df["time_steps"].mean()

        total_success += mean_success_run
        total_timesteps += mean_timesteps_run
    
    mean_success = (total_success/ num_runs)*100  
    mean_success_agent.append(mean_success) 

    mean_total_timesteps = total_timesteps/ num_runs  
    mean_timesteps_agent.append(mean_total_timesteps) 



    # filepath2 = "./know_csv/agent" + str(agent_num) + agent_char + "/Run" + str(run) + '.csv'
    # df2 = pd.read_csv(filepath2)
    # prey_known_percentage = df2['times_prey_known'].sum()*100/df2['time_steps'].sum()
    # prey_known.append(prey_known_percentage)
    # pred_known_percentage = df2['times_pred_known'].sum()*100/df2['time_steps'].sum()
    # pred_known.append(pred_known_percentage)
    # total_timesteps = df2['time_steps'].mean()
    # total_timesteps_run.append(total_timesteps)


    if num_trials == 0:
        num_trials = len(df)


# mean_success = sum(agent_success)*100/len(agent_success)
# print("Mean success rate for Graph {}: {}".format(graph_number,mean_success))
# sd_success = statistics.pstdev(agent_success)*100

# mean_failure_predator = sum(failure_pred)*100/len(failure_pred)
# sd_failure_predator = statistics.pstdev(failure_pred)*100

# mean_failure_timeout = sum(failure_timeout)*100/len(failure_timeout)
# sd_failure_timeout = statistics.pstdev(failure_timeout)*100



# mean_prey , sd_prey = weighted_avg_and_std(prey_known, total_timesteps_run)
# mean_pred , sd_pred = weighted_avg_and_std(pred_known, total_timesteps_run)



# sd_t_ts = statistics.pstdev(total_timesteps_run)

# df = pd.DataFrame()
# df["success"]= [mean_success,sd_success]
# df["failure_predator"]= [mean_failure_predator,sd_failure_predator]
# df["failure_timeout"]= [mean_failure_timeout,sd_failure_timeout]
# df["Prey_know"] = [mean_prey,sd_prey]
# df["Predator_known"] = [mean_pred,sd_pred]
# df["Average_Timesteps"] = [mean__t_ts,sd_t_ts]

# file_name = "./final/agent" + str(agent_num) + agent_char + '.csv'
# df.to_csv(file_name, encoding='utf-8')

print(mean_success_agent)
print(mean_timesteps_agent)

plotting(mean_success_agent, mean_timesteps_agent, num_runs, num_trials, agent_num, graph_number)
# plotting2(prey_known,pred_known, num_runs, num_trials, agent_num, agent_char)
