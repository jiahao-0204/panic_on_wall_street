# panic on wall street

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def compute_expectation(color, position):

    # reward
    reward_list = []
    # expectation
    sum = 0
    for i in range(6):
        new_position = position + dice_dict[color][i]
        new_position = max(0, new_position)
        new_position = min(7, new_position)
        reward = board_dict[color][new_position]
        reward_list.append(reward)
        sum += reward
    expectation = sum / 6

    # std
    variance = 0
    for i in range(6):
        new_position = position + dice_dict[color][i]
        new_position = max(0, new_position)
        new_position = min(7, new_position)
        variance += (board_dict[color][new_position] - expectation) ** 2
    variance = variance / 6
    std = variance ** 0.5

    # max and min of reward_list
    max_reward = max(reward_list)
    min_reward = min(reward_list)

    return expectation, std, max_reward, min_reward


def compute_transition_matrix(color):
    # initialize a 8x8 matrix
    # state^T x transition matrix = new state
    transition_matrix = np.zeros((8, 8))
    for position in range(8):
        for dice_number in range(6):
            new_position = position + dice_dict[color][dice_number]
            new_position = max(0, new_position)
            new_position = min(7, new_position)

            transition_matrix[position, new_position] += 1/6
    return transition_matrix


def compute_state_after_n_round(initial_state, transition_matrix, number_of_round):
    for i in range(number_of_round):
        initial_state = np.dot(initial_state, transition_matrix)
    return initial_state


def compute_total_seller_max_profit(color, current_position, number_of_rounds_left):
    
    # initial state
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    initial_state[current_position] = 1

    # compute total profit
    total_max_profit = 0
    for round in range(number_of_rounds_left):

        # current state distribution
        current_state = compute_state_after_n_round(initial_state, transition_matrix_dict[color], round)

        # buyer max price & seller max profit
        buyer_max_price = buyer_max_price_dict[color]
        seller_max_profit = buyer_max_price - tax
        
        # expectation of seller max profit over current state distribution
        current_max_profit = current_state.dot(seller_max_profit)

        # add to total
        total_max_profit += current_max_profit

    # return
    return total_max_profit


# board multiplier
board_multiplier = 2

# color list
color_list = ["r", "y", "g", "b"]

# board
board_dict = {}
board_dict["r"] = np.array([-20, -10, 0, 30, 40, 50, 60, 70]) * board_multiplier
board_dict["y"] = np.array([-10, 0, 0, 30, 40, 40, 60, 60]) * board_multiplier
board_dict["g"] = np.array([0, 10, 20, 30, 30, 40, 50, 60]) * board_multiplier
board_dict["b"] = np.array([20, 20, 20, 30, 30, 30, 40, 40]) * board_multiplier

# dice
dice_dict = {}
dice_dict["r"] = np.array([-7, -3, -2, 2, 3, 7])
dice_dict["y"] = np.array([-3, -2, -1, 1, 2, 3])
dice_dict["g"] = np.array([-2, -1, 0, 1, 2, 2])
dice_dict["b"] = np.array([-1, -1, 0, 0, 1, 1])

# color
color_rgb_dict = {}
color_rgb_dict["r"] = (158/255, 48/255, 46/255)
color_rgb_dict["y"] = (241/255, 154/255, 71/255)
color_rgb_dict["g"] = (58/255, 102/255, 55/255)
color_rgb_dict["b"] = (29/255, 61/255, 96/255)

# tax
tax = 10

# exp std max min
exp_dict = {}
std_dict = {}
max_dict = {}
min_dict = {}
for color in ["r", "y", "g", "b"]:
    exp_list = []
    std_list = []
    max_list = []
    min_list = []
    for dice_number in range(8):
        exp, std, max_reward, min_reward = compute_expectation(color, dice_number)
        exp_list.append(exp)
        std_list.append(std)
        max_list.append(max_reward)
        min_list.append(min_reward)

    exp_dict[color] = exp_list
    std_dict[color] = std_list
    max_dict[color] = max_list
    min_dict[color] = min_list

# fair price
fair_price_dict = {}
for color in ["r", "y", "g", "b"]:
    fair_price_list = []
    for dice_number in range(8):
        profit = exp_dict[color][dice_number] - tax
        if profit < 0:
            fair_price = 0
        else:
            half_profit = profit / 2
            fair_price = tax + half_profit
        fair_price_list.append(fair_price)
    fair_price_dict[color] = fair_price_list


# plot position
position_dict = {}
offset = 0
position_list = [0, 1, 2, 3, 4, 5, 6, 7]
position_dict["r"] = [i + offset*3 for i in position_list]
position_dict["y"] = [i + offset*2 for i in position_list]
position_dict["g"] = [i + offset*1 for i in position_list]
position_dict["b"] = position_list


# pd table
table_dict = {}
for color in ["r", "y", "g", "b"]:
    table_data = {
        # "Position": position_dict[color],
        "Board Price": board_dict[color],
        "Expectation Price": exp_dict[color],
        "Fair Price": fair_price_dict[color]
        # "Max Reward": max_dict[color],
        # "Min Reward": min_dict[color]
    }
    df = pd.DataFrame(table_data).T
    table_dict[color] = df

# transition matrix
transition_matrix_dict = {}
for color in ["r", "y", "g", "b"]:
    transition_matrix_dict[color] = compute_transition_matrix(color)

# buyer max price
buyer_max_price_dict = {}
for color in ["r", "y", "g", "b"]:
    buyer_max_price_list = []
    for dice_number in range(8):
        # floor round to 5 from expectation price
        buyer_max_price = int(5 * ((exp_dict[color][dice_number]) // 5))
        buyer_max_price_list.append(buyer_max_price)
    buyer_max_price_dict[color] = np.array(buyer_max_price_list)



def print_table_dict():
    pd.set_option('display.precision', 0) # no decimal
    for color in ["r", "y", "g", "b"]:
        print(color)
        print(table_dict[color].to_string(header=False))
        print("\n")


def plt_transition_bar_plot(number_of_round, initial_state = np.array([0, 0, 0, 1, 0, 0, 0, 0])):
    plt.figure()
    for j in range(4):
        color = color_list[j]
        for i in range(number_of_round):
            plt.subplot(number_of_round, 4, i*4+j+1)
            current_state = compute_state_after_n_round(initial_state, transition_matrix_dict[color], i)
            plt.bar(position_list, current_state, color=color_rgb_dict[color])
            plt.ylim(0, 1)
            plt.yticks([0, 1])
            plt.xticks(position_list, [])
        plt.xticks(position_list, position_list) 
    plt.show()


def plt_seller_max_profit(total_number_of_rounds_left, initial_state = np.array([0, 0, 0, 1, 0, 0, 0, 0])):

    # initialize plot
    fig, axs = plt.subplots(total_number_of_rounds_left, 1)

    # fill subplots
    for number_of_rounds_left in range(1, total_number_of_rounds_left+1):

        # initialize dict to plot
        dict_to_indicate = {}
        total_seller_max_profit_dict = {}

        # populate dict
        for color in color_list:
            
            # compute seller max profit 
            total_seller_max_profit_list = []
            for position in range(8):
                total_seller_max_profit = compute_total_seller_max_profit(color, position, number_of_rounds_left)
                total_seller_max_profit_list.append(total_seller_max_profit)
            
            # compute indicator
            current_state = compute_state_after_n_round(initial_state, transition_matrix_dict[color], total_number_of_rounds_left-number_of_rounds_left)
            indicator_list = current_state > 0

            # store
            total_seller_max_profit_dict[color] = total_seller_max_profit_list
            dict_to_indicate[color] = indicator_list

        # subplot
        plt_contour_plot_dict_indicated(axs[total_number_of_rounds_left-number_of_rounds_left], total_seller_max_profit_dict, dict_to_indicate, title=f"seller max profit vs current position ({number_of_rounds_left} rounds left) x{board_multiplier}", levels = np.arange(-20 * board_multiplier, 250 * board_multiplier, 5 * board_multiplier))
        
    # show
    plt.show()


def plt_range_and_error_bar():
    # initialize
    plt.figure()

    # plot
    for color in ["r", "y", "g", "b"]:
        # range
        plt.fill_between(position_dict[color], max_dict[color], min_dict[color], color=color_rgb_dict[color], alpha=0.2)

        # error bar
        plt.errorbar(position_dict[color], exp_dict[color], yerr=std_dict[color], fmt='o', color=color_rgb_dict[color])

    # show
    plt.show()


def plt_contour_plot_dict(ax, dict_to_plot, title, levels = np.arange(-20 * board_multiplier, 75 * board_multiplier, 5)):
    # convert to matrix to easier plot contour
    matrix = np.array([dict_to_plot["r"], dict_to_plot["y"], dict_to_plot["g"], dict_to_plot["b"]])[::-1]

    # plot contour
    ax.contourf(matrix, levels=levels)

    # plot marker and expectation value inside
    for i in range(4):
        for j in range(8):

            # coordinates
            x = j
            y = 3-i

            # plot marker
            ax.plot(x, y, marker = 'o', color = color_rgb_dict[color_list[i]], markersize=25) 

            # plot expectation value inside
            value_to_plot = round(matrix[y, x], 1)
            ax.text(x, y, value_to_plot, ha='center', va='center', color='white')

    # set limit and title
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title(title)
    ax.axis('off')


def plt_contour_plot_dict_indicated(ax, dict_to_plot, dict_to_indicate, title, levels = np.arange(-20 * board_multiplier, 75 * board_multiplier, 5)):
    # convert to matrix to easier plot contour
    matrix = np.array([dict_to_plot["r"], dict_to_plot["y"], dict_to_plot["g"], dict_to_plot["b"]])[::-1]
    indicate_matrix = np.array([dict_to_indicate["r"], dict_to_indicate["y"], dict_to_indicate["g"], dict_to_indicate["b"]])[::-1]

    # plot contour
    ax.contourf(matrix, levels=levels)

    # plot marker and expectation value inside
    for i in range(4):
        for j in range(8):

            # coordinates
            x = j
            y = 3-i

            # if not indicated, skip
            if indicate_matrix[y, x] == 0:
                continue

            # plot marker
            ax.plot(x, y, marker = 'o', color = color_rgb_dict[color_list[i]], markersize=25) 

            # plot expectation value inside
            value_to_plot = round(matrix[y, x], 1)
            ax.text(x, y, value_to_plot, ha='center', va='center', color='white')

    # set limit and title
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title(title)
    ax.axis('off')


# subplot transition bar
plt_transition_bar_plot(5)


# subplot both board price and expectation price
fig, axs = plt.subplots(3, 1)
plt_contour_plot_dict(axs[0], board_dict, title=f"Board Price x{board_multiplier}")
plt_contour_plot_dict(axs[1], exp_dict, title=f"Expectation Price x{board_multiplier}")
plt_contour_plot_dict(axs[2], buyer_max_price_dict, title=f"Buyer Max Price x{board_multiplier}")
plt.show()


# subplot both board price and expectation price
plt_seller_max_profit(5, np.array([0, 0, 0, 1, 0, 0, 0, 0]))