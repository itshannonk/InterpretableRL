"""
Takes npy files from the results directory and plots them.
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot(data_file_list: list, xlabel: str, ylabel: str, title: str, save_filename: str, add_average: bool = False):
    """
    Plot the data from the given filenames.

    Args:
        filenames (list): List of filenames to plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    colors = ['r', 'c']
    i = 0
    for data_file_path in data_file_list:
        data = np.load(data_file_path)
        algorithm_name = "GRPO" if "grpo" in data_file_path else "PPO"
        plt.plot(data, label=algorithm_name)
        if add_average:
            print(f"Average {algorithm_name}: {np.mean(data)}")
            plt.axhline(y=np.mean(data), color=colors[i], linestyle='--', label='Average (' + algorithm_name + ')')
        i += 1
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_filename)
    plt.close()

def plot_multiple_seeds(file_directories: list, filename: str, xlabel: str, ylabel: str, title: str, save_filename: str, add_average: bool = False):
    """
    Plot the data from the given directories.

    Args:
        file_directories (list): List of directory prefixes to plot
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.
    """
    colors = ['r', 'c']
    i = 0
    for directory in file_directories:
        algorithm_name = "GRPO" if "grpo" in directory else "PPO"
        results = []
        for seed in range(1, 7):  # we used seeds 1 - 6
            results.append(np.load(f"{directory}{seed}/{filename}"))
        plt.plot(np.mean(results, axis=0), label=algorithm_name)
        plt.fill_between(range(len(results[0])), np.min(results, axis=0), np.max(results, axis=0), alpha=0.2)

        if add_average:
            plt.axhline(y=np.mean(results), color=colors[i], linestyle='--', label='Average (' + algorithm_name + ')')
        i += 1
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_filename)
    plt.close()


if __name__ == "__main__":
    # # Parse the arguments
    # parser = argparse.ArgumentParser(description="Create comparison plots from npy files.")
    # parser.add_argument("--xlabel", type=str, default="Training Episode", help="Label for the x-axis")
    # parser.add_argument("--ylabel", type=str, default="Average Reward", help="Label for the y-axis")
    # parser.add_argument("--title", type=str, default="Average Reward vs Training Episode", help="Title of the plot")
    # parser.add_argument("--save_filename", type=str, help="Filename to save the plot")
    # parser.add_argument("--add_average", action="store_true", help="Add average line to the plot")
    # args = parser.parse_args()


    # # Example usage
    filenames = [
        "results/Humanoid-v3-grpo-seed=1/scores.npy",
        "results/Humanoid-v3-ppo-seed=1/scores.npy"
    ]
    plot(filenames,
         "Training Episode",
         "Average Reward",
         "Average Reward vs Training Episode",
         "results/grpo-ppo-comparison-cartpole.png")

    # plot(filenames,
    #      "Training Episode",
    #      "Peak Memory Usage (MB)",
    #      "Peak Memory Usage vs Training Episode",
    #      "results/grpo-ppo-memory-comparison-cartpole.png",
    #      add_average=True)
    
    # plot(filenames,
    #      args.xlabel,
    #      args.ylabel,
    #      args.title,
    #      args.save_filename,
    #      args.add_average)

    # Example usage
    directories = [
        "results/Humanoid-v3-grpo-seed=",
        "results/Humanoid-v3-ppo-seed="
    ]

    plot_multiple_seeds(directories,
         "scores.npy",
         "Training Episode",
         "Average Reward",
         "Average Reward vs Training Episode",
         "results/combined-grpo-ppo-reward-comparison-humanoid.png",
         add_average=False)

    plot_multiple_seeds(directories,
         "memory.npy",
         "Training Episode",
         "Peak Memory Usage (MB)",
         "Peak Memory Usage vs Training Episode",
         "results/combined-grpo-ppo-memory-comparison-humanoid.png",
         add_average=True)