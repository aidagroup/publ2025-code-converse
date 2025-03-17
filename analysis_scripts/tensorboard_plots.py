import pandas as pd
import os
import matplotlib.pyplot as plt

csv_path = "./Csvs_opt"

def extract_rewards_from_csv(csv_path):
    """Extracts all rewards and timesteps from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if not df.empty:
            if 'Value' in df.columns:
                return df['Value'].values, df.index.values  # Return rewards and corresponding timesteps (index)
            elif 'ep_rew_mean' in df.columns:
                return df['ep_rew_mean'].values, df.index.values  # Return rewards and corresponding timesteps (index)
            else:
                print(f"Error: 'Value' or 'ep_rew_mean' column not found in {csv_path}")
                return None, None
        else:
            print(f"Warning: CSV file {csv_path} is empty.")
            return None, None
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None, None

def extract_all_rewards(csvs_dir):
    """Extracts reward values and timesteps from all relevant CSV files."""
    all_rewards = {
        "_Adam": [],
        "_RAdam": [],
    }
    
    for filename in os.listdir(csvs_dir):
        if "Adam" in filename and filename.endswith(".csv"):
            filepath = os.path.join(csvs_dir, filename)
            rewards, timesteps = extract_rewards_from_csv(filepath)
            if rewards is not None and timesteps is not None:
                if "_Adam" in filename:
                    all_rewards["_Adam"].append((timesteps, rewards))
                elif "_RAdam" in filename:
                    all_rewards["_RAdam"].append((timesteps, rewards))
    
    return all_rewards

def plot_rewards(all_rewards):
    """Plots rewards for the given extract rewards data."""
    plt.figure(figsize=(10, 6))
    
    # Plot for 0.0 files
    for timesteps, rewards in all_rewards["_Adam"]:
        plt.plot(timesteps, rewards, color='blue', alpha=0.5, label='Reward for _Adam' if 'Reward for _Adam' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot for 0.5 files
    for timesteps, rewards in all_rewards["_RAdam"]:
        plt.plot(timesteps, rewards, color='green', alpha=0.5, label='Reward for _RAdam' if 'Reward for _RAdam' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title('Rewards over Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Rewards')
    plt.ylim(-15000,1000)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    plt.savefig("rewards_plot_com.png")
    plt.show()

# Extract all rewards and plot them
all_rewards = extract_all_rewards(csv_path)
plot_rewards(all_rewards)
