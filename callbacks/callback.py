import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import pickle

from rebuild_policy import linearize_NN


class CustomCallback(BaseCallback):
    def __init__(self, log_path=None, verbose: int = 0):
        self.log_path = log_path
        self.eval_freq = 100_000
        self.best_mean_reward = -np.inf
        self.mean_rewards = []
        super().__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        self.parameter_file = os.path.join(self.log_path, "training_logs.p")
        d = {"NN_updates": {"policy": {}, "linearization": [], "step": [], "episode": []}}
        with open(self.parameter_file, "wb") as f:
            pickle.dump(d, f)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_path), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                self.mean_rewards.append(mean_reward)
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                            self.best_mean_reward, mean_reward
                        )
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.log_path))
                    self.model.save(
                        os.path.join(self.log_path, "intermediate_models", f"best_model_{self.num_timesteps}.h5")
                    )
                    self.best_episode = [x[-1], y[-1]]
        return super()._on_step()

    def on_rollout_start(self):
        p = self.model.get_parameters()["policy"]
        K = linearize_NN(self.training_env, self.model)
        with open(self.parameter_file, "rb") as f:
            p_log = pickle.load(f)
        for key in p.keys():
            if key not in p_log["NN_updates"]["policy"].keys():
                p_log["NN_updates"]["policy"][key] = []

            p_log["NN_updates"]["policy"][key].append(p[key].detach().numpy())

        p_log["NN_updates"]["step"].append(self.training_env.envs[0].env.total_step_count)
        p_log["NN_updates"]["episode"].append(self.training_env.envs[0].env.episode_count)
        p_log["NN_updates"]["linearization"].append(K)

        with open(self.parameter_file, "wb") as f:
            pickle.dump(p_log, f)
