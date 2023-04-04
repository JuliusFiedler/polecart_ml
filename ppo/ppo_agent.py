import gymnasium
import sys, os
import datetime
import matplotlib.pyplot as plt
import pickle

sys.modules["gym"] = gymnasium
import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from ipydex import IPS

from util import *


class PPOAgent:
    def __init__(self, env, *args, **kwargs) -> None:
        self.env = env
        self.seed = env.c.START_SEED
        self.model = None
        self.tensorboard_log = None
        self.ppo_kwargs = kwargs
        self.action_net_kwargs = None

    def create_model(self):
        self.model = PPO(
            "MlpPolicy", self.env, verbose=1, seed=self.seed, tensorboard_log=self.tensorboard_log, **self.ppo_kwargs
        )
        if self.action_net_kwargs:
            if "bias" in self.action_net_kwargs.keys():
                self.model.policy.action_net.bias = self.action_net_kwargs["bias"]
                self.model.policy.action_net.weight.data = self.action_net_kwargs["weight"]

    def train(self, total_timesteps=300000, callback=None, save_model=True):
        self.env.training = True
        # Create Folders and setup logs
        if save_model:
            t = dt.datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
            self.model_name = self.env.name + "__" + t
            self.folder_path = os.path.join(ROOT_PATH, "models", self.model_name)
            os.makedirs(self.folder_path)

            self.env = Monitor(self.env, filename=self.folder_path)

            self.tensorboard_log = self.folder_path

            if callback is not None:
                callback.log_path = self.folder_path

            # save metadata, pre training!
            para_file_path = self.env.c.__file__
            para_file = None
            with open(para_file_path, "r") as f:
                para_file = f.read()
            metadata_path = os.path.join(self.folder_path, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(para_file)

        # create Model
        if self.model is None:
            self.create_model()
        # Training
        t1 = datetime.datetime.now()
        try:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            pass
        t2 = datetime.datetime.now()
        self.training_time = smooth_timedelta(t1, t2)
        print("Training Done")
        # Save Model data
        if save_model:
            self.save_model()
            self.eval()
        self.env.training = False

    def save_model(self):
        # save model
        model_path = os.path.join(self.folder_path, "model.h5")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")
        # save metadata
        try:
            history_path = os.path.join(self.folder_path, "parameter_log.pcl")
            with open(history_path, "rb") as f:
                paras = pickle.load(f)
            paras["history"] = self.env.history
            with open(history_path, "wb") as f:
                pickle.dump(paras, f)
        except Exception as e:
            yellow("History file not saved!")
            print(e)
            print(type(e))
            IPS()

        try:
            metadata_path = os.path.join(self.folder_path, "metadata.txt")
            with open(metadata_path, "a") as f:
                f.writelines(
                    [
                        "\n",
                        f"\nTotal Training Steps: {self.env.total_step_count}",
                        f"\nTotal Episode Count: {self.env.episode_count}",
                        f"\nTraining Duration: {self.training_time}",
                        f"\nNet Arch: {self.model.policy.net_arch}",
                    ]
                )
        except Exception as e:
            yellow("Parameter file not saved!")
            print(e)
            print(type(e))
            IPS()

    def load_model(self, name):
        if self.model is None:
            self.create_model()
        path = os.path.join("models", name, "model.h5")
        # self.model = PPO.load(path)
        self.model.set_parameters(path)
        self.model_name = name

    def get_action(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

    def play(self, num_ep=10, render=True):
        self.env.target_change_period = 500
        for i in range(num_ep):
            done = trunc = False
            obs, _ = self.env.reset()
            r_sum = 0
            while not done and not trunc:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.env.step(action)
                r_sum += reward
            print("Reward Ep ", i, r_sum)

    def eval(self):
        xs = []
        phis = []
        inside_tol = []
        total_cost = 0
        episodes = 3
        steps = 1000
        for i in range(episodes):
            obs, _ = self.env.reset(state=np.array(self.env.c.EVAL_STATE) / (i + 2))
            for j in range(steps):
                xs.append(obs[0])
                phis.append(obs[2])
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.env.step(action)
                # add cost
                total_cost += (obs.T @ self.env.c.Q @ obs + action.T * self.env.c.R * action)[0]
                if done:
                    break

        phis.reverse()
        xs.reverse()
        phi_tolerance = self.env.c.phi_tolerance
        x_tolerance = self.env.c.x_tolerance
        for i in range(episodes):
            for j, phi in enumerate(phis[i * 1000 : (i + 1) * 1000]):
                if np.abs(phi) > phi_tolerance or np.abs(xs[steps * i + j]) > x_tolerance:
                    inside_tol.append((episodes - i) * steps - j)
                    break
        phis.reverse()
        xs.reverse()

        tol_lines = [-phi_tolerance, phi_tolerance, -x_tolerance, x_tolerance]
        plt.hlines(tol_lines, xmin=0, xmax=steps * episodes, colors="gray", linewidth=0.5)
        plt.plot(np.arange(len(xs)), xs, label=r"$x$")
        plt.plot(np.arange(len(phis)), phis, label=r"$\varphi$")
        plt.vlines(inside_tol, ymin=min(phis), ymax=max(phis), colors="red")
        plt.legend()
        plt.title("Evaluation Episodes")
        path = os.path.join("models", self.model_name, "eval.pdf")
        plt.savefig(path, format="pdf")
        plt.clf()

        with open(os.path.join("models", self.model_name, "eval.txt"), "w") as f:
            f.write(f"total cost: {round(total_cost, 4)}\n")
            f.write(f"av cost per step: {round(total_cost / (episodes*steps), 4)}\n")
            f.write(f"mean steps until steady state: {round(np.mean(inside_tol) - steps, 4)}\n")

        path = os.path.join("models", self.model_name, "parameter_log.pickle")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h = pickle.load(f)
            h = h["history"]
            terminated_idxs = np.where(h["terminated"])[0]

            assert len(terminated_idxs) == h["episode"][-1] - h["episode"][0]

            # step by step data:
            plt.plot(h["step"], h["action"], label="Action")
            plt.plot(h["step"], h["reward"], label="Reward", linestyle="dashed")
            plt.vlines(terminated_idxs, ymin=-4, ymax=-3, colors="b")
            plt.xlabel("Step")
            plt.legend()
            plt.savefig(os.path.join("models", self.model_name, "step_data.pdf"), format="pdf")
            plt.clf()

            # episode data
            total_rews = []
            total_cost = []
            ep_length = []
            start = 0
            for i in range(len(terminated_idxs)):
                end = terminated_idxs[i]

                total_rews.append(sum(h["reward"][start:end]))
                ep_length.append(end - start)
                cost = 0
                for k in range(start, end, 1):
                    cost += (
                        h["state"][k].T @ self.env.c.Q[: len(h["state"][k]), : len(h["state"][k])] @ h["state"][k]
                        + h["action"][k] ** 2 * self.env.c.R
                    )
                total_cost.append(cost)

                start = end

            fig, ax = plt.subplots(2, 2)
            ax[0, 0].plot(np.arange(2, h["episode"][-1], 1), total_rews)
            ax[0, 0].set_title("Total Reward")

            ax[1, 0].plot(np.arange(2, h["episode"][-1], 1), total_cost)
            ax[1, 0].set_title("Total Cost")
            ax[1, 0].set_xlabel("Episode")
            ax[1, 0].sharex(ax[0, 0])

            ax[0, 1].plot(np.arange(2, h["episode"][-1], 1), ep_length, label="Episode Length")
            ax[0, 1].set_title("Episode Length")
            ax[0, 1].set_xlabel("Episode")
            fig.tight_layout()
            plt.savefig(os.path.join("models", self.model_name, "episode_data.pdf"), format="pdf")
            plt.clf()
