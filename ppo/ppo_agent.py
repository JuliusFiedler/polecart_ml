import gymnasium
import sys, os
import datetime
import matplotlib.pyplot as plt
import pickle
import copy
import torch as th
from ipydex import IPS

sys.modules["gym"] = gymnasium
import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from classical.classical_control import F_LQR_2 as LQR_IDEAL
from agents.agent import BaseAgent
from util import *



class PPOAgent(BaseAgent):
    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(env)
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

    def train(self, total_timesteps=300000, callback=None, save_model=True, eval=True):
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
            if eval:
                self.eval()
        self.env.training = False

    def save_model(self):
        # save model
        model_path = os.path.join(self.folder_path, "model.h5")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")
        # save metadata
        try:
            history_path = os.path.join(self.folder_path, "training_logs.p")
            with open(history_path, "rb") as f:
                paras = pickle.load(f)
            paras["training_history"] = self.env.history
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

    def get_action(self, obs, *args):
        return self.model.predict(obs, deterministic=True)[0]

    def get_value(self, state):
        if isinstance(state, np.ndarray):
            state = th.from_numpy(state)
        values = self.model.policy.predict_values(state)[:, 0].detach().numpy()
        return values

    def get_real_state_value(self, obs):
        env = copy.copy(self.env)
        env.render_mode = False
        env.reset(state=obs)
        done = False
        gamma = 0.99
        i = 0
        V = env.c.get_reward(env)[0]
        while not done:
            # if i%10 == 9:
            #     print(i)
            action = self.get_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            V += gamma**i * rew
            if term or trunc:
                done = True
            if abs(gamma**i * rew) < 1e-7:
                done = True
            i += 1
        return V

    def play(self, num_ep=10, render=True):
        self.env.target_change_period = 500
        for i in range(num_ep):
            done = trunc = False
            obs, _ = self.env.reset()
            r_sum = 0
            k = 0
            while not done and not trunc:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.env.step(action)
                # if k % 20 == 0:
                #     print(
                #         "real - est",
                #         self.get_real_state_value(obs)
                #         - self.model.policy.predict_values(th.tensor(np.array([obs]))).detach().numpy(),
                #     )
                r_sum += reward
                k += 1
            # print("Reward Ep ", i, r_sum)

    def eval(self):
        plots = []
        self.run_eval_episodes()
        from visualize_nets import visualize_actions_values
        self.folder_path = os.path.join(ROOT_PATH, "models", self.model_name)
        if self.env.observation_space.shape == (4,):
            visualize_actions_values(self.env, self, self.folder_path, self.model_name)

        # visualize training
        path = os.path.join("models", self.model_name, "training_logs.p")
        if os.path.isfile(path):
            with open(path, "rb") as f:
                log = pickle.load(f)
            h = log["training_history"]
            terminated_idxs = np.where(np.logical_or(h["terminated"], h["truncated"]))[0]

            assert len(terminated_idxs) == h["episode"][-1] - h["episode"][0]

            # step by step data:
            plt.plot(h["step"], h["action"], label="Action")
            plt.plot(h["step"], h["reward"], label="Reward", linestyle="dashed")
            plt.vlines(terminated_idxs, ymin=-4, ymax=-3, colors="b")
            plt.xlabel("Step")
            plt.legend()
            plt.savefig(os.path.join("models", self.model_name, "step_data.pdf"), format="pdf")
            plots.append(copy.copy(plt.gcf()))
            plt.clf()

            # process data
            total_rews = []
            avg_rews = []  # avg reward per step during an episode
            total_cost = []
            avg_cost = []  # avg cost per step during an episode
            ep_length = []
            rollout_length_ep = []
            steps_at_term_idx = []
            rollout_length_st = []
            start = 0
            for i in range(len(terminated_idxs)):
                end = terminated_idxs[i]

                total_rews.append(sum(h["reward"][start:end]))
                avg_rews.append(sum(h["reward"][start:end]) / (end - start))
                steps_at_term_idx.append(h["step"][end])
                ep_length.append(end - start)
                if i + 1 < len(log["NN_updates"]["episode"]):
                    rollout_length_ep.append(log["NN_updates"]["episode"][i + 1] - log["NN_updates"]["episode"][i])
                if i + 1 < len(log["NN_updates"]["step"]):
                    rollout_length_st.append(log["NN_updates"]["step"][i + 1] - log["NN_updates"]["step"][i])
                cost = 0
                for k in range(start, end, 1):
                    cost += (
                        h["state"][k].T @ self.env.c.Q[: len(h["state"][k]), : len(h["state"][k])] @ h["state"][k]
                        + h["action"][k] ** 2 * self.env.c.R
                    )
                total_cost.append(cost)
                avg_cost.append(cost / (end - start))

                start = end

            # episode data, reward, cost, length
            fig, ax = plt.subplots(3, 2, figsize=(10, 15))
            ax[0, 0].plot(np.arange(2, h["episode"][-1], 1), total_rews)
            ax[0, 0].set_title("Total Reward per Episode")
            ax[0, 0].set_xlabel("Episode")
            ax[0, 0].grid()
            ax00_2 = ax[0, 0].twiny()
            ax00_2.plot(steps_at_term_idx, total_rews, linestyle="dashed", color="tab:orange")
            ax00_2.set_xlabel("Total Step at End of Episode")

            ax[0, 1].plot(np.arange(2, h["episode"][-1], 1), ep_length, label="Episode Length")
            ax[0, 1].set_title("Episode Length")
            ax[0, 1].set_xlabel("Episode")
            ax[0, 1].grid()
            ax01_2 = ax[0, 1].twiny()
            ax01_2.plot(steps_at_term_idx, ep_length, linestyle="dashed", color="tab:orange")
            ax01_2.set_xlabel("Total Step at End of Episode")

            ax[1, 0].plot(np.arange(2, h["episode"][-1], 1), total_cost)
            ax[1, 0].set_title("Total Cost")
            ax[1, 0].set_xlabel("Episode")
            ax[1, 0].grid()
            ax10_2 = ax[1, 0].twiny()
            ax10_2.plot(steps_at_term_idx, total_cost, linestyle="dashed", color="tab:orange")
            ax10_2.set_xlabel("Total Step at End of Episode")

            ax[1, 1].plot(np.arange(len(rollout_length_ep)), rollout_length_ep, label="Rollout Length")
            ax[1, 1].set_title("Rollout Length")
            ax[1, 1].set_xlabel("Nr. Update")
            ax[1, 1].set_ylabel("Episodes")
            ax11_2 = ax[1, 1].twinx()
            ax11_2.plot(
                np.arange(len(rollout_length_st)), rollout_length_st, label="Rollout Length", linestyle="dashed"
            )
            ax11_2.set_ylabel("Steps")
            ax[1, 1].grid()

            ax[2, 0].plot(np.arange(2, h["episode"][-1], 1), avg_rews)
            ax[2, 0].set_title("Avg. Reward per Episode")
            ax[2, 0].set_xlabel("Episode")
            ax[2, 0].grid()
            ax20_2 = ax[2, 0].twiny()
            ax20_2.plot(steps_at_term_idx, avg_rews, linestyle="dashed", color="tab:orange")
            ax20_2.set_xlabel("Total Step at End of Episode")

            ax[2, 1].plot(np.arange(2, h["episode"][-1], 1), avg_cost, label="Episode Length")
            ax[2, 1].set_title("Avg. Cost")
            ax[2, 1].set_xlabel("Episode")
            ax[2, 1].grid()
            ax21_2 = ax[2, 1].twiny()
            ax21_2.plot(steps_at_term_idx, avg_cost, linestyle="dashed", color="tab:orange")
            ax21_2.set_xlabel("Total Step at End of Episode")

            fig.tight_layout()
            fig.suptitle(self.model_name)
            fig.subplots_adjust(top=0.85)
            plt.savefig(os.path.join("models", self.model_name, "episode_data.pdf"), format="pdf")
            plots.append(plt.gcf())
            plt.clf()

            # progression of linearized NN
            fig, ax = plt.subplots()
            ax2 = ax.twinx()
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            for i in range(len(log["NN_updates"]["linearization"][0])):
                ax.plot(
                    log["NN_updates"]["episode"],
                    np.array(log["NN_updates"]["linearization"])[:, i],
                    label=f"$K_{i+1}$",
                    color=colors[i],
                )
                ax.scatter(log["NN_updates"]["episode"][-1], -LQR_IDEAL["F"][0, i], label=f"$LQR K_{i}$", c=colors[i])
            ax.legend()
            ax.set_xlabel("Episode")
            ax.grid()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(log["NN_updates"]["linearization"][-1])
            ax2.set_yticklabels(log["NN_updates"]["linearization"][-1])
            fig.suptitle("Evolution of linearized NN")
            plt.savefig(os.path.join("models", self.model_name, "linearization_log.pdf"), format="pdf")
            plots.append(plt.gcf())
            plt.clf()

            # add all plots to one big plot
            # fig, ax = plt.subplots(2, 2)
            # for i in range(ax.shape[0]):
            #     for k in range(ax.shape[1]):
            #         try:
            #             ax[i, k] = plots[2 * i + k]
            #         except IndexError:
            #             pass
            # fig.suptitle(self.model_name)
            # plt.savefig(os.path.join("models", self.model_name, "Info.pdf"), format="pdf")
