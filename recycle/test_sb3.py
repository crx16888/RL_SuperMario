# 核心的包只有两个：stable_baselines3（强化学习框架，集成了各种算法）和gym_super_mario_bros（超级马里奥游戏和gym的集成）‘
import gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1")

model = A2C("MlpPolicy", env, verbose=1,tensorboard_log='logs')
model.learn(total_timesteps=1000000)

obs = env.reset()
for i in range(1000):
    obs = obs.copy()  # 复制数组以避免负步长问题
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()