import gym
from stable_baselines3 import A2C,PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation # 灰度图包装器
from test_obs import make_env

env = make_env()
model = PPO.load('C:/Users/95718/Desktop/vscode/RL/RL_SuperMario_Study/mario_model.zip', env=env)
obs = env.reset()
for i in range(1000):
    obs = obs.copy()  # 复制数组以避免负步长问题
    action, _state = model.predict(obs, deterministic=True) # 每一步使用模型预测并执行动作，显示游戏画面；而不更新模型参数
    obs, reward, done, info = env.step(action) # 因为make_env中重写了step，skip为一步
    env.render()
    if done:
      obs = env.reset()