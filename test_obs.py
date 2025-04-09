# 此处为测试环境所用的代码
# 分别对环境做灰度处理、跳帧处理、图像降维处理以减少训练量
# 需要对gym库里面的三个包好好学一学
from numpy import shape
import matplotlib.pyplot as plt
import gym
from stable_baselines3.ppo.ppo import PPO
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation,ResizeObservation # 灰度图包装器、图像降维包装器
from my_wrapper import SkipFrame #跳帧的包gym库的wrappers里没定义，所以我们自己写了一个放在了my_wrapper里


def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v2')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 就当前这个简单的代码，就算轮数1000后面再加无数个0马里奥也很难训练过第一关；
    # 所以我们需要对env做一些预处理，使得能够更轻松地根据这个环境训练好智能体
    # Wrappers包装器能够帮助不用修改底层代码，更好地修改环境；每一个包装器都会继承前一个包装器作为参数进一步包装
    # env = SkipFrame(env,skip=4) # 跳帧，马里奥每帧移动的太小，每8帧取1帧送给算法进行处理
    env = GrayScaleObservation(env,keep_dim=True) # 修改3通道为1的灰度图，训练量更小
    env = ResizeObservation(env,shape=(84,84)) # 图像降维
    # 注释掉是因为不进行下采样会让gpu参与更多图像处理工作，处理更大的图片
    # 不注释掉是因为cpu内存爆了，必须对图像数据降维
    return env

if __name__ == '__main__':
    env = make_env()
    done = True
    for step in range(4):
        if done:
            obs = env.reset()
        obs,reward,done,info = env.step(env.action_space.sample())
        plt.imshow(obs,cmap='gray')
        plt.show()
        # env.render()


