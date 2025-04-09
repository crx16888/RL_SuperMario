import gym
from stable_baselines3.ppo.ppo import PPO 
from stable_baselines3.a2c.a2c import A2C
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros # gym_super_mario_bros（超级马里奥游戏和gym的集成），gym做通用功能，gym_super_mario_bros做马里奥游戏相关功能
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation # 导入灰度图包装器
from test_obs import make_env
import matplotlib.pyplot as plt
from numpy import shape
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
def main():
    env = SubprocVecEnv([make_env for _ in range(4)]) # 多进程向量化环境
    env = VecFrameStack(env, 4, channels_order='last') # 向量化帧堆叠作为输入
    # 回调=定期评估模型
    eval_callback = EvalCallback(env, eval_freq=10000//8, n_eval_episodes=1,  #暂停训练，使用当前模型test一个完整回合用于评估
                                 log_path=r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\logs',
                                 best_model_save_path=r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\best_model')
    # 每次更新数据前会收集2048x8个数据，然后分n_epochs=10次每次训练batch个数据用于更新
    # 训练要既考虑cpu内存（存数据和图像数据）还要考虑gpu显存（处理图像），要让他们同时处于一个中高的位置而不是一个爆掉
    model = PPO("CnnPolicy", env, verbose=1,
               n_steps=2048, # 每一轮存储的数据，影响内存
            #    batch_size=2048*4,
               batch_size=2048, # 处理，主要影响显存
               tensorboard_log=r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\logs', 
               learning_rate=3e-4, #很可能模型一开始更新进行剧烈的我们不应该给它限制，所以注释了kl散度
            #    如果不喜欢后面震荡太厉害，那么可以减小学习率；但是收敛就会很慢
               ent_coef=0.1,
               n_epochs=4,
            #    target_kl=0.1 # kl散度大于0.1就停止更新
               gae_lambda=0.97  # 短视还是长远
               )
        # # r如果中断训练，下次从最好的模型参数这里开始训练
    # model = PPO.load(
        # r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\best_model\best_model.zip',env=env
    # "CnnPolicy", env, verbose=1,
            #    n_steps=2048, # 每一轮存储的数据，影响内存
            # #    batch_size=2048*4,
            #    batch_size=2048, # 处理，主要影响显存
            #    tensorboard_log=r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\logs', 
            #    learning_rate=3e-4, #很可能模型一开始更新进行剧烈的我们不应该给它限制，所以注释了kl散度
            # #    如果不喜欢后面震荡太厉害，那么可以减小学习率；但是收敛就会很慢
            #    ent_coef=0.1,
            #    n_epochs=4,
            # #    target_kl=0.1 # kl散度大于0.1就停止更新
            #    gae_lambda=0.97  # 短视还是长远) 
    model.learn(total_timesteps=int(1e7),callback=eval_callback)
    model.save(r'C:\Users\95718\Desktop\vscode\RL\RL_SuperMario_Study\mario_model.zip')
    # 训练文件，训练完了模型以后可以用test文件去做测试

if __name__ == '__main__':
    main()
    # 画图的测试代码：只是为了训练完了先简单看一下效果
    env = make_env()
    done = True
    for step in range(4):
        if done:
            obs = env.reset()
        obs,reward,done,info = env.step(env.action_space.sample())
        plt.imshow(obs,cmap='gray')
        plt.show()
        # env.render()
        print(shape(obs))     
    env.close()