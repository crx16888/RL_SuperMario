# 因为跳帧的类没有包装，所以我们必须自己写一个
import gym
class SkipFrame(gym.Wrapper): #SkipFrame继承父类gym.Wrapper
    def __init__(self, env, skip):
        super().__init__(env) #调用父类的初始化方法初始化父类中的env，这样self就可以调用父类中的env了
        self.skip = skip #self可以调用skip

    def step(self, action):
        obs,reward_total,done,info = None,0,False,None
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            reward_total += reward
            if done:
                break
        # 每skip帧取1帧，才返回结果给智能体，然后算法用这一帧结果去训练
        return obs, reward_total, done, info