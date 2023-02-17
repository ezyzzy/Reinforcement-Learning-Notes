# Reinforcement Learning - ShusenWang

* [深度强化学习(1_5)：基本概念 Deep Reinforcement Learning (1_5)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV12o4y197US/?buvid=XY800DF8CDA7AB0BE04B78A8D0C07BB59E650&is_story_h5=false&mid=eufKE7OFCU3ZXpBNx166EQ%3D%3D&p=1&plat_id=114&share_from=ugc&share_medium=android&share_plat=android&share_session_id=700db6c1-8eb2-4119-8d0a-cdd7880d960d&share_source=WEIXIN&share_tag=s_i&timestamp=1676090767&unique_k=mK7CTlC&up_id=52065264&vd_source=96d295f94289887fe4a4e50f1f408c24)
* [DRL/Slides at master · wangshusen/DRL · GitHub](https://github.com/wangshusen/DRL/tree/master/Slides)



## 1. 基本概念

### basic knowledge

* random variable $X$, observed value $x$
* PDF (Probability Density Function) ~ likelihood

* Random Sampling

```python
from numpy.random import choice
```

* ==state== & ==action== & ==agent==

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217095917906.png" alt="image-20230217095917906" style="zoom:67%;" />

* ==policy== $\pi$ $\pi(s,a)$

​          **策略：根据观测到的state做出决策控制agent运动**

​		  <u>RL学习的对象就是policy</u>

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217100303857.png" alt="image-20230217100303857" style="zoom:67%;" />

​           随机的目的是对手不知道自己的行为

* ==reward==

​    非常影响学习结果，**RL目标是reward尽量高**

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217100554543.png" alt="image-20230217100554543" style="zoom:67%;" />

* ==state transition==

​      **状态转移函数$p(s^`|s,a)$只有environment知道，agent不知道。**

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217100759794.png" alt="image-20230217100759794" style="zoom:67%;" />



### Randomness

* ==randomness 随机性==（1. policy 2. state transition）
* *    policy => action

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217101319612.png" alt="image-20230217101319612" style="zoom:67%;" />

* * ​    state transition => state

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217101436011.png" alt="image-20230217101436011" style="zoom:67%;" />

* ==trajectory==

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217101103045.png" alt="image-20230217101103045" style="zoom:67%;" />

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217102215676.png" alt="image-20230217102215676" style="zoom:67%;" />



### Reward & Return

* ==Return== $U$ & reward $R$

​		**回报**是**未来的奖励总和，越大越好。**

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217102858219.png" alt="image-20230217102858219" style="zoom:67%;" />

​		**未来的奖励没有现在的奖励值钱，所以需要打个折扣。**==Discounted==

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217103023425.png" alt="image-20230217103023425" style="zoom:67%;" />

​		hyperparameter ==discounted rate== $\gamma$ 

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217103418570.png" alt="image-20230217103418570" style="zoom:67%;" />

​		为什么U和R用大写字母，因为还<u>未被观察，是*随机*事件</u>。

​		**意思就说每一步policy取决于此刻的Return。**



### Value Function

U 是时刻t后的奖励总和的随便变量

$Q_\pi$ 是时刻t后的奖励总和的期望

$Q^*$ 是时刻t后找到最优策略$\pi$后的最大化奖励总和的期望

$V_\pi$ 是时刻t时不同action对应的$Q_\pi$的期望

* ==Action-value function== $Q_\pi$

​		$U_t$ 是个<u>随机变量</u>，$Q_\pi$ 是<u>期望</u>。

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217105509662.png" alt="image-20230217105509662" style="zoom:67%;" />

* ==Optimal action-value function== $Q^*$

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217105646486.png" alt="image-20230217105646486" style="zoom:67%;" />

​		**时刻 t 找到最好的 policy $\pi$ 让 Q 最大化**

* ==State-value function== 状态价值函数 $V_\pi$

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217110617065.png" alt="image-20230217110617065" style="zoom:67%;" />

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217110637250.png" alt="image-20230217110637250" style="zoom:67%;" />

​		用于判断<u>当前局势好不好</u>

* action & state - value function

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217110809204.png" alt="image-20230217110809204" style="zoom:67%;" />

​		前者评价 action，后者评价 ==situation==， 后者的期望评价 policy

* **目标函数** 策略学习 & 价值学习

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217111228931.png" alt="image-20230217111228931" style="zoom:67%;" />



### GAMES via RL

* OpenAI Gym: https://gym.openai.com/

```python
import gym
env = gym.make('CartPole-v0')
```

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217111846828.png" alt="image-20230217111846828" style="zoom:67%;" />



### Summary

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217112427508.png" alt="image-20230217112427508" style="zoom:67%;" />

<img src="C:\Users\11066\AppData\Roaming\Typora\typora-user-images\image-20230217112552821.png" alt="image-20230217112552821" style="zoom:67%;" />

​		核心思想奖励尽量多。

​		学$\pi$或者$Q^*$

