
'''
关于该竞赛解题思路的recording
以下内容作为状态输入
执行性声明性状态S1 init=empty 用于调度任务                                              //不能记录离线时间,因为不能预知未来
exeid categoryid whenfree devicetype(CLOUD BS XUEX) deviceid comspeed rate acpu amem onlinetime offlinetime(online?2023年6月3日:并不会出入非online的资源状态作为S,这一维度是否必要?) dim=
0           0       0                  0    0  1      1       xxx     xx    xx  xx      xx                     1 
0           0       0                  0    0  1      1       xxx     xx    xx  xx      xx                     1 
xx          xx      1s(1秒后)          1    0   0      0        xx      xx      xx      xx                      0
    
S2已经增加至S1中(2023年6月2日)
资源性声明状态S2 init=loadcsv TODO ALL avanode -dynamic id  or  mask unava node -static id  用于部署执行者 
devicetype deviceid acpu amem comspeed  onlinetime  dim=7

所到达的task声明S3 init with time(disperse+)
time taskid jobid categoryid parentid pardata childid childata preraretime rcpu rmem dim=11

输出部署动作A1
Time,Action,ExecutorId,CategoryId,DeviceType,DeviceId 
A1具体形式待敲定 可能为空 可能为+可能为-

输出调度动作A2
time task exeid

'''
# Import
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
#全局变量
num2item={'exeid':0,'categoryid':1,'whenfree':2,'Cloud':3,'Bs':4,'deviceid':5,'comspeed':6,'rate':7,'acpu':8,'amem':9,'online':10}

#实际上，这就是状态（服务状态）类，有关方法均在此类中定义，Env则调用，如在step（）中
class ExecutorInit:
    def __init__(self,bsmetric,bstable,cloudtable, hosttable):
        ''' self.exeid ,self.categoryid,self.whenfree,self.Cloud,self.Bs,self.deviceid,self.comspeed,self.rate,self.acpu,self.amem,self.online=\
            exeid,       categoryid,     whenfree,     Cloud,     Bs,     deviceid,     comspeed,     rate,     acpu,     amem,     online
            '''
        #trans multiple csv data -> init Executor info as input state
        exes=[]

        nowtime=0
        #先添加cloud上的host，再bs，在online的ue
        hostid,_,cpu,mem,speed=hosttable[['HostId','CloudId','CPU','Memory','ComputeFactor']].values[0]
        exeid=categoryid=whenfree=-1
        Cloud,Bs,online=1,0,1
        deviceid=hostid
        comspeed=speed
        acpu,amem=cpu,mem
        rate=cloudtable['Rate'].values[0]
        exes.append([exeid,categoryid,whenfree,Cloud,Bs,deviceid,comspeed,rate,acpu,amem,online])
        #end Cloud host init/start bs init
        initbs=bsmetric.loc[bsmetric['Time']==0,:]
        for _ , row in initbs.iterrows():
            Cloud,Bs,online=0,1,1
            exeid=categoryid=whenfree=-1
            deviceid,_,acpu,amem=row['BSId'],row['Time'],row['CPU'],row['Memory']
            _,rate,comspeed=bstable.loc[bstable['BSId']==deviceid,:].values[0]
            exes.append([exeid,categoryid,whenfree,Cloud,Bs,deviceid,comspeed,rate,acpu,amem,online])
        #end bs init/All ue offline
        # c=ExecutorInit(bsmetric,bstable,cloudtable, hosttable,uetable,uemetric) initexe=c.exes to get result as init input #
        self.exes=exes
        
    #NOTE：每时间步长都应该调用 OR task/job到达间隔作为步长???，bs和cloud应该在600s调用
    """usage,duration despite on your step 
    c.updateUe(599,601,uetable,uemetric)
    c.updateUe(601,1111,uetable,uemetric)
    c.updateUe(1111,1159,uetable,uemetric)
    """
    def updateUe(self,lasttime,nowtime,uetable,uemetric):
        tmp_offid=[]
        Online=uetable[(uetable['OnlineTime']<=nowtime) &(uetable['OnlineTime']>lasttime)& (uetable['OfflineTime']>nowtime)]
        for _ , row in Online.iterrows():
            #print(row['UEId'],row['ComputeFactor'],row['OnlineTime'],row['OfflineTime'])
            tmp_a=uemetric.loc[(uemetric['UEId']==row['UEId'])&(uemetric['Time']==nowtime),].iloc[0]
            #print(a['UEId'],a['Time'],a['Rate'],a['BSId'],a['CPU'],a['Memory'])
            deviceid,comspeed,rate,acpu,amem,online=row['UEId'],row['ComputeFactor'],tmp_a['Rate'],tmp_a['CPU'],tmp_a['Memory'],1
            self.exes.append([-1,-1,-1,0,0,deviceid,comspeed,rate,acpu,amem,online])
        #
        Offline=uetable[(uetable['OfflineTime']<=nowtime) &(uetable['OfflineTime']>lasttime)]
        for _ , row in Offline.iterrows():
            tmp_offid.append(row['UEId'])
        self.exes = list(filter(lambda x: (x[3] != 0) | (x[4]!=0)|(x[5] not in tmp_offid), self.exes))
        #TODO：cloud-host和bs在600s处的更新，list筛选+赋值
        
    def updataNowtime(self):
        self.nowtime=self.nowtime+1

    def to_input(self): #输出系列值作为nn状态输入，需要从list转为nparry
        inputs_arr = np.array(self.exes)
        inputs_arr = inputs_arr.reshape(inputs_arr.shape[0], 1, inputs_arr.shape[1])
        #print(inputs_arr.shape)应该是这个（n,1,11)
        return inputs_arr
    
    def addexe(self,):   #执行Add动作部署任务时

        pass
    def delexe():   #执行Del删除某一任务时

        pass
    def scheduletask(): #调度具体任务时

        pass
    def updatewithtime(): #随时间更新，如设备下线等等

        pass




'''
@param:datapath = 前缀路径,直到test0之前
@param:ff 多个test路径,这里分开写是为了方便后续加入迭代
@return dict
example:
load=loadalldata('./dataset/testset/','test0')
bsmetric=load['bsmetric']
'''

def loadalldata(datapath,ff):
    #BSId,Time,CPU,Memory.small scale
    bsmetric= pd.read_csv(os.path.join(datapath,ff,'bs_metric.csv'))
    #BSId,Time,CPU,Memory.small
    bstable=pd.read_csv(os.path.join(datapath,ff,'bs_table.csv'))
    #CloudId,Rate.small
    cloudtable=pd.read_csv(os.path.join(datapath,ff,'cloud_table.csv'))
    #HostId,CloudId,CPU,Memory,ComputeFactor.small
    hosttable=pd.read_csv(os.path.join(datapath,ff,'host_table.csv'))
    #JobId,ArriveTime
    jobtable= pd.read_csv(os.path.join(datapath,ff,'job_table.csv'))
    #TaskId,JobId,CategoryId,ParentTasks,ChildTasks,ComputeDuration
    tasktable=pd.read_csv(os.path.join(datapath,ff,'task_table.csv'))
    #CategoryId,RequestCPU,RequestMemory,PrepareDuration
    categorytable=pd.read_csv(os.path.join(datapath,ff,'category_table.csv'))
    #UEId,ComputeFactor,OnlineTime,OfflineTime
    uetable=pd.read_csv(os.path.join(datapath,ff,'ue_table.csv'))
    #UEId,Time,Rate,BSId,CPU,Memory
    uemetric=pd.read_csv(os.path.join(datapath,ff,'ue_metric.csv'))
    #num_ue=max(uetable['UEId'])
    return {'bsmetric':bsmetric,'bstable':bstable,
            'cloudtable':cloudtable,'hosttable':hosttable,
            'jobtable':jobtable,'tasktable':tasktable,
            'categorytable':categorytable,'uetable':uetable,'uemetric':uemetric}


def formatS(S1,S2):
    pass
    return S1,S2

def transA1(A1):
    pass
    return A1

def exeA1():
    pass


# Define the environment class
class Env:
    def __init__(self,rootfile,epsfile):
        #load data from rootfile/epsfile/xxx.csv
        load=loadalldata(rootfile,epsfile)
        bsmetric,bstable,cloudtable, hosttable, jobtable,tasktable,categorytable,uetable,uemetric =\
              load['bsmetric'],load['bstable'],load['cloudtable'],load['hosttable'],load['jobtable'],load['tasktable'],load['categorytable'],load['uetable'],load['uemetric']

    # Executor().__init__()

        
    '''        
    #dim as input's shape
        self.state_dim = 7
        self.action_dim = 7
        self.max_action = 11
    '''
    #并不需要reset 每次直接读取到下一个trainset就好了
    def reset(self):
        pass

    
    def step(self, action):
        next_state = np.zeros(self.state_dim)
        next_state[0] = self.state[0] + action[0]
        next_state[1] = self.state[1] + action[0]
        self.state = next_state

        reward = np.sum(self.state)

        done = False 
        return next_state, reward, done, {}
    def getreward():
        #reward NOTE simple complete task/job rate and delay
        pass
#end Custom Env


# Define the DDPG agent class
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Define the actor and critic networks
        self.actor = self.create_actor_network()
        self.critic = self.create_critic_network()
        
        # Define the target actor and critic networks
        self.target_actor = self.create_actor_network()
        self.target_critic = self.create_critic_network()
        
        # Initialize the target actor and critic networks with the same weights as the actor and critic networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Define the optimizer for the actor and critic networks
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def create_actor_network(self):
        # Define the input layer
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        
        # Define the hidden layers
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Define the output layer
        outputs = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        outputs = tf.keras.layers.Lambda(lambda x: x * self.max_action)(outputs)
        
        # Define the actor network
        actor = tf.keras.Model(inputs, outputs)
        
        return actor
    
    def create_critic_network(self):
        # Define the input layers
        state_inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        action_inputs = tf.keras.layers.Input(shape=(self.action_dim,))
        
        # Define the hidden layers for the state inputs
        x = tf.keras.layers.Dense(256, activation='relu')(state_inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Define the hidden layers for the action inputs
        y = tf.keras.layers.Dense(256, activation='relu')(action_inputs)
        
        # Combine the state and action inputs
        xy = tf.keras.layers.Concatenate()([x, y])
        
        # Define the output layer
        outputs = tf.keras.layers.Dense(1)(xy)
        
        # Define the critic network
        critic = tf.keras.Model([state_inputs, action_inputs], outputs)
        
        return critic
    
    def update_target_networks(self, tau):
        # Update the target actor and critic networks with the weights of the actor and critic networks
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)
        
    def train(self, replay_buffer, batch_size, gamma, tau):
        # Sample a batch of transitions from the replay buffer
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.sample(batch_size)
        
        # Convert the numpy arrays to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        done_batch = tf.convert_to_tensor(done_batch, dtype=tf.float32)
        
        # Compute the target Q values using the target actor and critic networks
        target_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic([next_state_batch, target_actions])
        target_q_values = reward_batch + gamma * target_q_values * (1 - done_batch)
        
        # Compute the Q values using the critic network
        with tf.GradientTape() as tape:
            q_values = self.critic([state_batch, action_batch])
            critic_loss = tf.keras.losses.MSE(target_q_values, q_values)
        
        # Compute the gradients of the critic loss with respect to the critic network weights
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        
        # Apply the gradients to the critic network using the critic optimizer
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Compute the gradients of the actor loss with respect to the actor network weights
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            actor_loss = -tf.math.reduce_mean(self.critic([state_batch, actions]))
        
        # Compute the gradients of the actor loss with respect to the actor network weights
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        # Apply the gradients to the actor network using the actor optimizer
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Update the target actor and critic networks
        self.update_target_networks(tau) 
#end DDPG

# Define the replay
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = []
        self.max_size = max_size
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        state_batch = np.array([self.buffer[i][0] for i in batch])
        action_batch = np.array([self.buffer[i][1] for i in batch])
        reward_batch = np.array([self.buffer[i][2] for i in batch])
        next_state_batch = np.array([self.buffer[i][3] for i in batch])
        done_batch = np.array([self.buffer[i][4] for i in batch])
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch
#end Replybuffer
   
# Define the hyperparameters
batch_size = 64
gamma = 0.99
tau = 0.001
buffer_size = 1000000
exploration_noise = 0.1
exploration_noise_decay = 0.99
exploration_noise_min = 0.01
max_episodes = 50 #50 train set
#every step is time in 1(s)
max_steps = 10 #should be 1200(s)

# Create the environment
env = Env()

# Create the DDPG agent
agent = DDPGAgent(env.state_dim, env.action_dim, env.max_action)

# Create the replay buffer
replay_buffer = ReplayBuffer(buffer_size)

# Train the agent
episode_rewards = []
for episode in range(max_episodes):
    state = env.reset()
    episode_reward = 0
    exploration_noise *= exploration_noise_decay
    exploration_noise = max(exploration_noise, exploration_noise_min)
    for step in range(max_steps):
        # Add exploration noise to  action emmm
        action = agent.actor(np.array([state]))[0]
        action += exploration_noise * np.random.randn(env.action_dim)
        action = np.clip(action, -env.max_action, env.max_action)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Add the transition to the replay buffer
        replay_buffer.add(state, action, reward, next_state, done)
        
        # Update the agent
        if len(replay_buffer.buffer) > batch_size:
            agent.train(replay_buffer, batch_size, gamma, tau)
        
        # Update the state and episode reward
        state = next_state
        episode_reward += reward
        
        # Check if the episode is done
        if done:
            break
    
    # Append the episode reward to the list of episode rewards
    episode_rewards.append(episode_reward)
    
    # Print the episode reward
    print('Episode:', episode, 'Reward:', episode_reward)

# Plot the episode rewards

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

