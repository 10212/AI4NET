
'''
关于该竞赛解题思路的recording
以下内容作为状态输入
执行性声明性状态S1 init=empty 用于调度任务                                              \\不能记录离线时间,因为不能预知未来     哎呦你干嘛~
exeid categoryid whenfree devicetype(CLOUD BS \\UE\\) deviceid comspeed rate acpu amem onlinetime offlinetime(online?2023年6月3日:并不会出入非online的资源状态作为S,这一维度是否必要?) dim=
0           0       0                  0    0  1      1       xxx     xx    xx  xx      xx                     1 
0           0       0                  0    0  1      1       xxx     xx    xx  xx      xx                     1 
xx          xx      1s(1秒后)          1    0   0      0        xx      xx      xx      xx                      0
    
S2已经增加至S1中(2023年6月2日)(disable x)
资源性声明状态S2 init=loadcsv TODO ALL avanode -dynamic id  or  mask unava node -static id  用于部署执行者 
devicetype deviceid acpu amem comspeed  onlinetime  dim=7

所到达的task声明S3 init with time
time taskid jobid categoryid parentid pardata childid childata preraretime rcpu rmem comduration  \\dim=11
now     N     x        x        x       x       x         x       x       x     x

输出部署动作A1
Time,Action,ExecutorId,CategoryId(根据taskgetid),DeviceType,DeviceId 
after(x),+-1 0, x   ,   x,          x,          x
A1具体形式待敲定 可能为空 可能为+可能为-

输出调度动作A2 time并不需要考虑进去 我只要parent complete就可以部署了 以parent complete存储类似于after的value  这也就是所谓的预计完成时间了
time taskid(get taskid) exeid
after(x),       x ,       x
2023年6月3日:已知的优化方向-使用numpy.ndarray取代list-据说性能更好
但是在dim和shape上面的东西还没理解-搁置
2023年6月5日 online是非必要的 只要在exes内的就是online 可以干脆点删掉 
在经验回放的加持下应该改为在线时长 尤其是对于UE 可能存在降下线的感知
#输出的联合动作
Action Exeid  devtype1 devtype2   devid     afterHowlongsec   
-1~1    many    0/1      0/1   many(omited)      continue
2023年6月8日 从5的基础上改为DQN without Replay buffer便于实现
目前真个循环并无问题 只是model太复杂 暂时改不起来 换个简单的先
2023年6月10日 现在看来 有许多源于理解的问题，比如 动作维度一减再减 输出的应该是概率还是指定动作 (软硬决策之分)
在动作输出部分 这里应该是del add两列 赋值 1则有对应操作

'''
# Import
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
#全局变量
#num2item={'exeid':0,'categoryid':1,'whenfree':2,'Cloud':3,'Bs':4,'deviceid':5,'comspeed':6,'rate':7,'acpu':8,'amem':9,'online':10}
#不好用

#这里的task/job 只适用于不断更新至下一条jod到达
class JobTask:
    def __init__(self,jobtable,tasktable,categorytable) :
        self.Nowtime=0 #
        self.Nowjobid=0
        self.jobarrlist=[]
        self.arrtask=[]
        self.alltask=[]
        self.alltasktime=[]
        self.taskindex=0
        self.tasktable,self.categorytable=tasktable,categorytable
        for _ ,  row in jobtable.iterrows():
            jobid=row['JobId']
            self.jobarrlist.append(row['ArriveTime']) #Now we get arr list with jobid  and  time \\不需要了
            for _, row2 in tasktable.iterrows():
                if row2['JobId']==jobid:
                    self.alltasktime.append(row['ArriveTime'])

            #现在我们有了一个以jodid为index，access by 'jobarrlist[jobid]=arrtime

    #time taskid jobid categoryid parentid pardata childid childata computeduration  rcpu rmem preraretime \\dim=11
    #now     N     x        x        x       x       x         x       x       x     x
    #here, return single task info show above . be called at state2 output
    def getsingletask(self,tasktable):
        re_list=[]
        taskindex=self.taskindex
        indexitem = tasktable.iloc[taskindex]
        rtaskid,rjobid,rcategoryid,rcomputeduration=indexitem['TaskId'],indexitem['JobId'],indexitem['CategoryId'],indexitem['ComputeDuration']
        cpu,mem,preraretime=get_cpumempretime(rcategoryid,self.categorytable)
        _ , rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration=getinfoBytaskid(self.tasktable,rtaskid)
        #update arrtask index
        taskindex+=1
        self.taskindex=taskindex
        re_list=[rtaskid,rjobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration,cpu,mem,preraretime]
        return re_list
    #这将是根据job的arrtime填充至task info 
    def getalltasktime(self,tasktable):
        alltasktime=[]
        jobid=0
        for j in self.jobarrlist:#j= arrtime
            for index, row in tasktable.iterrows():
                if row['JobId']==jobid:
                    #taskid=row['TaskId']
                    alltasktime.append(j)
                    '''  _ , rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration=getinfoBytaskid(self.tasktable,taskid)
                    cpu,mem,preraretime=get_cpumempretime(rcategoryid,self.categorytable)
                    alltask.append([taskid,jobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration,cpu,mem,preraretime])
                    ''' 
            jobid+=1
        self.alltasktime=alltasktime
        return alltasktime

    #although it's useful in theory.unuseful in implement
    def getsByjobid(self,jobid):
        outlist=[]
        for r in self.alltask:
            if r[1]==jobid:
                outlist.append(r)
        return outlist
        

    def to_input(self,outlist): #输出系列值作为nn状态输入，需要从list转为nparry
        inputs_arr = np.array(outlist)   #task single list#
        inputs_arr = inputs_arr.reshape(1, 1,11)
        #print(inputs_arr.shape)应该是这个（1,1,11)
        return inputs_arr
    #job\task 这里 
    '''
    def gettime(self,Nowtime,afterv):
        prev_time=self.Nowtime
        job_time=self.jobarrlist[self.Nowjobid]
        self.Nowjobid+=1
        return prev_time,job_time'''

#实际上，这就是状态（服务状态）类，有关方法均在此类中定义，Env则调用，如在step（）中
class ExecutorInit:
    def __init__(self,bsmetric,bstable,cloudtable, hosttable,uetable):
        ''' self.exeid ,self.categoryid,self.whenfree,self.Cloud,self.Bs,self.deviceid,self.comspeed,self.rate,self.acpu,self.amem,self.online=\
            exeid,       categoryid,     whenfree,     Cloud,     Bs,     deviceid,     comspeed,     rate,     acpu,     amem,     online
            '''
        #trans multiple csv data -> init Executor info as input state
        exes=[]
        self.hosttable=hosttable
        self.uetable=uetable
        self.bstable=bstable
        self.lasttime=0
        self.nowtime=0
        self.add_exeid=0 
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
    """usage,duration i,e.nowtime-lasttime\  depend on your step 
    instance with 'c' and init,then:
    c.updateUe(599,601,uetable,uemetric)
    c.updateUe(601,1111,uetable,uemetric)
    c.updateUe(1111,1159,uetable,uemetric)
    """
    def updateUe(self,lasttime,nowtime,uetable,uemetric):
        self.uetable=uetable #在更新ue时指定uetable并没问题
        tmp_offid=[]
        Online=uetable[(uetable['OnlineTime']<=nowtime)  & (uetable['OnlineTime']>lasttime) &  (uetable['OfflineTime']>nowtime)]
        for _ , row in Online.iterrows():
            #print(row['UEId'],row['ComputeFactor'],row['OnlineTime'],row['OfflineTime'])
            tmp_a=uemetric.loc[(uemetric['UEId']==row['UEId']) & (uemetric['Time']==nowtime),].iloc[0]
            #print(a['UEId'],a['Time'],a['Rate'],a['BSId'],a['CPU'],a['Memory'])
            deviceid,comspeed,rate,acpu,amem,online=row['UEId'],row['ComputeFactor'],tmp_a['Rate'],tmp_a['CPU'],tmp_a['Memory'],1
            self.exes.append([-1,-1,-1,0,0,deviceid,comspeed,rate,acpu,amem,online])
        #
        Offline=uetable[(uetable['OfflineTime']<=nowtime)  & (uetable['OfflineTime']>lasttime)]
        for _ , row in Offline.iterrows():
            tmp_offid.append(row['UEId'])
        self.exes = list(filter(lambda x: (x[3] != 0) | (x[4]!=0)|(x[5] not in tmp_offid), self.exes))
        #TODO\incomplete ：cloud-host和bs在600s处的更新，list筛选+赋值
        
    def updataNowtime(self,aftersec):
        #可以改的步长，作为nn输入的话还是要以task/job为间隔
        self.lasttime=self.nowtime
        self.nowtime=self.nowtime+aftersec

    def to_input(self): #输出系列值作为nn状态输入，需要从list转为nparry
        inputs_arr = np.array(self.exes)
        inputs_arr = inputs_arr.reshape(inputs_arr.shape[0], 1, inputs_arr.shape[1])
        #print(inputs_arr.shape)应该是这个（n,1,11)
        return inputs_arr
    '''
    map={'exeid':0,'categoryid':1,'whenfree':2,'Cloud':3,'Bs':4,'deviceid':5,'comspeed':6,'rate':7,'acpu':8,'amem':9,'online':10}    
    usage:
    '''
    def addexe(self,devicetype1,devicetype2,deviceid,categoryid,categorytable):   #执行Add动作部署任务时
        exes=self.exes
        #get category's correspond info（cpu mem preparetime）
        ccpu,cmem,preparetime  = get_cpumempretime(categoryid,categorytable)
        add_exeid=self.add_exeid
        self.add_exeid+=1
        add_reward=0
        for row in exes:
            if (row[3]==devicetype1)  and  (row[4]==devicetype2)  and  (row[5]==deviceid)  and  (row[0]==-1) and (row[8]>=ccpu) and (row[9]>=cmem):
                row[0],row[1],row[2],row[8],row[9]=add_exeid,categoryid,preparetime,row[8]-ccpu,row[9]-cmem #3 init value is -1
                add_reward=1
                #print('change',row)
                break
            elif (row[3]==devicetype1)  and  (row[4]==devicetype2)  and  (row[5]==deviceid)  and  (row[0]!=-1) and (row[8]>=ccpu) and (row[9]>=cmem):
                exes.append([add_exeid,categoryid,preparetime,devicetype1,devicetype2,deviceid,row[6],row[7],row[8],row[9],1])
                #print('append',row) #free after preparetime ,so 
                for r2 in exes:
                    if (r2[3]==devicetype1)  and  (r2[4]==devicetype2)  and  (r2[5]==deviceid):
                        r2[8],r2[9]=r2[8]-ccpu,r2[9]-cmem
                break 
            elif (row[3]==devicetype1)  and  (row[4]==devicetype2)  and  (row[5]==deviceid)  and  (row[0]!=-1) and (row[8]<=ccpu) and (row[9]<=cmem):
                self.delexe(self,0,row[0],self.categorytable)
            #else:
                #print('placement false')#In fact, else is unnecessary
                #add_reward=0           
        self.exes=exes
        return add_reward,add_exeid #用于区分这个add操作是否是有效动作
    #Del,exeid,,,,#18,Delete,0,,,
    def delexe(self,timestep,exeid,categorytable):   #执行Del删除某一任务时,#recording:timestep means next ?(s) to do.uesful?
        del_reward=0 
        exes=self.exes
        for row in exes:
            if (row[0]==exeid):
                devtype1,devtype2,devid,categoryid=row[3],row[4],row[5],row[1] #get dectype  and  id to update others res info in same dev
                ccpu,cmem,_ = get_cpumempretime(categoryid,categorytable)
                for r2 in exes:
                    if (r2[3]==devtype1) and (r2[4]==devtype2) and (r2[5]==devid):
                        r2[8],r2[9]=r2[8]+ccpu,r2[9]+cmem
                exes.remove(row)
                #print('remove'.exeid,row[1])
                del_reward=1    #success del_reward(meaningful action),fail with init 0.
        self.exes=exes
        return del_reward,devtype1,devtype2,devid#不会平白无故的删掉exe，因此返回devtype1,devtype2,devid用于add
    #Time,ExecutorId,TaskId
    #6,11,11 <-action like this
    #schedultime is next schedultime（s) from nowtime
    def scheduletask(self,schedultime,exeid,taskid,tasktable): #调度具体任务时
        schedul_reward=0
        walltime=900
        devspeed=0
        #nowtime=0
        categoryid,ComputeDuration=get_cateforyinfo(taskid,tasktable)
        exes=self.exes
        for row in exes:
            if (row[0]==exeid) and (row[1]==categoryid) : #先判断设备类型->设备计算速度->持续时间
                devid=row[5]
                if row[3]==1:
                    devspeed=self.get_ComputeFactorHost(devid) 
                    #print('host speed:',devspeed)
                elif row[4]==1 :
                    devspeed=self.get_ComputeFactorBs(devid)
                    #print('bs speed:',devspeed)
                else:
                    devspeed=self.get_get_ComputeFactorUe(devid)
                    #print('ue speed:',devspeed)
                walltime=devspeed*ComputeDuration
                row[2]=schedultime+walltime
                schedul_reward=1
                break
        self.exes=exes
        return schedul_reward,ComputeDuration
    def get_whenfree(self,exeid):
        when_free=-1
        exes=self.exes
        for row in exes:
            if row[0]==exeid:
                when_free=row[2]
        return int(when_free)

    #随时间更新，这里主要实现when free的更新逻辑
    #timestep 意味多久时间过去，
    def updatewithtime(self,timestep):
        #这里只是更新whenfree
        exes=self.exes
        for row in exes:
            if row[2]<=timestep:
                row[2]=0
            else:
                row[2]=row[2]-timestep
        self.exes=exes
    def get_ComputeFactorHost(self,devid):
        return self.hosttable.loc[self.hosttable['HostId']==devid,'ComputeFactor'].values[0]
    def get_ComputeFactorBs(self,devid):
        return self.bstable.loc[self.bstable['BSId']==devid,'ComputeFactor'].values[0]
    def get_get_ComputeFactorUe(self,devid):
        return self.uetable.loc[self.uetable['UEId']==devid,'ComputeFactor'].values[0]

"""def get_ComputeFactorHost(devid,hosttable):
    return hosttable.loc[hosttable['HostId']==devid,'ComputeFactor'].values[0]
def get_ComputeFactorBs(devid,bstable):
    return bstable.loc[bstable['BSId']==devid,'ComputeFactor'].values[0]
def get_get_ComputeFactorUe(devid,uetable):
    return uetable.loc[uetable['UEId']==devid,'ComputeFactor'].values[0]"""

def get_cateforyinfo(taskid,tasktable):
    return tasktable.loc[tasktable['TaskId']==taskid,'CategoryId'].values[0],tasktable.loc[tasktable['TaskId']==taskid,'ComputeDuration'].values[0]

def get_cpumempretime(categoryid,categorytable):
    _,cpu,mem,preraretime=categorytable.loc[categorytable['CategoryId']==categoryid,:].values[0]
    return cpu,mem,int(preraretime) 

#
#return  rjobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration
def getinfoBytaskid(tasktable,taskid):
    task_dependencies = {}
    for index, row in tasktable.iterrows():
        child_task_id,child_task_data,parent_task_id,parent_task_data=-1,0,-1,0
        child_tasks = eval(row['ChildTasks'])  # 将字符串转换为列表
        #print(child_tasks)
        parent_tasks = eval(row['ParentTasks'])  # 将字符串转换为列表
        #print(parent_tasks)
        for child_task in child_tasks:
            #child_task[0]为taskid，child_task[1]为数据量 #NOTE:这里只存储一个，并不完全
            child_task_id = child_task[0]
            child_task_data=child_task[1]
            #print(f'child:{child_task_id}')
        for parent_task in parent_tasks:
                parent_task_id = parent_task[0]
                parent_task_data=parent_task[1]
                #print(f'parent:{parent_task_id}')
        
        #index = taskid
        jobid,categoryid,computeduration=row['JobId'],row['CategoryId'],row['ComputeDuration']
        task_dependencies[index]=[jobid,categoryid,child_task_id,child_task_data,parent_task_id,parent_task_data,computeduration]
    rjobid,rcategoryid,rchild_task_id,rchild_task_data,rparent_task_id,rparent_task_data,rcomputeduration=task_dependencies[taskid]
    return rjobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration

        
'''
@param:datapath = 前缀路径,直到test0之前
@param:ff 多个test路径,这里分开写是为了方便后续加入迭代
@return dict
example:
load=loadalldata('./dataset/testset/','test0')
bsmetric=load['bsmetric']
TODO:待优化，直接多返回值 DONE√
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
    return bsmetric,bstable,cloudtable,hosttable,jobtable,tasktable,categorytable,uetable,uemetric
   #return {'bsmetric':bsmetric,'bstable':bstable,
   #         'cloudtable':cloudtable,'hosttable':hosttable,
   #         'jobtable':jobtable,'tasktable':tasktable,
   #         'categorytable':categorytable,'uetable':uetable,'uemetric':uemetric}

'''
这里我想干嘛来着？？？？
根据output输出 到csv文件里面 可以先不用
'''
def formatS(S1,S2):
    pass
    return S1,S2

def transA1(A1):
    pass
    return A1

def exeA1():
    pass

def get_devtype(devtype1,devtype2):
    if devtype1==1  and devtype2==0:
        devtype='Cloud'
    elif devtype1==0  and  devtype2==1:
        devtype='BS'
    else:
        devtype='UE'
    return devtype
#TODO:  2023年6月5日,完成Task Env的全部方法
# Define the environment class，the Env get all data from the responding floder and run with state update  and  job/task arr online.
#agent will output the joint action (placement/development and schedule)
class Env:
    def __init__(self,rootfile):
        #iter to next epsfiles, how to use
        #load data from rootfile/epsfile/xxx.csv
        self.rootfile=rootfile
        self.place_reward=0
        self.scheduler_reward=0
        self.comdelay=0
        self.newexeid=0
        '''self.epsfilenum=0
        self.epsfile='train'+str(self.epsfilenum)
        load=loadalldata(rootfile,self.epsfile)
        #get all csv file in trainsetxx
        bsmetric,bstable,cloudtable, hosttable, jobtable,tasktable,categorytable,uetable,uemetric =load
        self.bsmetric,self.bstable,self.cloudtable, self.hosttable, self.jobtable,self.tasktable,self.categorytable,self.uetable,self.uemetric =\
            bsmetric,bstable,cloudtable, hosttable, jobtable,tasktable,categorytable,uetable,uemetric
        #Now we get all datafile 
        self.ExeState=ExecutorInit(bsmetric,bstable,cloudtable, hosttable)
        #get jot/task instance
        self.jobtask=JobTask(jobtable,tasktable,categorytable)
        self.timesteplist=self.jobtask.jobarrlist
        self.timesteplistindex=0    #jobid 
        self.nowtime=0
        self.lasttime=0 '''
        
    """ Executor().__init__(
        #并不需要reset 每次直接读取到下一个trainset就好了   #   这句话是不对的！
        Time,Action,ExecutorId,CategoryId,DeviceType,DeviceId \\taskid 并不需要 
        after(x),+-1 0, x   ,   x,          x,          x
        1 means add, -1 means del, 0 means empty
    """
    def reset(self,epsfilenum):
        self.epsfile=epsfilenum
        print('reset to floder',epsfilenum)
        load=loadalldata(self.rootfile,self.epsfile)
        #get all csv file in trainsetxx
        bsmetric,bstable,cloudtable, hosttable, jobtable,tasktable,categorytable,uetable,uemetric =load
        self.bsmetric,self.bstable,self.cloudtable, self.hosttable, self.jobtable,self.tasktable,self.categorytable,self.uetable,self.uemetric =\
            bsmetric,bstable,cloudtable, hosttable, jobtable,tasktable,categorytable,uetable,uemetric
        print('load files success')
        #Now we get all datafile 
        self.ExeState=ExecutorInit(bsmetric,bstable,cloudtable, hosttable,uetable)
        #get jot/task instance
        self.jobtask=JobTask(jobtable,tasktable,categorytable)
        self.alltasktime=self.jobtask.alltasktime   #alltasktime [25,25,25,25,25,30,30,31……] 暗涵 taskid 
        self.timesteplist=self.jobtask.jobarrlist   #jobarrlist [25,30,31……]    暗涵 jobid
        self.timesteplistindex=0    #jobid X #now task id
        self.nowtime=0
        self.lasttime=0
        #get init state, iow first state


    def step(self, action,taskid,length):
        self.nowtime=self.alltasktime[self.timesteplistindex]
        timestepvalue=self.nowtime-self.lasttime
        reward=0
        if timestepvalue!=0:
            self.ExeState.updateUe(self.lasttime,self.nowtime,self.uetable,self.uemetric)
            self.ExeState.updatewithtime(timestepvalue)
        _,categoryid,_,_,_,_,_=getinfoBytaskid(self.tasktable,taskid)
        
        #对于之前已接收计算任务的节点，更新when free \\ exeid,devtype1,devtype2,devid
        exeid,devtype1,devtype2,devid,compute_speed=action
        ###
        #x#
        ###
        #start implement action
        if exeid==-1 :
            _,_,tpreraretime = get_cpumempretime(categoryid,self.categorytable)
            _,self.newexeid=self.ExeState.addexe(devtype1,devtype2,devid,categoryid,self.categorytable)
            self.scheduler_reward,self.comdelay=self.ExeState.scheduletask(tpreraretime,exeid,taskid,self.tasktable)
            reward=self.comdelay*compute_speed+tpreraretime
        #放置动作ADD or DEL 返回放置成功与否的0\1
        else :
            ppreparetime=self.ExeState.get_whenfree(exeid)
            self.scheduler_reward,self.comdelay=self.ExeState.scheduletask(ppreparetime,exeid,taskid,self.tasktable)
            reward=self.comdelay*compute_speed+ppreparetime
        #this implement in code of train
        #end action #TODO:consider the data trans delay,implement later
        if self.timesteplistindex==length:
            done=True
        else:
            done = False 
        self.lasttime=self.nowtime  #更新时间
        self.timesteplistindex+=1
        #arrtime=self.timesteplist[self.timesteplistindex]
        next_state1=self.ExeState.to_input() #get all ava param
        next_state2=self.jobtask.to_input(self.jobtask.getsingletask(self.tasktable)) 
        
        shape1=next_state1.shape[0]
        next_state2=tf.tile(next_state2, [shape1, 1,1])#State shape OK
        next_state = tf.concat([next_state1, next_state2], axis=-1)
        return next_state, (1100-reward), done, {}
    def getreward(self):
        #return self.place_reward*10+self.scheduler_reward*10+(1000-self.comdelay)
        return (1000-self.comdelay)
        #TODO:由于ue下线资源变化导致的任务失败，可以在ExecutorInit更新资源信息时判断whenfree是否为闲
        #NOTE simple complete task/job rate and delay
        #CHANGE:placement and shceduler is important,too.

#end Custom Env

#NO ddpg ,AC now
num_inputs = 4
num_actions = 1 # NOTE 这里动作维度一减再减，说明前期的构思还是不成熟的 P
num_hidden = 512
'''#输出的联合动作
Action Exeid  devtype1 devtype2   devid     afterHowlongsec   
-1~1    many    #get from demxxxinput#          continue
输出action '''
inputs = layers.Input(shape=(1,22))
#inputs2 = layers.Input(shape=(1,11))

# Define the hidden layers
#x = layers.Dense(256, activation='relu')(inputs)
x = tf.keras.layers.Dense(256, activation='relu')(inputs)
#x = tf.keras.layers.Dense(256, activation='relu')(x)

# Define the hidden layers
"""y = tf.keras.layers.Dense(256, activation='relu')(inputs2)
y = tf.keras.layers.Dense(256, activation='relu')(y)

xy = tf.keras.layers.Concatenate()([x, y])
xy = tf.keras.layers.Dense(256, activation='relu')(xy)"""
common = layers.Dense(num_hidden, activation="relu")(x)
action = layers.Dense(num_actions, activation="relu")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
huber_loss = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss=huber_loss, optimizer=optimizer)

#model.compile(loss='mean_squared_error', optimizer='sgd')
#optimizer = keras.optimizers.Adam(learning_rate=0.01)
#huber_loss = keras.losses.Huber()

action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0
# Define the hyperparameters
batch_size = 64
gamma = 0.99
tau = 0.001
#buffer_size = 1000000
#exploration_noise = 0.1    \\ unused
#exploration_noise_decay = 0.99
#exploration_noise_min = 0.01
#max_episodes = 50 #50 train set \\ unused
#every step is time in 1(s) no
#max_steps = 10 #should be 1200(s) or task arr step

# Create the environment 'dataset'
env = Env('./dataset/trainset/')
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
#after,place_action,exeid,devtype1,devtype2,devid=action NOTE:action format
# Train the agent
episode_rewards = []
#./dataset/testset/','test0
train_set_f=[]
for i in range (0,2):
   train_set_f.append('train'+str(i))
#get train0~train50
actionlist=[]

for episode in train_set_f:
    env.reset(episode) #init with trainset0
    episode_reward = 0
    lasttime=0
    #exploration_noise *= exploration_noise_decay
    #exploration_noise = max(exploration_noise, exploration_noise_min)
    steps=env.alltasktime #steps =[35,35,35,35,35……] 以task到达为步
    #print(f'step:{steps}')
    #init\first s1 s2
    state1=env.ExeState.to_input()
    shape1=state1.shape[0]
    state2=env.jobtask.to_input(env.jobtask.getsingletask(env.tasktable))
    state2=tf.tile(state2, [shape1, 1,1])
    state= tf.concat([state1, state2], axis=-1)
    taskid=0 
    #self.jobtask.to_input(self.jobtask.getsingletask(self.tasktable))
    with tf.GradientTape() as tape:
        for step in steps:
            nowtime=step #(s)now
            outs, critic_value = model(state)
            outs=abs(np.squeeze(outs))
            normolize_outs=outs/np.sum(outs)
            #print('n__',normolize_outs) #↓这个也是维度，一样的
            sample_dim=np.random.choice(len(normolize_outs), p=normolize_outs)
            #概率输出我要选择的维度（2023年6月10日
            action_probs_history.append(tf.math.log(normolize_outs[sample_dim]))
            critic_value_history.append(critic_value[sample_dim,0])
            #切片
            #get corresponding dem from state1
            #state1[sample_dim,0,:]
            #'exeid':0,'categoryid':1,'whenfree':2,'Cloud':3,'Bs':4,'deviceid':5,'comspeed':6,'rate':7,'acpu':8,'amem':9,'online':10
            exeid,_,when_free,devtype1,devtype2,devid,compute_speed,_,_,_,_,_,_,_,_,_,_,_,_,_,_,prepare_time=state[sample_dim, 0, :]    #获取操作的devid信息
            #re_list=[rtaskid,rjobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration,cpu,mem,preraretime]
            #_,_,_,_,_,_,_,_,_,_,prepare_time=state[sample_dim, 0, :] #不太需要
            # Take a step in the environment \\ after,place_action,exeid,devtype1,devtype2,devid=action
            formate_action=(exeid,devtype1,devtype2,devid,int(compute_speed))
            #print(formate_action)
            # print(f'now:{nowtime}to sched taskid:{taskid}at time{nowtime+place_after}')
            #now get step
            next_state, reward, done,_= env.step(formate_action,taskid,len(steps)-2)
            #print(f'task{taskid}is scheaduled to {exeid} and reward:{reward}')
            rewards_history.append(reward)
            # Update the state and episode reward
            state=next_state
            episode_reward += reward
            #shape1=state1.shape[0]
            #state2=tf.tile(state2, [shape1, 1,1])#State shape OK
            taskid+=1 #
            #actionlist.append(action)
            if done:  # Check if the episode is done
                break
        # Append the episode reward to the list of episode rewards
        running_reward = 0.1*episode_reward + (1 - 0.1)*running_reward
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        history = zip(action_probs_history, critic_value_history, returns)

        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob* ret)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    episode_rewards.append(episode_reward)
    # Print the episode reward
    print('Episode:', episode, 'Reward:', episode_reward)
print(episode_rewards)

model.save("model2.keras")


'''
test set ->>>>>>>
# Plot the episode rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
'''

