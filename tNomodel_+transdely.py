import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import csv
import random
import math


# 这里的task/job 只适用于不断更新至下一条jod到达
class JobTask:
    def __init__(self, jobtable, tasktable, categorytable):
        self.Nowtime = 0
        self.Nowjobid = 0
        self.jobarrlist = []
        self.arrtask = []
        self.alltask = []
        self.alltasktime = []
        self.taskindex = 0
        self.tasktable, self.categorytable = tasktable, categorytable
        for _,  row in jobtable.iterrows():
            jobid = row['JobId']
            # Now we get arr list with jobid  and  time \\不需要了
            self.jobarrlist.append(row['ArriveTime'])
            for _, row2 in tasktable.iterrows():
                if row2['JobId'] == jobid:
                    self.alltasktime.append(row['ArriveTime'])

            # 现在我们有了一个以jodid为index，access by 'jobarrlist[jobid]=arrtime

    # time taskid jobid categoryid parentid pardata childid childata computeduration  rcpu rmem preraretime \\dim=11
    # now     N     x        x        x       x       x         x       x       x     x
    # here, return single task info show above . be called at state2 output
    def getsingletask(self, tasktable):
        re_list = []
        taskindex = self.taskindex
        indexitem = tasktable.iloc[taskindex]
        rtaskid, rjobid, rcategoryid, rcomputeduration = indexitem['TaskId'], indexitem[
            'JobId'], indexitem['CategoryId'], indexitem['ComputeDuration']
        cpu, mem, preraretime = get_cpumempretime(
            rcategoryid, self.categorytable)
        _, _, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration = getinfoBytaskid(
            self.tasktable, rtaskid)
        # update arrtask index
        taskindex += 1
        self.taskindex = taskindex
        re_list = [rtaskid, rjobid, rcategoryid, rparent_task_id, rparent_task_data,
                   rchild_task_id, rchild_task_data, rcomputeduration, cpu, mem, preraretime]
        return re_list
    # 这将是根据job的arrtime填充至task info

    def getalltasktime(self, tasktable):
        alltasktime = []
        jobid = 0
        for j in self.jobarrlist:  # j= arrtime
            for index, row in tasktable.iterrows():
                if row['JobId'] == jobid:
                    # taskid=row['TaskId']
                    alltasktime.append(j)
                    '''  _ , rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration=getinfoBytaskid(self.tasktable,taskid)
                    cpu,mem,preraretime=get_cpumempretime(rcategoryid,self.categorytable)
                    alltask.append([taskid,jobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration,cpu,mem,preraretime])
                    '''
            jobid += 1
        self.alltasktime = alltasktime
        return alltasktime

    # although it's useful in theory.unuseful in implement
    def getsByjobid(self, jobid):
        outlist = []
        for r in self.alltask:
            if r[1] == jobid:
                outlist.append(r)
        return outlist

    def to_input(self, outlist):  # 输出系列值作为nn状态输入，需要从list转为nparry
        inputs_arr = np.array(outlist)  # task single list#
        inputs_arr = inputs_arr.reshape(1, 1, 11)
        # print(inputs_arr.shape)应该是这个（1,1,11)
        return inputs_arr
    # job\task 这里
    '''
    def gettime(self,Nowtime,afterv):
        prev_time=self.Nowtime
        job_time=self.jobarrlist[self.Nowjobid]
        self.Nowjobid+=1
        return prev_time,job_time'''


# 实际上，这就是状态（服务状态）类，有关方法均在此类中定义，Env则调用，如在step（）中
class ExecutorInit:
    def __init__(self, bsmetric, bstable, cloudtable, hosttable, uetable):
        ''' self.exeid ,self.categoryid,self.whenfree,self.Cloud,self.Bs,self.deviceid,self.comspeed,self.rate,self.acpu,self.amem,self.online=\
            exeid,       categoryid,     whenfree,     Cloud,     Bs,     deviceid,     comspeed,     rate,     acpu,     amem,     online
            '''
        # trans multiple csv data -> init Executor info as input state
        exes = []
        self.hosttable = hosttable
        self.uetable = uetable
        self.bstable = bstable
        self.lasttime = 0
        self.nowtime = 0
        self.add_exeid = 0
        # 先添加cloud上的host，再bs，在online的ue
        hostid, _, cpu, mem, speed = hosttable[[
            'HostId', 'CloudId', 'CPU', 'Memory', 'ComputeFactor']].values[0]
        exeid = categoryid = whenfree = -1
        Cloud, Bs, online = 1, 0, 1
        deviceid = hostid
        comspeed = speed
        acpu, amem = cpu, mem
        rate = cloudtable['Rate'].values[0]
        exes.append([exeid, categoryid, whenfree, Cloud, Bs,
                    deviceid, comspeed, rate, acpu, amem, online])
        # end Cloud host init/start bs init
        initbs = bsmetric.loc[bsmetric['Time'] == 0, :]
        for _, row in initbs.iterrows():
            Cloud, Bs, online = 0, 1, 1
            exeid = categoryid = whenfree = -1
            deviceid, _, acpu, amem = row['BSId'], row['Time'], row['CPU'], row['Memory']
            _, rate, comspeed = bstable.loc[bstable['BSId']
                                            == deviceid, :].values[0]
            exes.append([exeid, categoryid, whenfree, Cloud, Bs,
                        deviceid, comspeed, rate, acpu, amem, online])
        # end bs init/All ue offline
        # c=ExecutorInit(bsmetric,bstable,cloudtable, hosttable,uetable,uemetric) initexe=c.exes to get result as init input #
        self.exes = exes

    # NOTE：每时间步长都应该调用 OR task/job到达间隔作为步长???，bs和cloud应该在600s调用
    """usage,duration i,e.nowtime-lasttime\  depend on your step 
    instance with 'c' and init,then:
    c.updateUe(599,601,uetable,uemetric)
    c.updateUe(601,1111,uetable,uemetric)
    c.updateUe(1111,1159,uetable,uemetric)
    """

    def updateUe(self, lasttime, nowtime, uetable, uemetric):
        self.uetable = uetable  # 在更新ue时指定uetable并没问题
        tmp_offid = []
        Online = uetable[(uetable['OnlineTime'] <= nowtime) & (
            uetable['OnlineTime'] > lasttime) & (uetable['OfflineTime'] > nowtime)]
        for _, row in Online.iterrows():
            # print(row['UEId'],row['ComputeFactor'],row['OnlineTime'],row['OfflineTime'])
            tmp_a = uemetric.loc[(uemetric['UEId'] == row['UEId']) & (
                uemetric['Time'] == nowtime),].iloc[0]
            # print(a['UEId'],a['Time'],a['Rate'],a['BSId'],a['CPU'],a['Memory'])
            deviceid, comspeed, rate, acpu, amem, online = row['UEId'], row[
                'ComputeFactor'], tmp_a['Rate'], tmp_a['CPU'], tmp_a['Memory'], 1
            self.exes.append([-1, -1, -1, 0, 0, deviceid,
                             comspeed, rate, acpu, amem, online])
        #
        Offline = uetable[(uetable['OfflineTime'] <= nowtime)
                          & (uetable['OfflineTime'] > lasttime)]
        for _, row in Offline.iterrows():
            tmp_offid.append(row['UEId'])
        self.exes = list(filter(lambda x: (x[3] != 0) | (
            x[4] != 0) | (x[5] not in tmp_offid), self.exes))
        # TODO\incomplete ：cloud-host和bs在600s处的更新，list筛选+赋值

    def updataNowtime(self, aftersec):
        # 可以改的步长，作为nn输入的话还是要以task/job为间隔
        self.lasttime = self.nowtime
        self.nowtime = self.nowtime+aftersec

    def to_input(self):  # 输出系列值作为nn状态输入，需要从list转为nparry
        inputs_arr = np.array(self.exes)
        inputs_arr = inputs_arr.reshape(inputs_arr.shape[0], 1, 11)
        # print(inputs_arr.shape)应该是这个（n,1,11)
        return inputs_arr
    '''
    map={'exeid':0,'categoryid':1,'whenfree':2,'Cloud':3,'Bs':4,'deviceid':5,'comspeed':6,'rate':7,'acpu':8,'amem':9,'online':10}    
    usage:
    Time,Action,ExecutorId,CategoryId,DeviceType,DeviceId
    4,Add,0,20,Cloud,0
    '''
    #这里根据taskid-categoryid-预计最快完成时间的exeid

    def IsResok(self, devicetype1, devicetype2, deviceid, categoryid, categorytable):
        ccpu, cmem, preparetime = get_cpumempretime(categoryid, categorytable)
        state = True
        for row in self.exes:
            if (row[3] == devicetype1) and (row[4] == devicetype2) and (row[5] == deviceid) and (row[8] >= ccpu) and (row[9] >= cmem):
                # 节点资源充足
                state = True
                break
            if (row[3] == devicetype1) and (row[4] == devicetype2) and (row[5] == deviceid) and ((row[8] < ccpu) or (row[9] < cmem)):
                state = False
                break
        return state

    # 资源充足到达此处  #执行Add动作部署任务时
    def addexe(self, devicetype1, devicetype2, deviceid, categoryid, categorytable):
        out_action_list = []
        deviceid=int(deviceid)
        # exes=self.exes
        # get category's correspond info（cpu mem preparetime）
        ccpu, cmem, preparetime = get_cpumempretime(categoryid, categorytable)
        add_exeid = self.add_exeid
        self.add_exeid += 1
        add_reward = 0
        for row in self.exes:
            # 这里的逻辑需要注意，不建议修改exeid为-1的row内容，因为这可能会导致该dev的所有信息被删除
            """if (row[3]==devicetype1)  and  (row[4]==devicetype2)  and  (row[5]==deviceid)  and  (row[0]==-1) and (row[8]>=ccpu) and (row[9]>=cmem):
                row[0],row[1],row[2],row[8],row[9]=add_exeid,categoryid,preparetime,row[8]-ccpu,row[9]-cmem #3 init value is -1
                out_action_list.append(['Add',add_exeid,categoryid,get_devtype(devicetype1,devicetype2),int(deviceid)])
                add_reward=1
                #print('change',row)
                break  """
            # if (row[3]==devicetype1)  and  (row[4]==devicetype2)  and  (row[5]==deviceid)  and  (row[0]!=-1) and (row[8]>=ccpu) and (row[9]>=cmem):

            if (row[3] == devicetype1) and (row[4] == devicetype2) and (row[5] == deviceid):
                self.exes.append([add_exeid, categoryid, preparetime, devicetype1,
                                 devicetype2, deviceid, row[6], row[7], row[8], row[9], 1])
                out_action_list.append(['Add', add_exeid, categoryid, get_devtype(
                    devicetype1, devicetype2), deviceid])
                # print('append',row) #free after preparetime ,so
                for r2 in self.exes:
                    if (r2[3] == devicetype1) and (r2[4] == devicetype2) and (r2[5] == deviceid):
                        r2[8], r2[9] = r2[8]-ccpu, r2[9]-cmem
                #print(f'DeVid{deviceid} has CPU:{r2[8]}left MEM:{r2[9]} Left')
                break
        return add_reward, add_exeid, out_action_list  # 用于区分这个add操作是否是有效动作
    # Del,exeid,,,,#18,Delete,0,,,

    # 执行Del删除某一任务时,#recording:timestep means next ?(s) to do.uesful?
    def delexe(self,exeid, categorytable):
        out_action_list = []
        del_reward = 0
        for row in self.exes:
            if (row[0] == exeid) :
                # get dectype  and  id to update others res info in same dev
                devtype1, devtype2, devid, categoryid = row[3], row[4], row[5], row[1]
                ccpu, cmem, _ = get_cpumempretime(categoryid, categorytable)
                for r2 in self.exes:
                    if (r2[3] == devtype1) and (r2[4] == devtype2) and (r2[5] == devid) :
                        r2[8], r2[9] = r2[8]+ccpu, r2[9]+cmem
                out_action_list.append([row[2],'Delete', row[0], '', '', ''])
                self.exes.remove(row)
                # print('remove'.exeid,row[1])
                # success del_reward(meaningful action),fail with init 0.
                del_reward = 1
        # 不会平白无故的删掉exe，因此返回devtype1,devtype2,devid用于add（弃用该观点
        return del_reward, devtype1, devtype2, devid, out_action_list
    # Time,ExecutorId,TaskId
    # 6,11,11 <-action like this
    # schedultime is next schedultime（s) from nowtime
    # Time,ExecutorId,TaskId
    # 6,11,11
    def scheduletask(self, schedultime, exeid, taskid, tasktable,transdelay):  # 调度具体任务时
        # out_action_list=[]
        schedul_reward = 0
        #transdelay=20   #
        walltime = 900
        devspeed = 0
        # nowtime=0
        categoryid, ComputeDuration = get_cateforyinfo(taskid, tasktable)
        exes = self.exes
        for row in exes:
            if (row[0] == exeid) and (row[1] == categoryid):  # 先判断设备类型->设备计算速度->持续时间
                devid = row[5]
                if (row[3] == 1):
                    devspeed = self.get_ComputeFactorHost(devid)
                    #print(f'Task{taskid}->CLOUD COM-Factor:{devspeed}')
                    #print('host speed:',devspeed)
                elif (row[4] == 1):
                    devspeed = self.get_ComputeFactorBs(devid)
                    #print(f'Task{taskid}->BS COM-Factor:{devspeed}')
                    #print('bs speed:',devspeed)
                else:
                    devspeed = self.get_get_ComputeFactorUe(devid)
                    #print(f'Task{taskid}->UE COM-Factor:{devspeed}')
                    #print('ue speed:',devspeed)
                walltime = devspeed*ComputeDuration
                row[2] = schedultime+walltime+transdelay
                #这里应该再加上传输时延就对了
                #print(f'Task{taskid}->COM-Factor:{devspeed}')
                #print(f'{walltime}(s) to cpmpute||END at{row[2]}')
                schedul_reward = 1
                break
        self.exes = exes
        return schedul_reward, ComputeDuration

    def get_whenfree(self, exeid):
        when_free = -1
        exes = self.exes
        for row in exes:
            if row[0] == exeid:
                when_free = row[2]
        return int(when_free)

    # 随时间更新，这里主要实现when free的更新逻辑
    # timestep 意味多久时间过去，
    def updatewithtime(self, timestep):
        # 这里只是更新whenfree
        exes = self.exes
        for row in exes:
            if row[2] <= timestep:
                row[2] = 0
            else:
                row[2] = row[2]-timestep
        self.exes = exes

    def get_ComputeFactorHost(self, devid):
        return self.hosttable.loc[self.hosttable['HostId'] == devid, 'ComputeFactor'].values[0]

    def get_ComputeFactorBs(self, devid):
        return self.bstable.loc[self.bstable['BSId'] == devid, 'ComputeFactor'].values[0]

    def get_get_ComputeFactorUe(self, devid):
        return self.uetable.loc[self.uetable['UEId'] == devid, 'ComputeFactor'].values[0]
    def get_ComputeFactor():
        pass
def get_cateforyinfo(taskid, tasktable):
    return tasktable.loc[tasktable['TaskId'] == taskid, 'CategoryId'].values[0], tasktable.loc[tasktable['TaskId'] == taskid, 'ComputeDuration'].values[0]

#RequestCPU,RequestMemory,PrepareDuration
def get_cpumempretime(categoryid, categorytable):
    #print(f'looking for categoryid:{categoryid}')
    _,cpu, mem, preraretime = categorytable.loc[categorytable['CategoryId'] == categoryid, :].values[0] 
    #这样写会莫名其妙说超出索引 奇怪了。。。
    #categoryid怎么会有-1值？？？
    return cpu, mem, preraretime

#
# return  rjobid,rcategoryid,rparent_task_id,rparent_task_data,rchild_task_id,rchild_task_data,rcomputeduration

#    return rjobid, rcategoryid, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration
def getinfoBytaskid(tasktable, taskid):
    task_dependencies = {}
    for index, row in tasktable.iterrows():
        child_task_id, child_task_data, parent_task_id, parent_task_data = -1, 0, -1, 0
        child_tasks = eval(row['ChildTasks'])  # 将字符串转换为列表
        # print(child_tasks)
        parent_tasks = eval(row['ParentTasks'])  # 将字符串转换为列表
        # print(parent_tasks)
        for child_task in child_tasks:
            # child_task[0]为taskid，child_task[1]为数据量 #NOTE:这里只存储一个，并不完全
            child_task_id = child_task[0]
            child_task_data = child_task[1]
            # print(f'child:{child_task_id}')
        for parent_task in parent_tasks:
            parent_task_id = parent_task[0]
            parent_task_data = parent_task[1]
            # print(f'parent:{parent_task_id}')

        #index = taskid
        jobid, categoryid, computeduration = row['JobId'], row['CategoryId'], row['ComputeDuration']
        task_dependencies[index] = [jobid, categoryid, child_task_id,
                                    child_task_data, parent_task_id, parent_task_data, computeduration]
    rjobid, rcategoryid, rchild_task_id, rchild_task_data, rparent_task_id, rparent_task_data, rcomputeduration = task_dependencies[
        taskid]
    return rjobid, rcategoryid, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration
# (0, 10, -1, 0, 1, 4, 8) result like -1 means first task without parent


def loadalldata(datapath, ff):
    # BSId,Time,CPU,Memory.small scale
    bsmetric = pd.read_csv(os.path.join(datapath, ff, 'bs_metric.csv'))
    # BSId,Time,CPU,Memory.small
    bstable = pd.read_csv(os.path.join(datapath, ff, 'bs_table.csv'))
    # CloudId,Rate.small
    cloudtable = pd.read_csv(os.path.join(datapath, ff, 'cloud_table.csv'))
    # HostId,CloudId,CPU,Memory,ComputeFactor.small
    hosttable = pd.read_csv(os.path.join(datapath, ff, 'host_table.csv'))
    # JobId,ArriveTime
    jobtable = pd.read_csv(os.path.join(datapath, ff, 'job_table.csv'))
    # TaskId,JobId,CategoryId,ParentTasks,ChildTasks,ComputeDuration
    tasktable = pd.read_csv(os.path.join(datapath, ff, 'task_table.csv'))
    # CategoryId,RequestCPU,RequestMemory,PrepareDuration
    categorytable = pd.read_csv(os.path.join(
        datapath, ff, 'category_table.csv'))
    # UEId,ComputeFactor,OnlineTime,OfflineTime
    uetable = pd.read_csv(os.path.join(datapath, ff, 'ue_table.csv'))
    # UEId,Time,Rate,BSId,CPU,Memory
    uemetric = pd.read_csv(os.path.join(datapath, ff, 'ue_metric.csv'))
    # num_ue=max(uetable['UEId'])
    return bsmetric, bstable, cloudtable, hosttable, jobtable, tasktable, categorytable, uetable, uemetric
   # return {'bsmetric':bsmetric,'bstable':bstable,
   #         'cloudtable':cloudtable,'hosttable':hosttable,
   #         'jobtable':jobtable,'tasktable':tasktable,
   #         'categorytable':categorytable,'uetable':uetable,'uemetric':uemetric}


'''
这里我想干嘛来着？？？？
根据output输出 到csv文件里面 可以先不用
'''


def initcsv(filename, value):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row to the csv file
        writer.writerow(value)
# call in cycle to generate many actions recording


def addrowcsv(filename, value: list):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row to the csv file
        writer.writerow(value)
# time:int,action:int,exeid:int,categoryid:int,devicetype:int,deviceid:int\\like encoder


def formoutput_action_executor(list1: list):
    output_list = []
    action_t = ('Add', 'Delete',)  # 0-add 1-del
    device_t = ('Cloud', 'BS', 'UE',)  # 0-cloud 1-bs 2-ue
    time, action, exeid, categoryid, devicetype, deviceid = list1[:]
    output_list = [time, action_t[action], exeid,
                   categoryid, device_t[devicetype], deviceid]
    return output_list


def get_devtype(devtype1, devtype2):
    if devtype1 == 1 and devtype2 == 0:
        devtype = 'Cloud'
    elif devtype1 == 0 and devtype2 == 1:
        devtype = 'BS'
    else:
        devtype = 'UE'
    return devtype


class Env:
    def __init__(self, rootfile):
        # iter to next epsfiles, how to use
        # load data from rootfile/epsfile/xxx.csv
        self.rootfile = rootfile
        self.place_reward = 0
        self.scheduler_reward = 0
        self.comdelay = 0
        self.newexeid = 0
        self.taskid_complete_time = []
        self.datainfo=[]#taskid -devtype devid rate

    """ Executor().__init__(
        #并不需要reset 每次直接读取到下一个trainset就好了   #   这句话是不对的！
        Time,Action,ExecutorId,CategoryId,DeviceType,DeviceId \\taskid 并不需要 
        after(x),+-1 0, x   ,   x,          x,          x
        1 means add, -1 means del, 0 means empty
    """

    def reset(self, epsfilenum):
        self.datainfo.clear()
        self.taskid_complete_time.clear()
        self.epsfile = epsfilenum
        print('reset to floder', epsfilenum)
        load = loadalldata(self.rootfile, self.epsfile)
        # get all csv file in trainsetxx
        bsmetric, bstable, cloudtable, hosttable, jobtable, tasktable, categorytable, uetable, uemetric = load
        self.bsmetric, self.bstable, self.cloudtable, self.hosttable, self.jobtable, self.tasktable, self.categorytable, self.uetable, self.uemetric =\
            bsmetric, bstable, cloudtable, hosttable, jobtable, tasktable, categorytable, uetable, uemetric
        #print('load files success')
        # Now we get all datafile
        self.ExeState = ExecutorInit(
            bsmetric, bstable, cloudtable, hosttable, uetable)
        # get jot/task instance
        self.jobtask = JobTask(jobtable, tasktable, categorytable)
        # alltasktime [25,25,25,25,25,30,30,31……] 暗涵 taskid
        self.alltasktime = self.jobtask.alltasktime
        # jobarrlist [25,30,31……]    暗涵 jobid
        self.timesteplist = self.jobtask.jobarrlist
        self.timesteplistindex = 0  # now task id
        self.nowtime = 0
        self.lasttime = 0
        # get init state,  first state

    def update_parent_complete_time(self, theta):
        t = self.taskid_complete_time
        # theta=nowtime-lasttime
        for item in t:
            if item <= theta:
                item = 0
            else:
                item = item-theta
        self.taskid_complete_time = t

    def getFast(self,taskid):
        rate=1
        acategory=0
        e1,cloud,bs,devid,speed=0,1,0,0,self.ExeState.get_ComputeFactorHost(0)
        recording_time=900
        #    return rjobid, rcategoryid, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration
        rjobid, rcategoryid, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration=getinfoBytaskid(self.tasktable, taskid)
        #RequestCPU,RequestMemory,PrepareDuration
        ccpu,cmem,cprepatetime=get_cpumempretime(rcategoryid, self.categorytable)
        #根据category匹配exes
        for row in self.ExeState.exes:
            if (row[1]==rcategoryid) and (row[1]!=-1): 
                t1=rcomputeduration*row[6]+row[2]
                if t1 < recording_time:
                    recording_time=t1
                    e1,cloud,bs,devid,speed,acategory,rate=row[0],row[3],row[4],row[5],row[6],row[1],row[7]
            elif (row[7]>=ccpu) and (row[8]>=cmem):
                t1=cprepatetime+rcomputeduration*row[6]#暂不考虑删除sheduler以及传输时延
                if t1 < recording_time:
                    recording_time=t1
                    e1,cloud,bs,devid,speed,acategory,rate=row[0],row[3],row[4],row[5],row[6],row[1],row[7]
        return recording_time,e1,cloud,bs,devid,speed,acategory,rate

    def step(self, action, taskid, length, planexe, plantask):
        datainfo=self.datainfo
        parent_complete_time = 0
        self.nowtime = self.alltasktime[self.timesteplistindex]
        timestepvalue = self.nowtime-self.lasttime
        reward = 0
        if timestepvalue != 0:
            self.ExeState.updateUe(
                self.lasttime, self.nowtime, self.uetable, self.uemetric)
            self.ExeState.updatewithtime(timestepvalue)
            self.update_parent_complete_time(timestepvalue)
            #rjobid, rcategoryid, rparent_task_id, rparent_task_data, rchild_task_id, rchild_task_data, rcomputeduration
        _, categoryid, parent_taskid, parent_task_data, _, _, _ = getinfoBytaskid(
            self.tasktable, taskid)  # 这里是task对应的category

        # 检查parent task是否完成，获取预计完成时间
        if parent_taskid == -1 or parent_taskid > len(self.taskid_complete_time):
            parent_complete_time = 0
        else:
            parent_complete_time = self.taskid_complete_time[parent_taskid]
        #print(f'taskid:{taskid}のparent task:{parent_taskid}need {parent_complete_time}(s) to complete')

        # 对于之前已接收计算任务的节点，更新when free \\ exeid,devtype1,devtype2,devid
        # acategoryid是设备现有的category
        exeid, devtype1, devtype2, devid, compute_speed, acategoryid ,childrate= action
        # start implement action

        _, _, place_preraretime = get_cpumempretime(categoryid, self.categorytable)
        tpreraretime = max(place_preraretime, parent_complete_time)
        if parent_taskid==-1 or parent_taskid > len(self.taskid_complete_time):
            datadelay=0
        else:
            parent_devtype,parent_devid,parent_rate=datainfo[parent_taskid][0],datainfo[parent_taskid][1],datainfo[parent_taskid][2]
            if (get_devtype(devtype1,devtype2)==parent_devtype) and(parent_devid==devid):
                datadelay=0
            else:
                transrate=min(parent_rate,childrate)
                datadelay=math.ceil(parent_task_data/transrate)
        if acategoryid != categoryid:
            if self.ExeState.IsResok(devtype1, devtype2, devid, categoryid, self.categorytable):
                # 资源充足
                _, self.newexeid, out_actionlist0 = self.ExeState.addexe(
                    int(devtype1), int(devtype2), devid, categoryid, self.categorytable)
                for item in out_actionlist0:
                    addrowcsv(planexe, [self.nowtime, item[0],
                              item[1], item[2], item[3], item[4]])

                self.scheduler_reward, self.comdelay = self.ExeState.scheduletask(
                    tpreraretime, exeid, taskid, self.tasktable,datadelay)
                addrowcsv(plantask, [self.nowtime+tpreraretime,
                          int(self.ExeState.add_exeid), taskid])
            else:
                # 资源不足
                for row in self.ExeState.exes:
                    tmp1=[]
                    if (row[3] == devtype1) and (row[4] == devtype2) and (row[5] == devid) and (not self.ExeState.IsResok(devtype1, devtype2, devid, categoryid, self.categorytable)) and (row[0]!=-1):
                        _, _, _, _, out_actionlist1 = self.ExeState.delexe(row[0], self.categorytable)
                        # list1=[1,'Delete',1,'','',''] 并不关心删除哪个devtype1,devtype2,devid
                        #这里缺少一个逻辑：删除应用要等他free，所以del操作应该更晚一些
                        for item in out_actionlist1:
                            tmp1.append(item[0])
                            addrowcsv(
                                planexe, [self.nowtime+item[0], item[1], item[2], item[3], item[4],item[5]])
                        del_whenfree=max(tmp1)#获取最大的del_whenfree 意味全删掉才能ADD
                _, self.newexeid, out_actionlist2 = self.ExeState.addexe(
                    int(devtype1), int(devtype2), devid, categoryid, self.categorytable)
                t2preraretime = max(place_preraretime+del_whenfree, parent_complete_time)
                #这里是ADD
                for item in out_actionlist2:
                    addrowcsv(planexe, [self.nowtime+del_whenfree, item[0],
                              item[1], item[2], item[3], item[4]])
                    
                self.scheduler_reward, self.comdelay = self.ExeState.scheduletask(
                    t2preraretime, exeid, taskid, self.tasktable,datadelay)
                addrowcsv(plantask, [self.nowtime+t2preraretime,
                          int(self.ExeState.add_exeid), taskid])
            #_,_,tpreraretime = get_cpumempretime(categoryid,self.categorytable)
            # tpreraretime=max(tpreraretime,parent_complete_time)
            computedelay = self.comdelay*compute_speed
            reward = computedelay+tpreraretime

            #print(f'taskid{taskid} need{tpreraretime}(s) to ready and {computedelay}s to compute')
        elif acategoryid == categoryid:  # 意味可以直接等上一个任务完成就调度过去
            ppreparetime = self.ExeState.get_whenfree(exeid)
            ppreparetime = max(ppreparetime, parent_complete_time)
            self.scheduler_reward, self.comdelay = self.ExeState.scheduletask(
                ppreparetime, exeid, taskid, self.tasktable,datadelay)
            addrowcsv(plantask, [self.nowtime +
                      ppreparetime, int(exeid), taskid])
            computedelay = self.comdelay*compute_speed
            reward = computedelay+ppreparetime+datadelay
            #print(f'taskid{taskid} need{ppreparetime}(s) to ready and {computedelay}s to compute')
        # this implement in code of train
        # end action #TODO:consider the data trans delay,implement later

        self.taskid_complete_time.append(reward)
        if self.timesteplistindex >= length:
            done = True
        else:
            done = False
        self.lasttime = self.nowtime  # 更新时间
        self.timesteplistindex += 1

        # arrtime=self.timesteplist[self.timesteplistindex]
        #next_state1 = self.ExeState.to_input()  # get all ava param
        #next_state2 = self.jobtask.to_input(self.jobtask.getsingletask(self.tasktable))

        #shape1 = next_state1.shape[0]
        #next_state2 = tf.tile(next_state2, [shape1, 1, 1])  # State shape OK
        #next_state = tf.concat([next_state1, next_state2], axis=-1)
        return 1, reward, done, {}

    def getreward(self):
        # return self.place_reward*10+self.scheduler_reward*10+(1000-self.comdelay)
        return (1000-self.comdelay)
        # TODO:由于ue下线资源变化导致的任务失败，可以在ExecutorInit更新资源信息时判断whenfree是否为闲
        # NOTE simple complete task/job rate and delay
        # CHANGE:placement and shceduler is important,too.


env = Env('./dataset/testset/')
test_set_f = []
for i in range(0, 10):
    test_set_f.append('test'+str(i))

headerofexecutor = ['Time', 'Action', 'ExecutorId',
                    'CategoryId', 'DeviceType', 'DeviceId']
# header of task.csv
headeroftask = ['Time', 'ExecutorId', 'TaskId']

exe = 'executor.csv'
task = 'task.csv'

outpath2 = './621output'
for episode in test_set_f:
    os.makedirs(os.path.join(outpath2, episode))
    planexe = os.path.join(outpath2, episode, exe)
    plantask = os.path.join(outpath2, episode, task)

    initcsv(planexe, headerofexecutor)
    initcsv(plantask, headeroftask)

    env.reset(episode)

    lasttime = 0
    steps = env.alltasktime

    #尝试硬求解
    nomodel=env.ExeState

    taskid = 0
    for step in steps:
        nowtime = step

        o1,exeid,devtype1,devtype2,devid,speed,acategoryid,rate=env.getFast(taskid)

        #(f'task{taskid} get min delay @ {exeid},{devtype1},{devtype2}')
        categoryid, ComputeDuration = get_cateforyinfo(taskid, env.tasktable)
        formate_action = (exeid, devtype1, devtype2, devid,
                          speed, acategoryid,rate)
        env.datainfo.append([get_devtype(devtype1,devtype2),devid,rate])#taskid -devtype devid rate
        #print(env.datainfo)
        _, _, done, _ = env.step(formate_action, taskid, len(steps)-1, planexe, plantask)

        taskid += 1

        if done:  # Check if the episode is done
            break
