"""
Created on Tue Mar 11 09:02:17 2025

@author: Harry Thawley
"""

from datetime import datetime as dt
from datetime import timedelta as td
from numba import njit, objmode, float64, int64
from numba.experimental import jitclass
import numpy as np
import pandas as pd

from firm_ce.constants import JIT_ENABLED

if JIT_ENABLED:
    from numba import njit
else:
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        def wrapper(f):
            return f
        return wrapper

spec = [
        ('functions', int64[:]), 
        ('times', float64[:,:]),
        ('calls', int64[:])
        ]

@jitclass(spec)
class Timekeeper:
    def __init__(self):
        self.functions = np.zeros(1, dtype=np.int64)
        self.times = np.zeros((1, 3), dtype=np.float64)
        self.calls = np.zeros(1, dtype=np.int64)
    def Update(self, index, time):
        if len(self.functions) < index+1:
            self.functions = np.arange(index+1, dtype=np.int64)
        while len(self.times) < index+1:
            self.times = np.vstack((self.times, np.zeros((1,3), np.float64)))
        while len(self.calls) < index+1:
            self.calls = np.concatenate((self.calls, np.zeros(1, np.int64)))
        self.times[index] += time
        self.calls[index] += 1
        
def Print_tk(tk, console=True, path=None, namedict={}):
    if console is True:
        print("Timekeeper","="*50, sep='\n')
        if len(namedict) > 0:
            for index in tk.functions:
                print(f'Function: {namedict[index]}. Calls: {tk.calls[index]}. Time: {td(*tk.times[index])}.')
        else:
            for index in tk.functions:
                print(f'Function: {index}. Calls: {tk.calls[index]}. Time: {td(*tk.times[index])}.')
    if path is not None:
        if len(namedict) > 0:
            indices = [namedict[index] for index in tk.functions]
        else: 
            indices = tk.functions
            
        result = pd.DataFrame([], index=indices, 
                              columns=['calls', 'time'])
        for i, index in enumerate(indices):
            result.loc[index,:] = tk.calls[i], td(*tk.times[i])
        result.to_csv(path)

@njit 
def dt_now():
    now = np.empty(7, np.int64)
    with objmode():
        n = dt.now()
        now[:] = np.array([n.year, n.month, n.day, n.hour, n.minute, n.second, n.microsecond], np.int64)
    return now

@njit
def time_delta(start, end):
    delta = np.empty(3, np.float64)
    with objmode():
        d = dt(*end) - dt(*start)
        delta[:] = np.array([d.days, d.seconds, d.microseconds], np.float64)
    return delta

def keeptime(timekeeper, name):
    def decorator(func):
        def wrapper(*fargs):
            start=dt_now()
            ret=func(*fargs)
            timekeeper.Update(name, time_delta(start, dt_now()))
            return ret
        return wrapper
    return decorator

timekeeper = Timekeeper()

if __name__=='__main__':
    from time import sleep
    tk=Timekeeper()
    
    @keeptime(tk, 0)
    @njit
    def func1(n=1_000_000):
        x = list(range(n))
        y = [y for y in x] 
        return 
    
    @keeptime(tk, 1)
    @njit
    def func2(n=1_000_000):
        x = list(range(n))
        y = [y for y in x] 
        return 
    
    @keeptime(tk, 2)
    @njit
    def func3(n=3):
        with objmode():
            sleep(3)
        return 
 
    @keeptime(tk, 3)
    def func_rec():
        func1()
        func3()
         
 
    @keeptime(tk,4)
    def func4():
        sleep(2)
        return
    
    func1()
    func3()
    func2()
    func4()
    func_rec()
    namedict={0:'func1', 
              1:'func2', 
              2:'func3',
              3:'func_rec',
              4:'func4'}
    Print_tk(tk, True, 'results/timekeeper_test.csv', namedict)