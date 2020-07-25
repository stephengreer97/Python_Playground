#!/usr/bin/env python
# coding: utf-8

# # Stephen Greer - Python Practice
# 

# 
# ## Question 1 - Python
# 
# ### Part I:
# 
# Write two 2D lists (4x4) A, B (You can use your own)
# 
# ```python
# A=[[1,2,3,4],
#    [3,4,5,6],
#    [5,6,7,8],
#    [6,7,8,9]]
# B= [[10,11,12,13],
#     [14,15,15,17],
#     [15,16,17,18],
#     [16,17,18,19]]
# ```
# Calculate 
# 
# #### i. A + B
# #### ii. A .* B (Element by element product)
# 
# 
# ### Part II:
# 
# Write a function that convert between US Dollars to British Pound, Turkish Lira, Euro, and Canadian Dollar. Please use following exchange rates 
# 
# 1 USD  = 1.40 Canadian Dollar
# 1 USD  = 0.92 Euro
# 1 USD  = 7.2 Turkish Lira
# 1 USD  = 0.81 British Pound
# 
# Please use the following as a template
# 
# ```python
# def Exc_Conv(x,y,z):
#     '''
#     x : Amount
#     y : 1 -> Canadian Dollar
#         2 -> Euro
#         3 -> Turkish Lira
#         4 -> British Pound
#     z : Nothing -> USD to Canadian Dollar or Euro or Turkish Lira or Pound
#         1 -> Canadian Dollar or Euro or Turkish Lira or Pound to USD
#     '''
#     
#     return 
# 
# Exc_Conv(1000,1)
# 1400 Canadian Dollars
# 
# Exc_Conv(1000,3)
# 7200 Turkish Liras
# 
# Exc_Conv(1000,1,2)
# 1083 USD
# 
# ```

# In[1]:


#Part 1
#i
A=[[1,2,3,4],
   [3,4,5,6],
   [5,6,7,8],
   [6,7,8,9]]
B= [[10,11,12,13],
    [14,15,15,17],
    [15,16,17,18],
    [16,17,18,19]]
add = []
multiply = []
for i in range(len(A)):
    temp = []
    temp1 = []
    for j in range(len(A)):
        temp.append(A[i][j]+B[i][j])
        temp1.append(A[i][j]*B[i][j])
    add.append(temp)
    multiply.append(temp1)
print(add)
#ii
print(multiply)

#Part 2
def Exc_Conv(x,y,*z):
    '''
    x : Amount
    y : 1 -> Canadian Dollar
        2 -> Euro
        3 -> Turkish Lira
        4 -> British Pound
    z : Nothing -> USD to Canadian Dollar or Euro or Turkish Lira or Pound
        1 -> Canadian Dollar or Euro or Turkish Lira or Pound to USD
    '''
    ONE_USD_TO_X = {1:1.4,
                    2:0.92,
                    3:7.2,
                    4:0.81}
    name = {1:'Canadian Dollar',
            2:'Euro',
            3:'Turkish Lira',
            4:'British Pound'}
    
    if(len(z) == 0):
        print(ONE_USD_TO_X[y]*x, name[y])
    else:
        print(1/ONE_USD_TO_X[y]*x, 'USD')
        
print()
Exc_Conv(1000,1)
Exc_Conv(1000,3)
Exc_Conv(1000,1,1)


# ## Question 2 - Plotting
# 
# ### Part I:
# 
# Consider a signal x(t) that has 2000 Hz frequency and 4 Volts amplitude. 
# 
# #### i. Plot 5 cycles of x(t). 
# 
# #### ii. Plot x[n]  if x(t) is sampled with 10000 Hz sampling frequency.
# 
# 
# ### Part II:
# 
# Plot x(t), x(t-1) and x(2t). 
# 
# #### i. Using a single figure 
# #### ii. Using subplots
# 
# PS: All plotting questions please put appropriate axis labels, legend and title. 

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
#Part 1
#i
t = np.linspace(0,0.0025,100000*0.0025)
x = 4*(np.cos(2000*2*np.pi*t))
plt.xlabel('Time (s)')
plt.ylabel('Volts (V)')
plt.title('Signal x(t) Sampling Frequncy = 100000')
plt.plot(t,x,label = 'signal')
plt.xticks(rotation = 45)
plt.legend()
plt.show()

#ii
t = np.linspace(0,0.0025,10000*0.0025)
x = 4*(np.cos(2000*2*np.pi*t))
plt.plot(t,x,label = 'signal')
plt.xlabel('Time (s)')
plt.ylabel('Volts (V)')
plt.title('Signal x[n] Sampling Frequncy = 10000')
plt.show()
#Part 2
#i
t = np.linspace(0,0.0025,1000)
x = 4*(np.cos(2000*2*np.pi*t))
x1 = 4*(np.cos(2000*2*np.pi*t - 1))
x2 = 4*(np.cos(2*2000*2*np.pi*t))
plt.figure(figsize=(12,8))
plt.xlabel('Time (s)')
plt.ylabel('Volts (V)')
plt.title('Signals With Sampling Frequncy = 1000')
plt.plot(t,x,label='x(t)')
plt.plot(t,x1,label='x(t-1)')
plt.plot(t,x2,label='x(2t)')
plt.legend()
plt.show()
#ii
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.title('x(t)')
plt.tick_params(axis='x',which = 'both',bottom=False,labelbottom = False)
plt.plot(t,x)
plt.subplot(3,1,2)
plt.title('x(t-1)')
plt.tick_params(axis='x',which = 'both',bottom=False,labelbottom = False)
plt.plot(t,x1)
plt.ylabel('Volts (V)')
plt.subplot(3,1,3)
plt.title('x(2t)')
plt.plot(t,x2)
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()


# ## Question 3 - Numpy
# 
# ### Part I:
# 
# Please answer following questions:
# 
# #### i. Create a 4×4 numpy array of all True’s.
# 
# #### ii. Create a 4×4 numpy array of all zeros.
# 
# #### iii. Create a 4×4 numpy array of all ones.
# 
# #### iv. Replace all odd numbers in array with 0.
# 
# ```python
# a = array([3, 1, 2, 3, 13, 5, 6, 77, 8, 9,10,11])
# out = array([0, 0, 2, 0, 0, 0, 6, 0, 8, 0,10,0])
# ```
# 
# 
# #### v. Reshape the arrays in  (i,ii,iii) 2x8 and 8x2.
# 
# #### vi. Get the common items between two arrays x and y.
# 
# ```python
# x = array([3, 1, 2, 14, 13, 5, 6, 77, 8, 9,10,11])
# y = array([-3, 10, 2, 30, 3, 5, 60, 7, 98, 19,0,1])
# out = array([3,1,2,5,10])
# ```
# 
# ### Part II:
# 
# Please answer following questions: 
# 
# #### i. Reverse the rows of a 2D array x.
# ```python
# x = np.arange(16).reshape(4,4)
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
# # Desired output
# array([[12, 13, 14, 15],
#        [ 8,  9, 10, 11],
#        [ 4,  5,  6,  7],
#        [ 0,  1,  2,  3]])
# ```
# 
# #### ii. Create a 4x4  2D array containing random 
# ##### a. Random integers between [0,5] 
# ##### b. Random floats between [0,5] 
# 
# #### iiI. Create a 4x4  2D array containing random integers between [0,10]. Replace all values between [3,5] with 
# ##### a. 44
# ##### b. 'HELLO'
# 
# #### Create a 4x4  2D array containing random integers between [0,10]. Find the maximum value in each 
# ##### a. row
# ##### b. column

# In[60]:


import numpy as np
#Part 1
#i
# from https://stackoverflow.com/questions/21174961/how-to-create-a-numpy-array-of-all-true-or-all-false
A = np.ones((4, 4), dtype=bool)
print(A)
#ii
B = np.zeros((4,4))
print(B)
#iii
C = np.ones((4, 4))
print(C)
#iv
a = np.array([3, 1, 2, 3, 13, 5, 6, 77, 8, 9,10,11])
print(a)
for i in range(len(a)):
    if(a[i]%2 == 1):
        a[i] = 0
print(a)
#v
A = np.reshape(A,(2,8))
print(A)
A = np.reshape(A,(8,2))
print(A)
B = np.reshape(B,(2,8))
print(B)
B = np.reshape(B,(8,2))
print(B)
C = np.reshape(C,(2,8))
print(C)
C = np.reshape(C,(8,2))
print(C)
#vi
x = np.array([3, 1, 2, 14, 13, 5, 6, 77, 8, 9,10,11])
y = np.array([-3, 10, 2, 30, 3, 5, 60, 7, 98, 19,0,1])
z = np.empty(0)
for i in x:
    if i in y:
        z = np.append(z,i)
print(z)
#Part 2
#i
x = np.arange(16).reshape(4,4)
x = np.flip(x,axis = 0)
print(x)
#ii
#a
q = np.empty((4,4))
for i in range(len(q)):
    for j in range(len(q[i])):
        q[i][j] = np.random.randint(0,5)
print(q)
#b
r = np.empty((4,4))
for i in range(len(r)):
    for j in range(len(r[i])):
        r[i][j] = 5*np.random.rand()
print(r)
#iii
s = np.empty((4,4))
s = s.astype(object)
for i in range(len(s)):
    for j in range(len(s[i])):
        s[i][j] = np.random.randint(0,10)
print(s)
#a
for i in range(len(s)):
    for j in range(len(s[i])):
        if(s[i][j] == 3 or s[i][j] == 5):
            s[i][j] = 44
print(s)
#b
for i in range(len(s)):
    for j in range(len(s[i])):
        if(s[i][j] == 44):
            s[i][j] = 'HELLO'
print(s)
#iv
t = np.empty((4,4))
for i in range(len(t)):
    for j in range(len(t[i])):
        t[i][j] = np.random.randint(0,10)
print(t)
#a
print('max row',t.max(axis=1))
print('min row',t.min(axis=1))
#b
print('max col',t.max(axis=0))
print('min col',t.min(axis=0))


# ## Question 4 - Pandas
# 
# ### Part I:
# 
# #### i. There are 10 students in a class. Create a dictionary (HW) where key is student names and  values are HW grades. 
# 
# #### ii. Create an DataSet of this dictionary using Pandas
# 
# #### iii. Assign name 'Homework' to this series 
# 
# #### iv. Find max, min and avg of student HW grades using Pandas
# 
# 
# ### Part II:
# 
# #### i. There are 10 students in a class. Create a dictionary (Grades) where key is student names and  values are HW  , Exam, and Project grades . 
# ```python
# Grades={'Albert':[23,44,55],
#         'Jen':[66,77,88]
#        }
# ```
# 
# #### ii.Create an DataFrame (DF1) of this dictionary using Pandas where grades are rows and students names are columns. 
# 
# #### iii .Create an DataFrame (DF2) of this dictionary using Pandas where grades are columns and students names are rows. 
# 
# #### iv .Change the column names of DataFrame (DF2) HW, Exam and Project. 
# 
# #### v. Access the students grades for students between 3 and 7. 
# #### vi. Access the students Exam grades for students between 2 and 5. 
# 
# 
# ### Part III:
# 
# #### i. Save DF2 as an excel file.
# 
# #### ii. Open the excel file using Excel program and add column Quiz and put data and leave few blank. Save this file. 
# 
# #### iii .Open the excel file DataFrame. 

# In[72]:


import pandas as pd
#Part 1
#i
HW = {'Steve':59,
     'John':69,
     'Sam':90,
     'Alex':49,
     'Tom':89,
     'Susan':99,
     'Shawn':80,
     'Austin':23,
     'Daniel':56,
     'Sara':76}
#ii
DS = pd.Series(HW)
print(DS)
#iii
DS = DS.rename('Homework')
print(DS)
#iv
maxx = DS.max()
minn = DS.min()
average = DS.sum()/len(DS)
print('max',maxx)
print('min',minn)
print('average',average)


# In[83]:


#Part 2
Grades={'Albert':[23,44,55],
        'Jen':[66,77,88],
        'Sam':[90,54,24],
        'Alex':[49,67,87],
        'Tom':[56,89,89],
        'Susan':[56,79,99],
        'Shawn':[67,89,80],
        'Austin':[45,67,23],
        'Daniel':[90,78,56],
        'Sara':[90,89,76]}
#ii
DF2 = pd.DataFrame(Grades)
print(DF)
#iii
DF2 = DF2.T
print(DF2)
#iv
DF2.columns = ['HW','Exam','Project']
print(DF2)
#v
print(DF2[3:7])
#vi
print(DF2['Exam'][2:5])


# In[87]:


#Part 3
#i
DF2.to_excel('toexcel.xlsx',index = False)


# In[90]:


#iii
DF2 = pd.read_excel('toexcel.xlsx')
print(DF2)


# In[27]:





# ## Question 5 - Basic Statistics and Probability Using Python: (Numpy and/or Pandas)
# 
# Using the DF2 above
# 
# #### i. Create three lists HW, Exam, Project from DataFrame DF2. 
# 
# #### ii. Find average, max, and min  of HW, Exam, Project grades. 
# 
# #### iii. Find sum and product values  of HW, Exam, Project grades. 
# 
# #### iv. Find mean, standard deviation, and variance HW, Exam, Project grades. 
# 

# In[106]:


#i
hw = DF2['HW']
exam = DF2['Exam']
project = DF2['Project']
print(hw)
print(exam)
print(project)
#ii
print()
hwmax = hw.max()
hwmin = hw.min()
hwavg = hw.sum()/len(hw)
print('hw max', hwmax)
print('hw min', hwmin)
print('hw avg', hwavg)

projmax = project.max()
projmin = project.min()
projavg = project.sum()/len(project)
print('project max', projmax)
print('project min', projmin)
print('project avg', projavg)

exammax = exam.max()
exammin = exam.min()
examavg = exam.sum()/len(exam)
print('exam max', exammax)
print('exam min', exammin)
print('exam avg', examavg)
#iii
print()
hwSum = hw.sum()
hwProduct = hw.product()
print('hw sum',hwSum)
print('hw product',hwProduct)

projSum = project.sum()
projProduct = project.product()
print('project sum',projSum)
print('project product',projProduct)

examSum = exam.sum()
examProduct = exam.product()
print('exam sum',examSum)
print('exam product',examProduct)
#iv
print()
hwavg = hw.sum()/len(hw)
hwsd = hw.std()
print('hw mean', hwavg)
print('hw SD', hwsd)
print('hw variance',hwsd**2)

projavg = project.sum()/len(project)
projsd = project.std()
print('project mean', projavg)
print('project SD', projsd)
print('project variance',projsd**2)

examavg = exam.sum()/len(exam)
examsd = exam.std()
print('exam mean', examavg)
print('exam SD', examsd)
print('exam variance',examsd**2)


# In[ ]:




