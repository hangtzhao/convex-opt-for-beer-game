# convex-opt-for-beer-game
copy rights reserved

python ***code***

```
import cvxpy as cp
from cvxpy import maximum, abs, sum
import numpy as np

np.random.seed(0)
import logging

LOG_FORMAT = "%(message)s"
logging.basicConfig(
            filename=f'optimal.log', 
            level=logging.DEBUG,
            format=LOG_FORMAT)

week = 20

Customer = np.array([2,4,8,4,4,4,4,8,4,4,4,4,4,4,4,8,4,4,4,4])
print(len(Customer))

def val_cost_func(cost, A_REQ, A_SI, A_EI, A_OUT, Order, A_ORD):
    for i in range(week):
      A_REQ[i] = Order[i]
      if i == 0:
         A_SI[i] = 8
         A_EI[i] = maximum(A_SI[i] - A_REQ[i],0)
         A_OUT[i] = maximum(A_REQ[i]-A_SI[i], 0)
      elif i == 1:
         A_SI[i] = A_EI[i-1]
         A_EI[i] = maximum(A_SI[i] - A_REQ[i] - A_OUT[i-1], 0)
         A_OUT[i] = maximum(A_OUT[i-1] + A_REQ[i]- A_SI[i], 0)
      else:
         acc_req = sum(A_REQ[:i+1])
         acc_ord = sum(A_ORD[:i-1])   
         A_SI[i] = A_ORD[i-2] + A_EI[i-1] 
         A_EI[i] = maximum(8+acc_ord-acc_req, 0)
         A_OUT[i] = maximum(acc_req-8-acc_ord, 0)
      cost += (A_SI[i] + A_EI[i])/2 + 2*abs(A_OUT[i])
    return cost


def cost_func(cost, A_REQ, A_SI, A_EI, A_OUT, Order, A_ORD):
    for i in range(week):
      A_REQ[i] = Order[i]

      if i == 0:
         A_SI[i] = 8
         A_EI[i] = A_SI[i] - A_REQ[i]
         A_OUT[i] = 0 
      elif i == 1:
         A_SI[i] = A_EI[i-1]
         A_EI[i] = maximum(A_SI[i] - A_REQ[i] - A_OUT[i-1], 0)
         A_OUT[i] = A_OUT[i-1] + A_REQ[i]- A_SI[i]
      else:
         acc_req = sum(A_REQ[:i+1])
         acc_ord = sum(A_ORD[:i-1])   
         A_SI[i] = A_ORD[i-2] + A_EI[i-1] 
         A_EI[i] = maximum(8+acc_ord-acc_req, 0)
         A_OUT[i] = acc_req-8-acc_ord
      cost += (A_SI[i]/2) + A_EI[i]/2 + 2*abs(A_OUT[i])
    return cost


def final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD,
          B_REQ, B_SI, B_EI, B_OUT, B_ORD,
          C_REQ, C_SI,C_EI,C_OUT, C_ORD,
          D_REQ, D_SI, D_EI,D_OUT, D_ORD,
          E_REQ, E_SI,E_EI,E_OUT, E_ORD):
    cost = 0
    cost = cost_func(cost, A_REQ, A_SI,A_EI,A_OUT, Customer, A_ORD)
    cost = cost_func(cost, B_REQ, B_SI,B_EI,B_OUT, A_ORD, B_ORD)
    cost = cost_func(cost, C_REQ, C_SI,C_EI,C_OUT, B_ORD, C_ORD)
    cost = cost_func(cost, D_REQ, D_SI,D_EI,D_OUT, C_ORD, D_ORD)
    cost = cost_func(cost, E_REQ, E_SI,E_EI,E_OUT, D_ORD, E_ORD)
    return cost


def val_final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD,
          B_REQ, B_SI, B_EI, B_OUT, B_ORD,
          C_REQ, C_SI,C_EI,C_OUT, C_ORD,
          D_REQ, D_SI, D_EI,D_OUT, D_ORD,
          E_REQ, E_SI,E_EI,E_OUT, E_ORD):
    cost = 0
    cost = val_cost_func(cost, A_REQ, A_SI,A_EI,A_OUT, Customer, A_ORD)
    cost = val_cost_func(cost, B_REQ, B_SI,B_EI,B_OUT, A_ORD, B_ORD)
    cost = val_cost_func(cost, C_REQ, C_SI,C_EI,C_OUT, B_ORD, C_ORD)
    cost = val_cost_func(cost, D_REQ, D_SI,D_EI,D_OUT, C_ORD, D_ORD)
    cost = val_cost_func(cost, E_REQ, E_SI,E_EI,E_OUT, D_ORD, E_ORD)
    return cost

value = Customer[:week]

A_ORD = cp.Variable(week,value=value,integer=True)
B_ORD = cp.Variable(week,value=value,integer=True)
C_ORD = cp.Variable(week,value=value,integer=True)
D_ORD = cp.Variable(week,value=value,integer=True)
E_ORD = cp.Variable(week,value=value,integer=True)

A_SI = [0]*week
A_EI  = [0]*week
A_REQ = [0]*week
A_OUT = [0]*week

B_SI = [0]*week
B_EI  = [0]*week
B_REQ = [0]*week
B_OUT = [0]*week

C_SI = [0]*week
C_EI  = [0]*week
C_REQ = [0]*week
C_OUT = [0]*week

D_SI = [0]*week
D_EI  = [0]*week
D_REQ = [0]*week
D_OUT = [0]*week

E_SI = [0]*week
E_EI  = [0]*week
E_REQ = [0]*week
E_OUT = [0]*week 


min_cost = 100000
iteration = 10000000
prev_cost = 0
A_ORD_temp = np.array([0]*20) 
B_ORD_temp = np.array([0]*20)
C_ORD_temp = np.array([0]*20)
D_ORD_temp = np.array([0]*20)
E_ORD_temp = np.array([0]*20)

for i in range(5*iteration):
    
    if i % 5  == 0:
        cost_f = final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD_temp,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD_temp,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD_temp,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD_temp)
        train = A_ORD
        train_temp = A_ORD_temp
    elif i % 5 == 1:
        cost_f = final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD_temp,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD_temp,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD_temp,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD_temp)
        train = B_ORD
        train_temp = B_ORD_temp
    elif i % 5 == 2:
        cost_f = final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD_temp,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD_temp,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD_temp,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD_temp)
        train = C_ORD
        train_temp = C_ORD_temp
    elif i % 5 == 3:
        cost_f = final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD_temp,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD_temp,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD_temp,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD_temp)
        train = D_ORD
        train_temp = D_ORD_temp
    elif i % 5 == 4:
        cost_f = final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD_temp,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD_temp,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD_temp,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD_temp,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD)
        train = E_ORD
        train_temp = E_ORD_temp

    obj = cp.Minimize(cost_f)
    
    constraints = [ A_ORD <= 8,
                  B_ORD <= 8,
                  C_ORD <= 8,
                  D_ORD <= 8,
                  E_ORD <= 8,
                  A_ORD >=0,
                  B_ORD >=0,
                  C_ORD >=0,
                  D_ORD >=0,
                  E_ORD >=0]
    prob = cp.Problem(obj, constraints)
  
    prob.solve(solver=cp.GLPK_MI, verbose=False, gp=False)
    
    if i % 5  == 0:
        A_ORD_temp = A_ORD.value
    elif i % 5 == 1:
        B_ORD_temp = B_ORD.value 
    elif i % 5 == 2:
        C_ORD_temp = C_ORD.value 
    elif i % 5 == 3:
        D_ORD_temp = D_ORD.value 
    elif i % 5 == 4:
        E_ORD_temp = E_ORD.value 

    val_cost = val_final_cost(A_REQ, A_SI, A_EI, A_OUT, A_ORD_temp,
      B_REQ, B_SI, B_EI, B_OUT, B_ORD_temp,
      C_REQ, C_SI,C_EI,C_OUT, C_ORD_temp,
      D_REQ, D_SI, D_EI,D_OUT, D_ORD_temp,
      E_REQ, E_SI,E_EI,E_OUT, E_ORD_temp)
    
    if prev_cost == val_cost.value and i%5 == 4:  
        
        logging.debug("change random value")
        print("change random value")
        if i % 5  == 0:
            A_ORD_temp = np.maximum(A_ORD.value + np.random.randint(-2,2,size=week),0)
        elif i % 5 == 1:
            B_ORD_temp = np.maximum(B_ORD.value + np.random.randint(-2,2,size=week),0)
        elif i % 5 == 2:
            C_ORD_temp = np.maximum(C_ORD.value + np.random.randint(-2,2,size=week),0)
        elif i % 5 == 3:
            D_ORD_temp = np.maximum(D_ORD.value + np.random.randint(-2,2,size=week),0)
        elif i % 5 == 4:
            E_ORD_temp = np.maximum(E_ORD.value + np.random.randint(-2,2,size=week),0)
    prev_cost = val_cost.value
    #print(f"optimal {cost_f}")
    
    if min_cost > val_cost.value:
        min_cost = val_cost.value
        
        final_A_ORD = A_ORD_temp
        final_B_ORD = B_ORD_temp
        final_C_ORD = C_ORD_temp
        final_D_ORD = D_ORD_temp
        final_E_ORD = E_ORD_temp
  
    logging.debug(f"iteration {i // 5} cost: {val_cost.value} \n A:{A_ORD_temp}\n B:{B_ORD_temp}\n C: {C_ORD_temp}\n D:{D_ORD_temp}\n E :{E_ORD_temp}\n")
