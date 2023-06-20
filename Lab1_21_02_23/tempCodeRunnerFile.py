# Activation Function: Sigmoid
def f(x,w,b): 
    return 1/(1+np.exp(-(w*x+b)))

# def error(w,b):
#     err = 0.0
#     for x,y in zip(X,Y):
#         fx = f(x,w,b)
#         err += (fx-y)**2
#     return 0.5*err

def grad_b(x,w,b,y):
    fx = f(x,w,b)
    return (fx-y)*fx*(1-fx)

def grad_w(x,w,b,y):
    fx = f(x,w,b)
    return (fx-y)*fx*(1-fx)*x

i = 1
list_w = []
list_b = []
def gradient_descent():
    w,b,eta,max_epochs = 0,0,0.5,4000
    for i in range(max_epochs):
        dw,db = 0,0
        for x,y in zip(X,Y):
            dw += grad_w(x,w,b,y)
            db += grad_b(x,w,b,y)
        w = w - eta*dw
        b = b - eta*db  
        
        list_w.append(w)
        list_b.append(b)
        
        print('Weight: ', w, '    Bias: ', b)
        #print('Bias: ', b)
        print('Iteration ', i+1)


# In[41]:


gradient_descent()

list_w
list_b
    
import matplotlib.pyplot as plt
plt.figure(figsize = (8,6))
plt.plot(list_w, list_b)
plt.scatter(list_w, list_b, marker='o', color='red')
plt.title("Bias vs Weights")
plt.ylabel("Bias")
plt.xlabel("Weight")
plt.show()


# # Momentum Based Gradient Descent
