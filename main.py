import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import random
import math
from solution import solution
import time
from numpy import array 
from tensorflow.keras.layers import Dense, Activation, LSTM, GRU, SimpleRNN, Conv1D, TimeDistributed, MaxPooling1D, Flatten, Dropout

# Load dataset 
data = pd.read_csv('your data path')
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)

feature_columns = ['your data columns']
scaler = MinMaxScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# create_sequences 
def create_sequences(data, feature_columns, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        # Extract the sequence of features from the current index to the next n_steps
        X.append(data[feature_columns].iloc[i:i+n_steps].values)
        # Append the target variable (Demand) at the next time step
        y.append(data['column as target'].iloc[i + n_steps])
    return np.array(X), np.array(y)

n_steps = 60
X, y = create_sequences(data, feature_columns, n_steps)
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]

# Define LSTM model performance as an objective function for HHO
def lstm_model_performance(hyperparameters):
    # Ensure lstm_units and batch_size are within their respective valid ranges
    lstm_units = max(1, int(hyperparameters[0]))  # Ensure lstm_units is at least 1
    batch_size = max(1, int(hyperparameters[1]))  # Ensure batch_size is at least 1
    
    # compile the LSTM model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'), batch_input_shape=(None, None, 4, 18)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    #-----------------------------------
    
    model.add(Dense(units=7))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])
    # Train 
    try:
        history = model.fit(X_train, y_train, epochs=3, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
    except Exception as e:
        print(f"Encountered exception with lstm_units={lstm_units} and batch_size={batch_size}: {e}")
        val_loss = np.inf  # Assign a high loss for invalid configurations

    return val_loss


import random
import numpy
import math
from solution import solution
import time

def HHO(objf,lb,ub,dim,SearchAgents_no,Max_iter):    
    # initialize the location and Energy of the rabbit
    Rabbit_Location=numpy.zeros(dim)
    Rabbit_Energy=float("inf")  #change this to -inf for maximization problems    
    #Initialize the locations of Harris' hawks
    X=numpy.random.uniform(0,1,(SearchAgents_no,dim)) *(ub-lb)+lb   
    #Initialize convergence
    convergence_curve=numpy.zeros(Max_iter)        
    ############################
    s=solution()
    print("HHO is now tackling  \""+objf.__name__+"\"")    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################
    t=0  # Loop counter
    
    # Main loop
    while t<Max_iter:
        for i in range(0,SearchAgents_no):           
            # Check boundries    
            X[i,:]=numpy.clip(X[i,:], lb, ub)
            # fitness of locations
            fitness=objf(X[i,:])
            # Update the location of Rabbit
            if fitness<Rabbit_Energy: # Change this to > for maximization problem
                Rabbit_Energy=fitness 
                Rabbit_Location=X[i,:].copy() 
        E1=2*(1-(t/Max_iter)) # factor to show the decreaing energy of rabbit    
        # Update the location of Harris' hawks 
        for i in range(0,SearchAgents_no):
            E0=2*random.random()-1;  # -1<E0<1
            Escaping_Energy=E1*(E0)  # escaping energy of rabbit Eq. (3) in the paper
           
            if abs(Escaping_Energy)>=1:
                #Harris' hawks perch randomly based on 2 strategy:
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no*random.random())
                X_rand = X[rand_Hawk_index, :]
                if q<0.5:
                    # perch based on other family members
                    X[i,:]=X_rand-random.random()*abs(X_rand-2*random.random()*X[i,:])

                elif q>=0.5:
                    #perch on a random tall tree (random site inside group's home range)
                    X[i,:]=(Rabbit_Location - X.mean(0))-random.random()*((ub-lb)*random.random()+lb)

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy)<1:
                #Attacking the rabbit using 4 strategies regarding the behavior of the rabbit
                #phase 1: ----- surprise pounce (seven kills) ----------
                #surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r=random.random() # probablity of each event
                
                if r>=0.5 and abs(Escaping_Energy)<0.5: # Hard besiege Eq. (6) in paper
                    X[i,:]=(Rabbit_Location)-Escaping_Energy*abs(Rabbit_Location-X[i,:])

                if r>=0.5 and abs(Escaping_Energy)>=0.5:  # Soft besiege Eq. (4) in paper
                    Jump_strength=2*(1- random.random()); # random jump strength of the rabbit
                    X[i,:]=(Rabbit_Location-X[i,:])-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])
                
                #phase 2: --------performing team rapid dives (leapfrog movements)----------

                if r<0.5 and abs(Escaping_Energy)>=0.5: # Soft besiege Eq. (10) in paper
                    #rabbit try to escape by many zigzag deceptive motions
                    Jump_strength=2*(1-random.random())
                    X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:]);

                    if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                    else: # hawks perform levy-based short rapid dives around the rabbit
                        X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X[i,:])+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                        if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                if r<0.5 and abs(Escaping_Energy)<0.5:   # Hard besiege Eq. (11) in paper
                     Jump_strength=2*(1-random.random())
                     X1=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))
                     
                     if objf(X1)< fitness: # improved move?
                        X[i,:] = X1.copy()
                     else: # Perform levy-based short rapid dives around the rabbit
                         X2=Rabbit_Location-Escaping_Energy*abs(Jump_strength*Rabbit_Location-X.mean(0))+numpy.multiply(numpy.random.randn(dim),Levy(dim))
                         if objf(X2)< fitness:
                            X[i,:] = X2.copy()
                
        convergence_curve[t]=Rabbit_Energy
        if (t%1==0):
               print(['At iteration '+ str(t)+ ' the best fitness is '+ str(Rabbit_Energy)])
        t=t+1
    
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.optimizer="HHO"   
    s.objfname=objf.__name__
    s.best =Rabbit_Energy 
    s.bestIndividual = Rabbit_Location

    return s

def Levy(dim):
    beta=1.5
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) 
    u= 0.01*numpy.random.randn(dim)*sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v),(1/beta))
    step = numpy.divide(u,zz)
    return step

# Integration with HHO
lb = np.array([10, 32])  # Lower bounds for LSTM units and batch size
ub = np.array([100, 256])  # Upper bounds for LSTM units and batch size
dim = 2  # Optimizing two hyperparameters: LSTM units and batch size
SearchAgents_no = 10  # Number of search agents
Max_iter = 3  # Number of iterations

best_solution = HHO(lstm_model_performance, lb, ub, dim, SearchAgents_no, Max_iter)

print("Best Hyperparameters Found:")
print(f"LSTM Units: {int(best_solution.bestIndividual[0])}, Batch Size: {int(best_solution.bestIndividual[1])}")
print(f"Best Validation Loss: {best_solution.best}")


#plot
import matplotlib.pyplot as plt

# Retrain model with the best hyperparameters
best_lstm_units = int(best_solution.bestIndividual[0])
best_batch_size = int(best_solution.bestIndividual[1])

model = Sequential()
model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, activation='relu', padding='same'), batch_input_shape=(None, None, 4, 18)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
#-----------------------------------

model.add(Dense(units=7))
model.compile(loss='mae', optimizer='adam', metrics=['mse'])

# Fit model 
model.fit(X_train, y_train, epochs=3, batch_size=best_batch_size, verbose=0)

# Generate predictions
predictions = model.predict(X_train, verbose=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(y_train, label='Actual', color='blue', marker='o')
plt.plot(predictions, label='Predicted', color='red', linestyle='--')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Values')
plt.legend()
plt.show()

# Train the best model with optimal hyperparameters
lstm_units = int(best_solution.bestIndividual[0])
batch_size = int(best_solution.bestIndividual[1])

model = Sequential()
model.add(LSTM(lstm_units, activation='relu', input_shape=(n_steps, len(feature_columns))))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model on the training set
model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=1)

# Generate predictions on the validation set
predictions = model.predict(X_val)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_val, label='Actual Values')
plt.plot(predictions, label='Predicted Values', alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Demand')
plt.legend()
plt.show()
