import numpy as np
from matplotlib import pyplot as plt

#Read coordinate csv. Col 1 is x coords and Col2 is y coords.
def read_coords(filename):    
    myFile = open(filename)
    row =0
    coords =[]
    for line in myFile:
        coords.append(line.rstrip().split(",")[:])
        #coords[row] = line.rstrip().split(",")[:]
    myFile.close()
    return coords
#Find the sum of x to a certain power. This is used to construct the terms of the left hand side of the system
def findSum(coords,power):
    sumOfCoords = 0
    for point in coords:
        sumOfCoords += point**power
    return sumOfCoords
#Find the sum of x*y where x is raised to a power. This is used to determine terms on the RHS of system
def findEndTerm(coords, y_train, power):
    sumOfCoords = 0
    index = 0
    for point in coords:
        sumOfCoords += (point**power)*y_train[index]
        index += 1
    return sumOfCoords

'''
System looks as follows:
(sum[x^0])*a0 + (sum[x^1])*a1 + ... + (sum[x^n])*an = sum[x^m * y]
(sum[x^1])*a0 + (sum[x^2])*a1 + ... + (sum[x^n])*an = sum[x^m * y]
(sum[x^2])*a0 + (sum[x^3])*a1 + ... + (sum[x^n])*an = sum[x^m * y]

Notice that the first term in each row, is always to the power of the zero-indexed row number, so
in row 1 ( row 0 ) we start with 0, and pattern remains, so we will start iteraing from `rowNumber`
Then it goes up to rowNumber + dimension, so if our data is a 3x3 square matrix, dimension is 3 so on each line
we iterate up to rowNumber + 3. (Basically take r+c for each power where r is row number and c is col number)
The power term is then always raised to the same power as the first term, so the index of the current row. 

Using these rules we can form our system:
'''
def formSystem(coords, y_train, dim):
    #Define system arr
    system = []
    
    for row in range(dim):
        #Define current row, that will contain column data as its entries
        temp = []
        #We iterate up to row number + dimension on each row, +1 for inclusivity
        for col in range(row, row+dim+1):
            #If this isnt the last row then find the sum term
            if(col != row+dim):
                sumToPow = findSum(coords, col)
                temp.append(sumToPow)
            #If got to last term, calculate that
            else:
                lastTerm = findEndTerm(coords, y_train, row)
                temp.append(lastTerm)
        #Add row
        system.append(temp)
    return system

#Wrapper for regression
def pol_regression(features_train, y_train, degree):
    system = formSystem(features_train, y_train, degree+1)
    
    #Convert to np array
    system = np.asarray(system)
    #.shape[1] is how many columns are in each row
    cols = system.shape[1]
    #keep all but the last row (RHS)
    a = system[:,0:cols-1]
    #Right hand side is then 
    b = system[:,cols-1]
	#Solve left hand against right hand
    solution = np.linalg.solve(a, b)
    return system, solution

#Generate polynomial dynamically from coefficients and set of x ([-5, 5])
def genPoly(x, coeffs):
    o = len(coeffs)
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y
#Plot polynomials and data set
def plot_polynomials(solutions, train_x, train_y):
    #Interval [-5, 5]
    x = np.arange(-5, 6)
    #Plot current polynomial and label appropriately for legend detection
    plt.plot(x, genPoly(x, solutions), label="x^"+str(len(solutions)-1))
    plt.title("Polynomial Regression on Train data set")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    return plt
#Add training points and display
def finish_plot(plot, train_x, train_y):
    #Set these points to appear on foreground
    plot.scatter(train_x, train_y, label="Test data", zorder=5, c="#000000")
    plot.legend()
    #Need to limit the axes otherwise the x^10 polynomial goes down to ~-2000 on the y axis
    #rendering everything else unreadable
    plot.gca().set_ylim([-250, 80])
    plot.show()

#Read in data, convert to numpy array and to float array
data = read_coords("ML_task1.csv")
data = np.asarray(data)
data = data.astype(float)
#system,solution = pol_regression(data[:,0], data[:,1], 2)
#plot_polynomials(solution, data[:,0], data[:,1])

#Contains matplotlib.pylot.plot, init to retain in the scope
currPlot = 0
#Generate graph by iterating from 0 to 11 
for i in range(11):
    #Degrees that we want to generate
    if i in [0, 1, 2, 3, 5, 10]:
        #Get polynomial
        system,solution = pol_regression(data[:,0], data[:,1], i)
        #Plot and return current state of plot
        currPlot = plot_polynomials(solution, data[:,0], data[:,1])
    else:
        continue
#When we got everything, show
finish_plot(currPlot, data[:,0], data[:,1] )
