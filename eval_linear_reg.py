
#Wrapper function for calculating RMSE values for ALL requested polynomials
def eval_polys(data):
    #Data needs to be 2d so that we can shuffle while retaining x,y pairs
    #Shuffle data for randomness
    np.random.shuffle(data)
    elems = len(data)
    #Define set proportions we want
    train_prop = 0.7
    test_prop = 1 - train_prop
    
    #Slice data array into two seperate arrays depending on proportions
    train_data = data[0:int(elems*train_prop)]
    test_data = data[int(elems*train_prop):]
    
    #Then generate each polynomial again on just the train set
    #We will store RMSE's in a 2D array, where each sub-array will have 
    #[0] as the train RMSE
    #[1] as the test RMSE
    rmses = []
    for i in range(11):
        if i in [0, 1, 2, 3, 5, 10]:
            #RMSE
            temp = []
            train_rmse = eval_pol_regression(train_data[:,0], train_data[:,1], i)
            test_rmse = eval_pol_regression(test_data[:,0], test_data[:,1], i)
            #Store
            temp.append(train_rmse)
            temp.append(test_rmse)
            rmses.append(temp)
    return rmses

#Calculates RMSE between params (predicted y) and y
def calc_rmse(params, x, y):
    rmse = 0
    index = 0
    size = len(y)
    for point in params:
        difference = (point - y[index])**2/size
        rmse = rmse + difference
        index = index + 1
    rmse = np.sqrt(rmse)
    return rmse

#Generates polynomial on data and calculates RMSE
def eval_pol_regression(x, y, degree):
    #Get solution of system based on training data
    system,solution = pol_regression(x, y, degree)
    #Generate predicted values
    yhat = genPoly(x, solution)
    #Generate RMSE 
    rmse = calc_rmse(yhat, x, y)
    return rmse

def plotRMSE(rmses):
    x = [0, 1, 2, 3, 5, 10]
    plt.xticks(x)
    for i in range(len(x)):
        plt.scatter(x[i], rmses[i][0], c="#555FFF")
        plt.scatter(x[i], rmses[i][1], c="#CC0000")
    plt.legend(["Train RMSE", "Test RMSE"])
    plt.show()
	
rmses = eval_polys(data)
plotRMSE(rmses)