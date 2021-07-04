

def majority_vote(X: np.ndarray, Y: np.ndarray) -> str:
    """This function computes the output label of the given dataset, following the 
    majority vote algorithm

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances

    Returns the majority label
    """
    ### YOUR CODE HERE
    
    # We're going to use a dictionary to count the votes. Not the
    # most efficient way to do it, but convenient.
    votes = {}
        
    # Loop over the Y labels.
    for attr_value in Y:
        # If we have seen this label before, increment its count
        if attr_value in votes:
            votes[attr_value] += 1
        # If we've never seen this label before, initialize its
        # count to one.
        else:
            votes[attr_value] = 1
            
    # Now find the maximum value in the dictionary we just built
    max_value = 0
    
    # The label corresponding to the max value
    max_values_label = None
    
    for (k, v) in votes.items():
        # Compare the vote count (v of the key-value pair) to the
        # maximum so far. If it's greater, then we have a new maximum.
        if v > max_value:
            max_value = v
            max_values_label = k
    
    return str(max_values_label) # replace this line with your return statement


def split(X: np.ndarray, 
          Y: np.ndarray, 
          split_attribute: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function takes a dataset and splits it into two sub-datasets according 
    to the values of the split attribute. The left and right values of the split 
    attribute should be in alphabetical order. The left dataset should correspond 
    to the left attribute value, and similarly for the right dataset. 

    Parameters
    ----------
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    split_attribute: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of two sub-datasets, i.e. (X_left, Y_left, X_right, Y_right)
    """
    ### YOUR CODE HERE 
        
    # Let's do some sanity checks: in particular that their N values
    # are the same:
    x_n, x_m = X.shape
    y_n, = Y.shape

    if x_n != y_n:
        print("Error: X and Y have inconsistent shapes")
        return None, None, None, None
    
    # It would be really nice to write a general-purpose n-way
    # split function that can take any sort of labels, but I'm
    # short on time, so I'll take advantage of the fact that
    # our datasets have only binary values and that they're
    # always called '0' and '1'.
    
    # Use lists to accumulate the results; later turn them into
    # ndarrays.
    x_left_list = []
    x_right_list = []
    y_left_list = []
    y_right_list = []
    
    # Loop through the attribute we're splitting on
    for index in range(x_n):
        test_val = X[index, split_attribute]
        
        if test_val == '0':
            # It goes on the left
            x_left_list.append(X[index])
            y_left_list.append(Y[index])
            
        else: # test_val == '1'
            # It goes on the right
            x_right_list.append(X[index])
            y_right_list.append(Y[index])

    # Use the ndarray constructors to create the return objects
    X_left = np.array(x_left_list)
    X_right = np.array(x_right_list)
    Y_left = np.array(y_left_list)
    Y_right = np.array(y_right_list)
        
    return X_left, Y_left, X_right, Y_right


def train(X_train: np.ndarray, Y_train: np.ndarray, attribute_index: int) -> Tuple[str, str]:
    """This function takes the training dataset and a split attribute and outputs the 
    tuple of (left_label, right_label), corresponding the label on the left and right 
    leaf nodes of the decision stump

    Parameters
    ----------
    X_train: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N training instances
    Y_train: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N training instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the tuple of labels, i.e. (left_label, right_label)
    """
    ### YOUR CODE HERE 
    
    x_left, y_left, x_right, y_right = split(X_train, Y_train, split_attribute=attribute_index)
        
    left_label = majority_vote(x_left, y_left)
    right_label = majority_vote(x_right, y_right)
        
    return left_label, right_label # replace this line with your return statement


def predict(left_label: str, right_label: str, X: np.ndarray, attribute_index: int) -> np.ndarray:
    """This function takes in the output of the train function (left_label, right_label), 
    the dataset without label (i.e. X), and the split attribute index, and returns the 
    label predictions for X

    Parameters
    ----------
    left_label: type `str`
        The label corresponds to the left leaf node of the decision stump
    right_label: type `str`
        The label corresponds to the right leaf node of the decision stump
    X: type `np.ndarray`, shape (N, M)
        Numpy arrays with N rows, M columns containing the attribute values for N instances
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns the numpy arrays with shape (N,) containing the label predictions for X
    """
    ### YOUR CODE HERE
    
    # Use a python list to build up the return array
    return_value = []
    
    for row in X:
        if row[attribute_index] == '0':
            return_value.append(left_label)
        else:
            return_value.append(right_label)


    # Convert the return array to a np array
    return np.array(return_value) # replace this line with your return statement  


def error_rate(Y: np.ndarray, Y_pred: np.ndarray) -> float:    
    """This function computes the error rate (i.e. number of incorrectly predicted
    instances divided by total number of instances)

    Parameters
    ----------
    Y: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the true labels for N instances
    Y_pred: type `np.ndarray`, shape (N, )
        Numpy arrays with N rows containing the predicted labels for N instances

    Returns the error rate, which is a float value between 0 and 1 
    """
    ### YOUR CODE HERE
    
    error_count = 0
    array_len, = Y.shape
    
    for index in range(array_len):
        if Y[index] != Y_pred[index]:
            error_count += 1
    
    return error_count / array_len # replace this line with your return statement  


def train_and_test(train_filename: str, test_filename: str, attribute_index: int) -> Dict[str, Any]:
    """This function ties the above implemented functions together. The inputs are 
    filepaths of the train and test datasets as well as the split attribute index. The
    output is a dictionary of relevant information (e.g. train and test error rates)

    Parameters
    ----------
    train_filename: type `str`
        The filepath to the training file
    test_filename: type `str`
        The filepath to the testing file
    attribute_index: type `int`
        The index of the attribute to split the dataset on

    Returns an output dictionary
    """
    X_train, Y_train, attribute_names = load_data(train_filename)
    X_test, Y_test, _ = load_data(test_filename)
    
    left_label, right_label = train(X_train, Y_train, attribute_index)
    
    Y_pred_train = predict(left_label, right_label, X_train, attribute_index)
    Y_pred_test = predict(left_label, right_label, X_test, attribute_index)
    
    train_error_rate = error_rate(Y_pred_train, Y_train)
    test_error_rate = error_rate(Y_pred_test, Y_test)

    return {
        'attribute_names' : attribute_names,
        'stump'           : (left_label, right_label),
        'train_error_rate': train_error_rate,
        'test_error_rate' : test_error_rate
    }

