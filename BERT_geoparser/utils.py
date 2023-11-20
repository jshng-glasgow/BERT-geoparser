import numpy as np

def flatten(lists):
    "Flattens a list of lists into a single list."
    return [x for sublist in lists for x in sublist]

def convert(int32:np.int32)->int:
    """Utility function to convert int32 objects into regular int.
    parameters
    ----------
    value : np.int32
        numpy.int32 object to be converted.
    return : int
    """
    if isinstance(int32, np.int32): 
        return int(int32)  
    raise TypeError("Input should be np.int32")