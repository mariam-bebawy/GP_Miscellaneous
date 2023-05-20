def opticalFlow(ref, curr, type):
    if (type == 0): return "lucas kanade"
    if (type == 1): return "horn-schunk"
    if (type == 2): return "farneback"


def sliceSelection(vec, mode, type):
    size = vec.shape[0]
    
    if (mode == 0):
        ref, curr = vec[0], vec[1]
        for i in (vec.shape[0] - 1):
            opticalFlow(ref, curr, type)
            curr = vec[i+1]
            
    if (mode == 1):
        ref, curr = vec[0], vec[1]
        for i in (vec.shape[0] - 1):
            opticalFlow(ref, curr, type)
            ref = vec[i]
            curr = vec[i+1]
            
    if (mode == 2):
        ref, curr = vec[0], vec[2]
        for i in (vec.shape[0] - 2):
            opticalFlow(ref, curr, type)
            ref, curr = vec[i], vec[i+2]
            
    if (mode == 3):
        ref, curr = vec[0], vec[3]
        for i in (vec.shape[0] - 3):
            opticalFlow(ref, curr, type)
            ref, curr = vec[i], vec[i+3]
