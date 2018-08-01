import numpy as np

"""
    For conv2D methods:
        Weights shape must be in form of (o, i, k_h, k_w), where 'o' stands for number of outputs, 'i' number of inputs, 'k_h' 
        is kernel height and 'k_w' is kernel width 

        fMaps stands for Feature Maps, or input images, its shape must be in form of (i, h, w), where 'i' is the number of inputs,
        'h' is image height and 'w' is image width 

    For dense method:
        Weights shape must be in form of (o, i), where 'o' stands for number of outputs and 'i' number of inputs

        fMaps stands for Feature Maps, or input images, its a flattened array 

"""


# Convolves feature maps and weights
def conv2D(fMaps, weights, bias, padding='same'):
    kernels_per_fmap = weights.shape[1]
    image_initial_addr = [0,0]
    
    if padding == 'same':
        ### PADDING == 'SAME'
        if len(fMaps.shape) > 2:
            fMap_height = fMaps.shape[1]
            fMap_width = fMaps.shape[2]
            convolved_fMap = np.zeros((weights.shape[0], fMaps.shape[1], fMaps.shape[2]))
        else:
            fMap_height = fMaps.shape[0]
            fMap_width = fMaps.shape[1]
            convolved_fMap = np.zeros((weights.shape[0], fMaps.shape[0], fMaps.shape[1]))
        
        for j, w in enumerate(weights): # Loops over FMaps weights
            convolved_rows = -1
            while(convolved_rows < (fMap_height-1)):
                convolved_cols = -1
                while (convolved_cols < (fMap_width-1)):
                    convolved = np.zeros((3,3)) #Convolved Matrix
                    for i, kernel in enumerate(w): # Loops over weights 
                        # Convolve image and kernel
                        for col in range(0, 3):
                            for row in range(0, 3):
                                col_addr = convolved_cols + col
                                row_addr = convolved_rows + row
                                # Verifies if convolution is occurring at image borders
                                if col_addr < image_initial_addr[1] or col_addr == fMap_width or row_addr < image_initial_addr[0] or row_addr == fMap_height:
                                    convolved[row][col] += 0
                                else:
                                    convolved[row][col] += (fMaps[i][row_addr,col_addr] * kernel[row][col])
                                    
                    summ = np.asarray(convolved).sum() + bias[j]
                    
                    convolved_fMap[j][convolved_rows+1, convolved_cols+1] = summ

                    convolved_cols += 1

                convolved_rows += 1   # Counts how many lines have been convolved

    else:
        ### PADDING == 'VALID'
        if len(fMaps.shape) > 2:
            fMap_height = fMaps.shape[1]
            fMap_width = fMaps.shape[2]
            convolved_fMap = np.zeros((weights.shape[0], fMaps.shape[1]-2, fMaps.shape[2]-2))
        else:
            fMap_height = fMaps.shape[0]
            fMap_width = fMaps.shape[1]
            convolved_fMap = np.zeros((weights.shape[0], fMaps.shape[0]-2, fMaps.shape[1]-2))
        
        for j, w in enumerate(weights): # Loops over FMaps weights
            im_row_addr = image_initial_addr[0]
            while(im_row_addr < (fMap_height - 2)):
                im_col_addr = image_initial_addr[1]
                while (im_col_addr < (fMap_width-2)):
                    convolved = np.zeros((3,3)) #Convolved Matrix
                    #Creates a 3x3 kernel matrix
                    for i, kernel in enumerate(w): # Loops over weights 
                        for k in range(0, 3):
                            convolved[0][k] += (fMaps[i][im_row_addr,k+im_col_addr] * kernel[0][k])
                            convolved[1][k] += (fMaps[i][1+im_row_addr,k+im_col_addr] * kernel[1][k])
                            convolved[2][k] += (fMaps[i][2+im_row_addr,k+im_col_addr] * kernel[2][k])
                            
                    summ = np.asarray(convolved).sum() + bias[j]
                    
                    convolved_fMap[j][im_row_addr, im_col_addr] += summ

                    im_col_addr += 1
                    
                im_row_addr += 1   # Counts how many lines have been denoised
        
    return convolved_fMap


### DENSE 
def dense(fMap, weights, bias):
    
    out = np.zeros((weights.shape[0],)) ## Output vector
    
    for j, w in enumerate(weights):
        summ = 0
        for i, k in enumerate(w):
            summ += k*fMap[i]
            
        summ = summ.sum() + bias[j]
        
        out[j] = summ
        
    return out

def softmax(x):
    ### Compute softmax values for each sets of scores in x.
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def LeakyReLU(x, alpha): ### Leaky Rectified Linear Unit activation
    ## If 'alpha' is equal to zero, then it becomes a standard ReLU 
    return np.maximun(x, alpha*x)
