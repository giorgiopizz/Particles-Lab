import numpy as np
def matrix_population(start_stop):
    M = np.array([])
    i=0
    while i<len(start_stop):
        if start_stop[i] == 'A':
            l = [0]
            for j in range(i+1, len(start_stop)):
                if start_stop[j] == 'A':
                    i = j
                    l = [0]
                elif start_stop[j] == 'C':
                    l.append(1)
                elif start_stop[j] == 'B':
                    l.append(1)
                    row = np.concatenate((np.zeros(i), np.array(l), np.zeros(len(start_stop)-(j+1))))
                    if M.size!=0:
                        M = np.vstack([M,row])
                    else:
                        M = np.array([row])
                    i = j+1
                    break
        i+=1
    return M


s = ['A','B','C','C','B','A','A','C','C','B','A','A','A','C','C','A','A','C','B']


M = matrix_population(s)
print(M)
print((M.shape[1]))
