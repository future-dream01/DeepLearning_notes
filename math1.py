import numpy as np
def main():
    matrix_t=np.matrix([[0.5,0.5],
                        [0.5,0.5]],dtype=float)
    vector_p=np.matrix([[0.1,0.9]],dtype=float)
    for i in range(0,100):
        vector_after=vector_p*matrix_t
        vector_p=vector_after
        print (vector_after)
main()