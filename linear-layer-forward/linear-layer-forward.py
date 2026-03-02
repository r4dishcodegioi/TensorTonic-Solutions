def linear_layer_forward(x, w, b):
    N = len(x)          
    D_in = len(x[0])     
    D_out = len(w[0])    
    
    out = [[0.0 for _ in range(D_out)] for _ in range(N)]
    
    for i in range(N):
        for j in range(D_out):
            dot_product = 0.0
            for k in range(D_in):
                dot_product += x[i][k] * w[k][j]
            out[i][j] = dot_product + b[j]
            
    return out