import math
import numpy as np

def calculate_cost(x, y, w, b):
    m = len(x)
    
    f_wb = compute_f_wb(x, w , b)

    cost = np.sum((f_wb - y)**2)

    total_cost = 1 / (2 * m) * cost

    return total_cost

def calculate_gradient(x, y, w, b):
    m = len(x)

    f_wb = compute_f_wb(x, w, b)

    dj_dw = (f_wb - y) * x
    dj_db = (f_wb - y)

    w = np.sum(dj_dw) / m
    b = np.sum(dj_db) / m

    return w, b

def compute_f_wb(x, w, b):
    return w * x + b

def get_gradient_descent(x, y, init_w, init_b, a, iterations):

    cost_history = []
    wb_history = []
    w = init_w
    b = init_b
    m = len(x)

    for i in range(iterations):
        dj_dw, dj_db = calculate_gradient(x, y, w, b)

        w = w - a * dj_dw
        b = b - a * dj_db

        cost = calculate_cost(x, y, w, b)
        if i<100000:
            cost_history.append(cost)
            wb_history.append((w, b))
        if i% math.ceil(iterations/10) == 0:
            print(f"Iteration {i:4}: Cost {cost_history[i]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, cost_history, wb_history

