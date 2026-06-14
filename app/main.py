from fastapi import FastAPI
import numpy as np
from service import get_gradient_descent, compute_f_wb

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/predict", response_model=float)
async def predict_price(x_inp:float):
    x = np.array([1.0, 2.0, 5.0])
    y = np.array([200, 300, 500])
    # initialize parameters
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, J_hist, p_hist = get_gradient_descent(x ,y, w_init, b_init, tmp_alpha, iterations)
    predicted_price = compute_f_wb(x_inp, w_final, b_final)
    print(type(predict_price))
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    print(f"predicted price found: ({predicted_price:8.2f})")
    
    return round(predicted_price, 2)

    

