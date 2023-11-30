def predict(x, P, u, dt):
    # Prediction for the position due to initial velocity and gravity
    g=-9.81
    x+=u*dt+0.5*g*dt**2

    # Predicted covariance
    A = 1
    B = dt
    Q = 0.01 # control noise (launching system is not perfect)
    
    P = A*P*A+B*Q*B

    return x, P

def update(x, P, z):
    # measurement noise
    R = 0.1  
    
    # Measurement Jacobian
    H = 1
        
    # Innovation covariance
    S = H*P*H+R
    
    # Kalman gain
    K = P*H/S
    
    # Innovation 
    y = z-x

    # Update state estimate
    x = x+K*y
    
    # Update covariance estimate
    P = (1-K*H)*P

    return x, P

def main():
    # Initial position (on the ground)
    x = 0.0 # state mean
    P = 0.1 # state covariance

    # Launch velocity (control input)
    u = 10.0 #[m/s]
    
    # Time when estimation performs
    dt = 1.0 #[s]

    # Step1: Prediction step
    x_predict, P_predict = predict(x, P, u, dt)

    print(f"Predicted position: {x_predict}")
    
    # Sensor measurement
    z = 5.05 #[m] from the ground

    # Step2: Update step
    x, P = update(x_predict, P_predict, z)

    print(f"Updated position: {x}")

if __name__ == "__main__":
    main()
