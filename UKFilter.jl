function UKFilter(x_est, P_est, Q, y_true, R, t, α, β, κ)
    states = size(x_est, 1)
    observations = size(y_true, 1)
    vNoise = size(Q,2)
    wNoise = size(R,2)

    noises = vNoise + wNoise

    if !isnothing(noises)
        N = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
        P_a = [P_est zeros(states, noises); zeros(noises, states) N]
        x_mean_a = [x_est; zeros(noises,1)]
    else
        P_a = P_est
        x_mean_a = x_est
    end
    #return P_a, x_mean_a

    # Calculate the sigma points and the sigma_weights
    sigma_points, sigma_weights, number_of_sigma_points = getSigmaPoints(x_mean_a, P_a, α, β, κ);

    W_x = repeat(sigma_weights[:,2:number_of_sigma_points],states,1);
    W_y = repeat(sigma_weights[:,2:number_of_sigma_points],observations,1);

    x_sigma_points = f(sigma_points[1:states,:], sigma_points[states+1:states+vNoise,:], t);
    y_sigma_points = h(x_sigma_points, sigma_points[states+vNoise+1:states+noises,:], t);


    ############################################################################
    x_mean_pred = sum(W_x .* (x_sigma_points[:,2:number_of_sigma_points] 
        - repeat(x_sigma_points[:,1],1,number_of_sigma_points-1)),dims = 2);
    y_mean_pred = sum(W_y .* (y_sigma_points[:,2:number_of_sigma_points] 
        - repeat(y_sigma_points[:,1],1,number_of_sigma_points-1)),dims = 2);

    x_mean_pred = x_mean_pred + x_sigma_points[:,1];
    y_mean_pred = y_mean_pred + y_sigma_points[:,1];

    x_diff = x_sigma_points[:,1] - x_mean_pred;
    y_diff = y_sigma_points[:,1] - y_mean_pred;

    P_pred = sigma_weights[number_of_sigma_points+1]*x_diff*Matrix(x_diff');
    P_xy = sigma_weights[number_of_sigma_points+1]*x_diff*Matrix(y_diff');
    P_yy = sigma_weights[number_of_sigma_points+1]*y_diff*Matrix(y_diff');

    x_diff = x_sigma_points[:,2:number_of_sigma_points] - repeat(x_mean_pred,1,number_of_sigma_points-1);
    y_diff = y_sigma_points[:,2:number_of_sigma_points] - repeat(y_mean_pred,1,number_of_sigma_points-1);

    P_pred = P_pred + (W_x .* x_diff) * Matrix(x_diff');
    P_yy = P_yy .+ (W_y .* y_diff) * Matrix(y_diff');
    P_xy = P_xy .+ x_diff * Matrix((W_y .* y_diff)');

    K = P_xy / P_yy;

    x_est = x_mean_pred + K*( y_true - y_mean_pred[1]);

    P_est = P_pred - K*P_yy*Matrix(K');

    #return sigma_points, sigma_weights, number_of_sigma_points
    return x_est[1], P_est[1]
    #return sqrt_matrix
end

function getSigmaPoints(x_mean_previous_a, P_previous_a, alpha, beta, kappa)    
    n_x_mean_a = size(x_mean_previous_a,1); 
    number_of_points = n_x_mean_a*2 + 1;
    
    lambda = alpha^2*(n_x_mean_a + kappa)-n_x_mean_a; 
    
    sigma_points = zeros(n_x_mean_a, number_of_points);
    sigma_weights = zeros(1, number_of_points);
    
    sqrt_matrix = Matrix(Matrix(sparse(cholesky((n_x_mean_a+lambda)*P_previous_a).U))');
    # Define the sigma_points columns
    sigma_points = [zeros(size(P_previous_a,1),1) -sqrt_matrix sqrt_matrix];
    # Add mean to the rows
    sigma_points = sigma_points + repeat(x_mean_previous_a, 1, number_of_points);
    
    # Define the sigma_weights columns 
    sigma_weights = [lambda 0.5*ones(1,(number_of_points-1)) 0] / (n_x_mean_a+lambda);
    
    sigma_weights[number_of_points+1] = sigma_weights[1] + (1-(alpha^2)+beta); 
    
    return sigma_points, sigma_weights, number_of_points
    #return sqrt_matrix
end
    

function f(x_previous, v_previous, t_previous)
    x_current = 0.0
    omega = 4*exp(1)-2;
    phi1 = 0.5;
    
    # Define helping parameters
    n_part = size(x_previous,2);
    sin_term = sin(omega*pi*t_previous);
    
    x_current = ones(1,n_part) + repeat([sin_term],1,n_part) + phi1.*x_previous + v_previous;
    return x_current
end
    
function h( x_value, n, t)
    y_predict = 0.0
    if t <= 30
        y_predict = 0.2 * x_value.^(2) + n;
    else
        y_predict = 0.5 * x_value .- 2 + n;
    end
    return y_predict
end

function predictY( x_value, t)
    y_predict = 0.0
    if t <= 30
        y_predict = 0.2 * x_value.^(2);
    else
        y_predict = 0.5 * x_value .- 2; 
    end
    return y_predict
end
    
    