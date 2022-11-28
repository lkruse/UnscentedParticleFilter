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
    λ = 2
    n = length(x_est)
    weights = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

    #μp, Σp, Sp, Sp′ = unscented_transform(μb, Σb, s->fT(s,a), λ, weights)
    x̄p, Pp, Sp, Sp′ = unscented_transform(x_mean_a, P_a, s -> f(s[1:states,:], s[states+1:states+vNoise,:], t), λ, weights)
    Pp = Pp + [Q]
    x̄o, Po, So, So′ = unscented_transform(x̄p, Pp, s -> my_h(s[1:states,:], t), λ, weights)

    #h(𝒳, S[states+vNoise+1:states+noises,:], t)
    # Calculate the sigma points and the sigma_weights
    S, ws, n = getSigmaPoints(x_mean_a, P_a, α, β, κ);

    W_x = repeat(ws[:,2:n],states,1);
    W_y = repeat(ws[:,2:n],observations,1);

    𝒳 = f(S[1:states,:], S[states+1:states+vNoise,:], t);
    𝒴 = h(𝒳, S[states+vNoise+1:states+noises,:], t);

    ############################################################################
    x̄ₜ₋₁ = sum(W_x .* (𝒳[:,2:n] 
        - repeat(𝒳[:,1],1,n-1)),dims = 2);
    ȳₜ₋₁ = sum(W_y .* (𝒴[:,2:n] 
        - repeat(𝒴[:,1],1,n-1)),dims = 2);

    x̄ₜ₋₁ = x̄ₜ₋₁ + 𝒳[:,1];
    ȳₜ₋₁ = ȳₜ₋₁ + 𝒴[:,1];

    x_diff = 𝒳[:,1] - x̄ₜ₋₁;
    y_diff = 𝒴[:,1] - ȳₜ₋₁;

    P_pred = ws[n+1]*x_diff*Matrix(x_diff');
    P_xy = ws[n+1]*x_diff*Matrix(y_diff');
    P_yy = ws[n+1]*y_diff*Matrix(y_diff');
    #P_pred = (W_x .* x_diff) * Matrix(x_diff');
    #P_xy = (W_x .* x_diff) * Matrix(y_diff');
    #P_yy = (W_x .* y_diff) * Matrix(y_diff');

    x_diff = 𝒳[:,2:n] - repeat(x̄ₜ₋₁,1,n-1);
    y_diff = 𝒴[:,2:n] - repeat(ȳₜ₋₁,1,n-1);

    P_pred = P_pred + (W_x .* x_diff) * Matrix(x_diff');
    P_yy = P_yy .+ (W_y .* y_diff) * Matrix(y_diff');
    P_xy = P_xy .+ x_diff * Matrix((W_y .* y_diff)');

    K = P_xy / P_yy;

    x_est = x̄ₜ₋₁ + K*( y_true - ȳₜ₋₁[1]);

    P_est = P_pred - K*P_yy*Matrix(K');

    return x_est[1], P_est[1]
end

function getSigmaPoints(x_mean_previous_a, P_previous_a, alpha, beta, kappa) 
    
    n_x_mean_a = size(x_mean_previous_a,1); 

    n = n_x_mean_a*2 + 1;
    
    #lambda = alpha^2*(n_x_mean_a + kappa)-n_x_mean_a; 
    lambda = 2

    λ = lambda; #n = n_x_mean_a
    
    S = zeros(n_x_mean_a, n);
    ws = zeros(1, n);
    
    sqrt_matrix = Matrix(Matrix(sparse(cholesky((n_x_mean_a+lambda)*P_previous_a).U))');
    # Define the sigma_points columns
    S = [zeros(size(P_previous_a,1),1) -sqrt_matrix sqrt_matrix];
    # Add mean to the rows
    S = S + repeat(x_mean_previous_a, 1, n);
    
    # Define the sigma_weights columns 
    ws = [lambda 0.5*ones(1,(n-1)) 0] / (n_x_mean_a+lambda);
    
    ws[n+1] = ws[1] + (1-(alpha^2)+beta); 
    #ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

    return S, ws, n
    #return sqrt_matrix
end
    
function unscented_transform(x̄, P, f, λ, ws)
    n = length(x̄)
    Δ = cholesky((n + λ) * P).L
    S = [x̄]
    for i in 1:n
        push!(S, x̄ + Δ[:, i])
        push!(S, x̄ - Δ[:, i])
    end
    S′ = f.(S)
    x̄′ = sum(w*s for (w,s) in zip(ws, S′))
    P′ = sum(w*(s - x̄′)*(s - x̄′)' for (w,s) in zip(ws, S′))

    return (x̄′, P′, S, S′)
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

function my_h( x_value, t)
    y_predict = 0.0
    if t <= 30
        y_predict = 0.2 * x_value[1].^(2);
    else
        y_predict = 0.5 * x_value[1] .- 2;
    end
    return y_predict
end

function my_f(x_previous, t_previous)
    x_current = 0.0
    omega = 4*exp(1)-2;
    phi1 = 0.5;
    
    # Define helping parameters
    n_part = size(x_previous,2);
    sin_term = sin(omega*pi*t_previous);
    
    #x_current = ones(1,n_part) + repeat([sin_term],1,n_part) + phi1.*x_previous
    x_current = 1 + sin_term + phi1.*x_previous[1]
    return [x_current; 0.0; 0.0]
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
    

##
#=
function safe_cholesky(A)
    if dim(A) > 1
        Δ = cholesky(A).L
    else
        Δ = sqrt(A)
    end
    return Δ
end
=#

safe_cholesky(A) = dim(A) > 1 ? Δ = cholesky(A).L : Δ = sqrt(A)