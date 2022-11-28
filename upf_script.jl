##
using Distributions
using LinearAlgebra
using PGFPlots
using SparseArrays
using StatsBase
include("UKFilter.jl")

##
function generateData(T, x₀)
    ϕ₁ = 0.5
    ϕ₂ = 0.2
    ϕ₃ = 0.5
    
    xData =[x₀]; yData = []; 

    for t = 2:T
        push!(xData, generateX(t, xData[t-1]))
        push!(yData, generateY(t, xData[t-1]))
    end
    return xData, yData
end

α  = 3
θ  = 2

sigma = 2

function generateX(t, xₜ)
    ϕ₁ = 0.5
    ω = 4*exp(1) - 2
    return 1 + sin(ω*π*t) + ϕ₁*xₜ + rand(Normal(0, sigma))#+ rand(Gamma(α,θ))
end

function generateY(t, xₜ)
    ϕ₂ = 0.2
    ϕ₃ = 0.5
    σ = 1e-5
    if t <= 30
        return ϕ₂*xₜ^2 + rand(Normal(0, σ))
    else
        return ϕ₃*xₜ - 2 + rand(Normal(0, σ))
    end
end


##
σ       = 1e-5
T       = 60
P₀      = α/(θ^2)
N       = 200
Q       = 2*(0.75)
R       = 0.1

# Sigma point parameters
α_ukf   = 1
β       = 0
κ       = 2

# Rename later?

x₀ = 1.0

xData, yData = generateData(T, x₀)

x = ones(T,N);
P = P₀ * ones(T,N);
# x_predict = ones(T,N);
x̂ = ones(N)

P_predict = ones(T,N);

#x_mean_predict = ones(T,N);
x̄ = ones(N)

y_predict = ones(T,N);
w = ones(T,N)./N;

𝒳 = []; 𝒴 = [];
sqrt_matrix = []
likelihood = 0; likelihood_exponent= 0; prior = 0; proposal = 0; prior_exponent = 0; prior_term=0;

#=
vNoise = size(Q,2)
wNoise = size(R,2)
states = size(x[1,1], 1)
N = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
noises = vNoise + wNoise

P_a = [P_predict[1,1] zeros(states, noises); zeros(noises, states) N]
x_mean_a = [x[1,1]; zeros(noises,1)]

getSigmaPoints(x_mean_a, P_a, α_ukf, β, κ)
S, sigma_weights, n = getSigmaPoints(x_mean_a, P_a, α_ukf, β, κ);

=#

states = 1#size(x_est, 1)
observations = 1#size(y_true, 1)
vNoise = size(Q,2)
wNoise = size(R,2)

noises = vNoise + wNoise

##
d = MvNormal([0.0], R)
for t = 2:T-1
    for i = 1:N
            
        myN = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
        P_a = [P[t-1,i] zeros(states, noises); zeros(noises, states) myN]
        x_mean_a = [x[t-1,i]; zeros(noises,1)]

        #return P_a, x_mean_a
        λ = 2
        n = length(x_mean_a)
        weights = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

        x̄p, Pp, Sp, Sp′ = unscented_transform(x_mean_a, P_a, s -> my_f(s, t), λ, weights)

        Pp[diagind(Pp)] = [Pp[1] + Q; Q; R]
        x̄o, Po, So, So′ = unscented_transform(x̄p, Pp, s -> my_h(s, t), λ, weights)

        Po = Po + R
        Ppo = sum(w*(s - x̄p)*(s′ - x̄o)' for (w,s,s′) in zip(weights, So, So′))

        K = Ppo / Po

        #x̄ = x̄p + K*(o - x̄o)
        #P = Pp - K*Po*K'
        o = yData[t]
        x_temp = x̄p + K*(o - x̄o)
        x̄[i] = x_temp[1]
        P_temp = Pp - K*Po*K'

        P_predict[t,i] = P_temp[1]

        #=
        x̄[i], P_predict[t,i] = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α_ukf, β, κ)
        =#
        x̂[i] = rand(Normal(x̄[i], sqrt(P_predict[t,i])))
    end

    # Evaluate importance weights up to a normalizing constant
    for i = 1:N
        y_predict[t, i] = predictY(x̂[i], t);

        likelihood_exponent = -0.5 * inv(σ) * ((yData[t] - y_predict[t, i])^2);
        likelihood = 1e-50 + inv(sqrt(σ)) * exp(likelihood_exponent);
        # Calculate prior
        #=
        prior_exponent = -θ*(x̂[i]) - x[t-1, i];
        prior_term = x̂[i] - x[t-1, i]^(α);
        prior = prior_term * exp(prior_exponent);
        # prior = (x_term^(k-1)*exp(-x_term/theta))/((theta^k)*gamma(k));
        =#
        #=
        prior_exponent = -θ*x̂[i] - x[t-1, i];
        prior_term = x̂[i] - x[t-1, i]^(α-1);
        prior = prior_term * exp(prior_exponent);
        =#
        prior_exponent = -0.5 * inv(sigma) * ((x̂[i] - x[t-1, i])^2);
        prior = inv(sqrt(sigma)) * exp(prior_exponent);

        #prior = prior / (2*(θ^α))

        # prior = (x_term^(k-1)*exp(-x_term/theta))/((theta^k)*gamma(k));
        # Calculate proposal
        proposal_term = inv(sqrt(P_predict[t, i]));
        proposal_exponent = -0.5 * inv(P_predict[t, i]) * ( x̂[i]-x̄[i] )^2;
        proposal = proposal_term * exp(proposal_exponent);
        
        # Assign a value to the weight
        w[t, i] = likelihood * prior / proposal;
        #w[t, i] = likelihood
        #w[t, i] = logpdf(d,[yData[t] - y_predict[t, i]])
        #w[t, i] = 0.001*rand()
    end
    ## Normalize weights
    weightSum = sum(w[t, :]);
    w[t, :] = w[t, :] ./ weightSum;

    resampledPoints = sample(1:N, Weights(w[t,:]), N, replace=true)
    
    x[t, :] = x̂[resampledPoints];
    P[t, :] = P[t, resampledPoints];
end

estimated_x_means = mean(x,dims=2)

##
p = Axis([
    PGFPlots.Linear(1:T, estimated_x_means[:], 
            style="blue, thick,mark options={scale=0.6, fill=blue, solid}",legendentry="UPF Estimate"), 
            PGFPlots.Linear(1:T, xData, 
                 style="black, dashed, thick, mark options={scale=0.6,fill=black, solid}", legendentry="True State Value"),

],
style="enlarge x limits=false,grid=both",
ylabel="y", xlabel="time",
title="UPF State Estimation",
        legendPos = "north east",legendStyle="nodes = {scale = 0.75}")
save("upf.pdf", p)

##
vNoise = size(Q,2)
wNoise = size(R,2)
states = size(x[10,1], 1)
observations = size(yData[10],1)

N = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
noises = vNoise + wNoise

P_a = [P_predict[9,1] zeros(states, noises); zeros(noises, states) N]
x_mean_a = [x[10,1]; zeros(noises,1)]

S, sigma_weights, n = getSigmaPoints(x_mean_a, P_a, α, β, κ);

##
W_x = repeat(sigma_weights[:,2:n],states,1);
W_y = repeat(sigma_weights[:,2:n],observations,1);

𝒳 = f(S[1:states,:], S[states+1:states+vNoise,:], 2);
𝒴 = h(𝒳, S[states+vNoise+1:states+noises,:], 2);

##
x̄ₜ₋₁ = sum(W_x .* (𝒳[:,2:n] - repeat(𝒳[:,1],1,n-1)),dims = 2);
ȳₜ₋₁ = sum(W_y .* (𝒴[:,2:n] - repeat(𝒴[:,1],1,n-1)),dims = 2);

x̄ₜ₋₁ = x̄ₜ₋₁ + 𝒳[:,1];
ȳₜ₋₁ = ȳₜ₋₁ + 𝒴[:,1];

x_diff = 𝒳[:,1] - x̄ₜ₋₁;
y_diff = 𝒴[:,1] - ȳₜ₋₁;

##
P_pred = sigma_weights[n+1]*x_diff*Matrix(x_diff');
P_xy = sigma_weights[n+1]*x_diff*Matrix(y_diff');
P_yy = sigma_weights[n+1]*y_diff*Matrix(y_diff');

##
x_diff = 𝒳[:,2:n] - repeat(x̄ₜ₋₁,1,n-1);
y_diff = 𝒴[:,2:n] - repeat(ȳₜ₋₁,1,n-1);

##
P_pred = P_pred + (W_x .* x_diff) * Matrix(x_diff');
P_yy = P_yy .+ (W_y .* y_diff) * Matrix(y_diff');
P_xy = P_xy .+ x_diff * Matrix((W_y .* y_diff)');

K = P_xy / P_yy;

x_est = x̄ₜ₋₁ + K*( yData[10] - ȳₜ₋₁[1]);

P_est = P_pred - K*P_yy*Matrix(K');

##
P_pred

##
my_diff = 𝒳 - repeat(x̄ₜ₋₁,1,n)
ws = sigma_weights[1:7]
my_P_pred = sum(w*(s - x̄ₜ₋₁[1])*(s - x̄ₜ₋₁[1])' for (w,s) in zip(ws, 𝒳))


##
P₀      = α/(θ^2)
N       = 200
Q       = 2*(0.75)
R       = 0.1

𝐱0 = ones(N)

x̄0 = [mean(𝐱0)]
P0 = cov(𝐱0)*I

N = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
P0 = [P₀ zeros(states, noises); zeros(noises, states) N]
x̄0 = [mean(𝐱0); zeros(noises,1)]

o = yData[1]

λ = 2
t = 1

##
n = 3

ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

x̄p, Pp, Sp, Sp′ = unscented_transform(x̄0, P0, s -> my_f(s, t), λ, ws)
#Pp[1] = Pp[1] + Q
Pp[diagind(Pp)] = [Pp[1] + Q; Q; R]
x̄o, Po, So, So′ = unscented_transform(x̄p, Pp, s -> my_h(s, t), λ, ws)
Po = Po + R

Ppo = sum(w*(s - x̄p)*(s′ - x̄o)' for (w,s,s′) in zip(ws, So, So′))

K = Ppo / Po

x̄ = x̄p + K*(o - x̄o)
P = Pp - K*Po*K'

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