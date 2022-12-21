#=
Author: 
    Liam Kruse
Email: 
    lkruse@stanford.edu
simulation.jl:
    A script for running a state estimation task with the Unscented Particle
    Filter, as presented in 
    "Van Der Merwe, R., Doucet, A., De Freitas, N., & Wan, E. (2000). The 
    Unscented Particle Filter. Advances in Neural Information Processing Systems, 13."
=#

##
#*******************************************************************************
# PACKAGES AND SETUP
#*******************************************************************************
using Distributions
using LinearAlgebra
using PGFPlots
using SparseArrays
using StatsBase

include("ukf.jl")

##
#*******************************************************************************
# FUNCTIONS
#*******************************************************************************
function simulate(T, x₀)    
    states =[x₀]; observations = []; 
    sigma = 1; Q = 1
    R = 0.00001^2
    for t = 2:T
        push!(states, my_f(states[t-1], t) + rand(Normal(0, sigma)))
        push!(observations, my_h(states[t-1], t) + rand(Normal(0, 0.00001)))
    end
    return states, observations
end

##
σ       = 1e-5
T       = 60
P₀      = 0.75
N       = 200
Q       = 2*(0.75)
R       = 0.1

# Sigma point parameters
α_ukf   = 1
β       = 0
κ       = 2

# Rename later?

x₀ = 1.0

xData, yData = simulate(T, x₀)

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


states = 1
observations = 1
vNoise = size(Q,2)
wNoise = size(R,2)

noises = vNoise + wNoise

##
d = MvNormal([0.0], R)
for t = 2:T-1
    for i = 1:N
            
        #myN = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
        #P_a = [P[t-1,i] zeros(states, noises); zeros(noises, states) myN]
        P_a = P[t-1,i]
        #x_mean_a = [x[t-1,i]; zeros(noises,1)]
        x_mean_a = x[t-1,i]

        #return P_a, x_mean_a
        λ = 2
        n = length(x_mean_a)
        weights = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]

        x̄p, Pp, Sp, Sp′ = unscented_transform(x_mean_a, P_a, s -> my_f(s, t), λ, weights)

        Pp = Pp + Q
        x̄o, Po, So, So′ = unscented_transform(x̄p, Pp, s -> my_h(s, t), λ, weights)

        Po = Po + R
        Ppo = sum(w*(s - x̄p)*(s′ - x̄o)' for (w,s,s′) in zip(weights, So, So′))

        K = Ppo / Po

        #x̄ = x̄p + K*(o - x̄o)
        #P = Pp - K*Po*K'
        o = yData[t]
        #x_temp = x̄p + K*(o - x̄o)
        #x̄[i] = x_temp[1]
        x̄[i] = x̄p + K*(o - x̄o)
        
        #P_temp = Pp - K*Po*K'

        #P_predict[t,i] = P_temp[1]
        P_predict[t,i] = Pp - K*Po*K'

        #=
        x̄[i], P_predict[t,i] = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α_ukf, β, κ)
        =#
        x̂[i] = rand(Normal(x̄[i], sqrt(P_predict[t,i])))
    end

    # Evaluate importance weights up to a normalizing constant
    for i = 1:N
        y_predict[t, i] = my_h(x̂[i], t);

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
        #sigma = 1
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