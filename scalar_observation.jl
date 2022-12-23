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
using PGFPlots
using StatsBase

include("upf.jl")

##
#*******************************************************************************
# FUNCTIONS
#*******************************************************************************
# Transition model
function f(x, t)
    ϕ1 = 0.5
    ω = 4*exp(1) - 2
    x′ = 1 + sin(ω*π*t) + ϕ1*x

    return x′
end

# Measurement model
function h(x, t)
    ϕ2 = 0.2; ϕ3 = 0.5
    if t <= 30
        y = ϕ2*x^2;
    else
        y = ϕ3*x - 2;
    end

    return y
end

# Function to run a simulation starting at state x0 and using the transition 
# model f and measurement model h specified by the filter
function simulate(upf, T, x0)   
    Q, R, f, h = upf.Q, upf.R, upf.f, upf.h
    states =[x0]; observations = [NaN]
    for t = 2:T
        push!(states, f(states[t-1], t) + rand(Normal(0.0, sqrt(Q))))
        push!(observations, h(states[t], t) + rand(Normal(0.0, sqrt(R))))
    end
    return states, observations
end

##
#*******************************************************************************
# SIMULATION SETUP
#*******************************************************************************
T   = 60            # Simulation length
N   = 200           # Number of particles

upf = UnscentedParticleFilter(2, 1.0, 0.001, f, h) # Unscented particle filter

x0  = 1.0
x   = fill(x0, N)   # Initial particle set values
P0  = 0.75
P   = P0*ones(N)    # Initial particle set covariances

x̂ = ones(N)         # Data structure to hold sampled particles
x̄ = ones(N)         # Data structure to hold posterior means
P̂ = ones(N)         # Data structure to hold posterior covariances

μ = [mean(x)]       # Vector to hold mean data for plotting
Σ = [cov(x)]        # Vector to hold covariance data for plotting

# Generate simulated data
states, observations = simulate(upf, T, x0)

##
#*******************************************************************************
# FILTERING
#*******************************************************************************
for t = 2:T
    Q, R = upf.Q, upf.R # Noise statistics
    y = observations[t] # Latest observation     
    for i = 1:N
        # Perform a belief update
        x̄[i], P̂[i] = update(upf, x[i], P[i], y, t)
        # Draw a new particle from the proposal distribution
        x̂[i] = rand(Normal(x̄[i], sqrt(P̂[i])))
    end

    # Evaluate importance weights up to a normalizing constant
    w = ones(N)/N;
    for i = 1:N
        ŷ = h(x̂[i], t)  # Predicted measurement

        # Compute observation likelihood
        likelihood = inv(sqrt(2π*R)) * exp(-0.5*inv(R)*((y - ŷ)^2))
        # Calculate prior
        prior = inv(sqrt(2π*Q))*exp(-0.5*inv(Q)*((x̂[i] - x[i])^2))
        # Calculate proposal
        proposal = inv(sqrt(2π*P̂[i]))*exp(-0.5*inv(P̂[i])*(x̂[i] - x̄[i])^2)
        
        # Compute the particle weight
        w[i] = likelihood * prior / proposal
    end
    # Normalize weights
    w = w/sum(w)

    # Resample particles
    resampled_idx = sample(1:N, Weights(w), N, replace=true)
    x = x̂[resampled_idx];
    P = P[resampled_idx];

    # Store plotting data
    push!(μ, mean(x)); push!(Σ, cov(x))
end

##
#*******************************************************************************
# PLOTTING
#*******************************************************************************
p = Axis([
    Plots.Linear(1:T, μ, legendentry="UPF Estimate",
        style="blue, thick, mark options={scale=0.6, fill=blue, solid}"), 
    Plots.Linear(1:T, states, legendentry="True State Value",
        style="black, dashed, thick, mark options={scale=0.6,fill=black, solid}"),
],
style="enlarge x limits=false,grid=both",
ylabel="y", xlabel="time",
title="UPF State Estimation",
        legendPos = "north east",legendStyle="nodes = {scale = 0.75}")
save("scalar_observation.pdf", p)
