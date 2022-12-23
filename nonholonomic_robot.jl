#=
Author: 
    Liam Kruse
Email: 
    lkruse@stanford.edu
nonholonomic_robot.jl:
    A script for estimating the position of a nonholonomic robot with the 
    Unscented Particle Filter, as presented in 
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
using Random
using StatsBase

include("upf.jl")

##
#*******************************************************************************
# FUNCTIONS
#*******************************************************************************
# Transition model
function f(s, u)
    v, ϕ = u
    Δt = 0.1; L= 2.8
    s′ = s + Δt*([v*cos(s[3]); v*sin(s[3]); v*tan(ϕ)/L])
    return s′
end

# Measurement model
function h(s)
    y = [s[1], s[2]]
end

# Function to run a simulation starting at state x0 and using the transition 
# model f and measurement model h specified by the filter
function simulate(upf, T, x0, actions)   
    Q, R, f, h = upf.Q, upf.R, upf.f, upf.h
    states =[x0]; observations = [[NaN, NaN]]
    n = length(x0)
    for t = 2:T
        push!(states, f(states[t-1], actions[t]) + rand(MvNormal(zeros(n), Q)))
        push!(observations, h(states[t]) + rand(MvNormal(zeros(2), R)))
    end
    return states, observations
end

##
#*******************************************************************************
# SIMULATION SETUP
#*******************************************************************************
T   = 30            # Simulation length
N   = 50           # Number of particles
Δt = 0.1
times = 0:Δt:T
action_sequence = [[6.0, 0.005*sin(0.5*t)] for t in times]

x0 = 0.0            # Initial value for x-coordinate [m]
y0 = 0.0            # Initial value for y-coordinate [m]
θ0 = 0.0            # Initial value for heading angle [rad]

# Noise statistics
Q = Δt*0.05*Matrix{Float64}(I,3,3)
R = 0.1*Matrix{Float64}(I,2,2)

upf = UnscentedParticleFilter(2, Q, R, f, h) # Unscented particle filter

s0  = [x0; y0; θ0]
x   = fill(s0, N)   # Initial particle set values
P0  = 0.01*Matrix{Float64}(I,3,3)
P   = fill(P0, N)    # Initial particle set covariances

x̂ = fill(s0, N)     # Data structure to hold sampled particles
x̄ = fill(s0, N)     # Data structure to hold posterior means
P̂ = fill(P0, N)     # Data structure to hold posterior covariances

μ = [mean(x)]       # Vector to hold mean data for plotting
Σ = [cov(x)]        # Vector to hold covariance data for plotting

# Generate simulated data
states, observations = simulate(upf, lastindex(times), s0, action_sequence)

##
#*******************************************************************************
# FILTERING
#*******************************************************************************
for t = 2:lastindex(times)
    Q, R = upf.Q, upf.R # Noise statistics
    y = observations[t] # Latest observation     
    for i = 1:N
        # Perform a belief update
        x̄[i], P̂[i] = my_update(upf, x[i], P[i], y, action_sequence[t])
        # Draw a new particle from the proposal distribution
        x̂[i] = rand(MvNormal(x̄[i], Hermitian(P̂[i])))
    end

    # Evaluate importance weights up to a normalizing constant
    w = ones(N)/N;
    for i = 1:N
        ŷ = h(x̂[i])  # Predicted measurement

        # Compute observation likelihood
        likelihood = pdf(MvNormal(ŷ, R), y)
        # Calculate prior
        prior = pdf(MvNormal(x[i], Q), x̂[i])
        # Calculate proposal
        proposal = pdf(MvNormal(x̄[i], Hermitian(P̂[i])), x̂[i])
        
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
x_data = [s[1] for s in states]; y_data = [s[2] for s in states]
μ_x = [mu[1] for mu in μ]; μ_y = [mu[2] for mu in μ]

p = Axis([
    Plots.Linear(μ_x, μ_y, legendentry="UPF Estimate",
        style="blue, thick, mark options={scale=0.6, fill=blue, solid}"), 
    Plots.Linear(x_data, y_data, legendentry="True State Value",
        style="black, dashed, thick, no marks"),
],
style="enlarge x limits=false,grid=both",
ylabel="y", xlabel="time",
title="UPF State Estimation",
        legendPos = "north east",legendStyle="nodes = {scale = 0.75}")
save("nonholonomic_robot.pdf", p)