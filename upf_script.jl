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

function generateX(t, xₜ)
    ϕ₁ = 0.5
    ω = 4*exp(1) - 2
    return 1 + sin(ω*π*t) + ϕ₁*xₜ + rand(Gamma(3,2))
end

function generateY(t, xₜ)
    ϕ₂ = 0.2
    ϕ₃ = 0.5
    σ = 1e-5
    if t <= 30
        return ϕ₂*xₜ^2 + rand(Normal(0, σ))
    else
        return ϕ₃*xₜ + rand(Normal(0, σ))
    end
end


##
σ       = 1e-5
T       = 60
P₀      = 0.75
N       = 200
Q       = 2*(0.75)
R       = 0.1

# Sigma point parameters
α       = 1
β       = 0
κ       = 2

# Rename later?
gamma1  = 3
gamma2  = 2

x₀ = 1.0

xData, yData = generateData(T, x₀)

x = ones(T,N);
P = P₀ * ones(T,N);
x_predict = ones(T,N);
P_predict = ones(T,N);
x_mean_predict = ones(T,N);
y_predict = ones(T,N);
w = ones(T,N)./N;

x_sigma_points = []; y_sigma_points = [];
sqrt_matrix = []
likelihood = 0;likelihood_exponent= 0;prior = 0; proposal = 0;prior_exponent = 0; prior_term=0;

##
for t = 2:T-1
    for i = 1:N
        x_mean_predict[t,i], P_predict[t,i] = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α, β, κ)
        #x_sigma_points, y_sigma_points = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α, β, κ)
        x_predict[t,i] = rand(Normal(x_mean_predict[t,i], sqrt(P_predict[t,i])))
    end

    # Evaluate importance weights up to a normalizing constant
    for i = 1:N
        y_predict[t, i] = predictY(x_predict[t, i], t);

        likelihood_exponent = -0.5 * inv(σ) * ((yData[t] - y_predict[t, i])^2);
        likelihood = 1e-50 + inv(sqrt(σ)) * exp(likelihood_exponent);

        # Calculate prior
        prior_exponent = -gamma2*(x_predict[t, i]) - x[t-1, i];
        prior_term = x_predict[t, i] - x[t-1, i]^(gamma1);
        prior = prior_term * exp(prior_exponent);

        # Calculate proposal
        proposal_term = inv(sqrt(P_predict[t, i]));
        proposal_exponent = -0.5 * inv(P_predict[t, i]) * ( x_predict[t, i]-x_mean_predict[t, i] )^2;
        proposal = proposal_term * exp(proposal_exponent);
        
        # Assign a value to the weight
        w[t, i] = likelihood * prior / proposal;
    end
    ## Normalize weights
    weightSum = sum(w[t, :]);
    w[t, :] = w[t, :] ./ weightSum;

    resampledPoints = sample(1:N, Weights(w[t,:]), N, replace=true)
    
    x[t, :] = x_predict[t, resampledPoints];
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
ymin=0, ylabel="y", xlabel="time",
title="UPF State Estimation",
        legendPos = "north east",legendStyle="nodes = {scale = 0.75}")
save("upf.pdf", p)


