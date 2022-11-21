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

gamma1  = 3
gamma2  = 2

function generateX(t, xₜ)
    ϕ₁ = 0.5
    ω = 4*exp(1) - 2
    return 1 + sin(ω*π*t) + ϕ₁*xₜ + rand(Gamma(gamma1,gamma2))
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

x_sigma_points = []; y_sigma_points = [];
sqrt_matrix = []
likelihood = 0;likelihood_exponent= 0;prior = 0; proposal = 0;prior_exponent = 0; prior_term=0;

#=
vNoise = size(Q,2)
wNoise = size(R,2)
states = size(x[1,1], 1)
N = [Q zeros(vNoise, wNoise); zeros(wNoise, vNoise) R]
noises = vNoise + wNoise

P_a = [P_predict[1,1] zeros(states, noises); zeros(noises, states) N]
x_mean_a = [x[1,1]; zeros(noises,1)]

getSigmaPoints(x_mean_a, P_a, α, β, κ)
sigma_points, sigma_weights, number_of_sigma_points = getSigmaPoints(x_mean_a, P_a, α, β, κ);

=#

##
d = MvNormal([0.0], R)
for t = 2:T-1
    for i = 1:N
        x̄[i], P_predict[t,i] = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α, β, κ)
        #x_sigma_points, y_sigma_points = UKFilter(x[t-1,i], P[t-1,i], Q, yData[t], R, t, α, β, κ)
        x̂[i] = rand(Normal(x̄[i], sqrt(P_predict[t,i])))
    end

    # Evaluate importance weights up to a normalizing constant
    for i = 1:N
        y_predict[t, i] = predictY(x̂[i], t);

        likelihood_exponent = -0.5 * inv(σ) * ((yData[t] - y_predict[t, i])^2);
        likelihood = 1e-50 + inv(sqrt(σ)) * exp(likelihood_exponent);

        # Calculate prior
        prior_exponent = -gamma2*(x̂[i]) - x[t-1, i];
        prior_term = x̂[i] - x[t-1, i]^(gamma1);
        prior = prior_term * exp(prior_exponent);

        # Calculate proposal
        proposal_term = inv(sqrt(P_predict[t, i]));
        proposal_exponent = -0.5 * inv(P_predict[t, i]) * ( x̂[i]-x̄[i] )^2;
        proposal = proposal_term * exp(proposal_exponent);
        
        # Assign a value to the weight
        w[t, i] = likelihood * prior / proposal;
        w[t, i] = logpdf(d,[yData[t] - y_predict[t, i]])
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
ymin=0, ylabel="y", xlabel="time",
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

getSigmaPoints(x_mean_a, P_a, α, β, κ)
sigma_points, sigma_weights, number_of_sigma_points = getSigmaPoints(x_mean_a, P_a, α, β, κ);

W_x = repeat(sigma_weights[:,2:number_of_sigma_points],states,1);
W_y = repeat(sigma_weights[:,2:number_of_sigma_points],observations,1);

x_sigma_points = f(sigma_points[1:states,:], sigma_points[states+1:states+vNoise,:], 2);
y_sigma_points = h(x_sigma_points, sigma_points[states+vNoise+1:states+noises,:], 2);

##
x_mean_pred = sum(W_x .* (x_sigma_points[:,2:number_of_sigma_points] 
- repeat(x_sigma_points[:,1],1,number_of_sigma_points-1)),dims = 2);
y_mean_pred = sum(W_y .* (y_sigma_points[:,2:number_of_sigma_points] 
- repeat(y_sigma_points[:,1],1,number_of_sigma_points-1)),dims = 2);

x_mean_pred = x_mean_pred + x_sigma_points[:,1];
y_mean_pred = y_mean_pred + y_sigma_points[:,1];

x_diff = x_sigma_points[:,1] - x_mean_pred;
y_diff = y_sigma_points[:,1] - y_mean_pred;

##
P_pred = sigma_weights[number_of_sigma_points+1]*x_diff*Matrix(x_diff');
P_xy = sigma_weights[number_of_sigma_points+1]*x_diff*Matrix(y_diff');
P_yy = sigma_weights[number_of_sigma_points+1]*y_diff*Matrix(y_diff');

##
x_diff = x_sigma_points[:,2:number_of_sigma_points] - repeat(x_mean_pred,1,number_of_sigma_points-1);
y_diff = y_sigma_points[:,2:number_of_sigma_points] - repeat(y_mean_pred,1,number_of_sigma_points-1);

##
P_pred = P_pred + (W_x .* x_diff) * Matrix(x_diff');
P_yy = P_yy .+ (W_y .* y_diff) * Matrix(y_diff');
P_xy = P_xy .+ x_diff * Matrix((W_y .* y_diff)');


##
P_pred

##
my_diff = x_sigma_points - repeat(x_mean_pred,1,number_of_sigma_points)
ws = sigma_weights[1:7]
my_P_pred = sum(w*(s - x_mean_pred[1])*(s - x_mean_pred[1]) for (w,s) in zip(ws, x_sigma_points))

##
#=
d = Gamma(3, 2)

i = 1; t = 2
my_x = x̂[i] - x[t-1, i]

pdf(d, my_x)
=#
