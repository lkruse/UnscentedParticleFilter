function unscented_transform(x̄, P, f, λ, ws)
    n = length(x̄)
    S = [x̄]

    if n > 1
        Δ = cholesky((n + λ) * P).L
        for i in 1:n
            push!(S, x̄ + Δ[:, i])
            push!(S, x̄ - Δ[:, i])
        end
    else
        Δ = sqrt(P)
        push!(S, x̄ + Δ)
        push!(S, x̄ - Δ)
    end

    S′ = f.(S)
    x̄′ = sum(w*s for (w,s) in zip(ws, S′))
    P′ = sum(w*(s - x̄′)*(s - x̄′)' for (w,s) in zip(ws, S′))

    return (x̄′, P′, S, S′)
end

function my_f(x_previous, t)
    ϕ₁ = 0.5
    ω = 4*exp(1) - 2
    
    # Define helping parameters
    n_part = size(x_previous,2);

    x_current = 1 + sin(ω*π*t) + ϕ₁*x_previous
    return x_current
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