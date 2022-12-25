struct UnscentedKalmanFilter
    λ   # Spread parameter
    Q   # Process noise covariance matrix
    R   # Measurement noise covariance matrix
    f   # Transition model
    h   # Measurement model
end

function unscented_transform(x̄, P, func, λ, ws)
    n = length(x̄)
    𝒳 = [x̄]
    if n > 1
        Δ = cholesky(Hermitian((n + λ) * P)).L
        for i in 1:n
            push!(𝒳, x̄ + Δ[:, i])
            push!(𝒳, x̄ - Δ[:, i])
        end
    else
        Δ = sqrt(P)
        push!(𝒳, x̄ + Δ)
        push!(𝒳, x̄ - Δ)
    end
    𝒳′  = func.(𝒳)
    x̄′ = sum(w*s for (w,s) in zip(ws, 𝒳′))
    P′ = sum(w*(s - x̄′)*(s - x̄′)' for (w,s) in zip(ws, 𝒳′))

    return (x̄′, P′, 𝒳, 𝒳′)
end

function update(upf, x, P, y, a)
    λ, Q, R, f, h = upf.λ, upf.Q, upf.R, upf.f, upf.h
    n = length(x)
    ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]
    x̄p, Pp, 𝒳, 𝒳′ = unscented_transform(x, P, s -> f(s, a), λ, ws)
    Pp = Pp + Q
    ȳ, Pyy, 𝒴, 𝒴′ = unscented_transform(x̄p, Pp, s -> h(s, a), λ, ws)
    Pyy = Pyy + R
    Pxy = sum(w*(s - x̄p)*(s′ - ȳ)' for (w,s,s′) in zip(ws, 𝒴, 𝒴′))
    K = Pxy / Pyy
    x̄ = x̄p + K*(y - ȳ)
    P̂ = Pp - K*Pyy*K'

    return (x̄, P̂)
end

