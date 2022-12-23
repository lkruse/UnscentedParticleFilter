struct UnscentedParticleFilter
    λ
    Q
    R
    f
    h
end

function unscented_transform(x̄, P, f, λ, ws)
    n = length(x̄)
    S = [x̄]

    if n > 1
        Δ = cholesky(Hermitian((n + λ) * P)).L
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

function update(upf, x, P, y, a)
    λ, Q, R, f, h = upf.λ, upf.Q, upf.R, upf.f, upf.h
    n = length(x)
    ws = [λ / (n + λ); fill(1/(2(n + λ)), 2n)]
    x̄p, Pp, 𝒳, 𝒳′ = my_unscented_transform(x, P, s -> f(s, a), λ, ws)
    Pp = Pp + Q
    ȳ, Pyy, 𝒴, 𝒴′ = my_unscented_transform(x̄p, Pp, s -> h(s), λ, ws)
    Pyy = Pyy + R
    Pxy = sum(w*(s - x̄p)*(s′ - ȳ)' for (w,s,s′) in zip(ws, 𝒴, 𝒴′))
    K = Pxy / Pyy
    x̄ = x̄p + K*(y - ȳ)
    P̂ = Pp - K*Pyy*K'

    return (x̄, P̂)
end

