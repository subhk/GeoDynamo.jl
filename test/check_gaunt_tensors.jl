using Random
using Printf

using bcs.Topography

function gradient_gaunt_numeric(l1::Int, m1::Int, l2::Int, m2::Int, L::Int, M::Int,
                                cache::GauntTensorCache{T}) where T
    if m1 != m2 + M
        return zero(T)
    end
    if abs(l1 - l2) > L || l1 + l2 < L
        return zero(T)
    end
    if isodd(l1 + l2 + L)
        return zero(T)
    end
    if l2 == 0 || L == 0
        return zero(T)
    end

    Y1 = evaluate_spherical_harmonics_grid(l1, m1, cache)
    grad2_theta, grad2_phi = evaluate_spherical_harmonic_gradient_grid(l2, m2, cache)
    gradL_theta, gradL_phi = evaluate_spherical_harmonic_gradient_grid(L, M, cache)

    nlat, nlon = cache.nlat, cache.nlon
    dphi = 2π / nlon
    weights = cache.weights

    result = zero(Complex{T})
    for i in 1:nlat
        w = weights[i]
        for j in 1:nlon
            dot_grad = grad2_theta[i, j] * gradL_theta[i, j] + grad2_phi[i, j] * gradL_phi[i, j]
            result += w * dphi * conj(Y1[i, j]) * dot_grad
        end
    end

    return T(real(result))
end

function random_mode(lmax::Int, lmax_t::Int)
    while true
        l1 = rand(0:lmax)
        l2 = rand(0:lmax)
        L = rand(0:lmax_t)
        m1 = rand(-l1:l1)
        m2 = rand(-l2:l2)
        M = m1 - m2

        if abs(M) > L
            continue
        end
        if abs(l1 - l2) > L || l1 + l2 < L
            continue
        end
        if isodd(l1 + l2 + L)
            continue
        end

        return l1, m1, l2, m2, L, M
    end
end

function main()
    lmax = 6
    lmax_t = 6
    ntest = 200
    seed = 1234

    Random.seed!(seed)
    cache = GauntTensorCache(lmax, lmax_t)

    max_abs = 0.0
    max_rel = 0.0
    worst = nothing

    max_abs_grad = 0.0
    max_rel_grad = 0.0
    worst_grad = nothing

    for _ in 1:ntest
        l1, m1, l2, m2, L, M = random_mode(lmax, lmax_t)

        G_wigner = gaunt_on_the_fly(l1, m1, l2, m2, L, M; use_wigner=true)
        G_num = compute_gaunt_tensor(l1, m1, l2, m2, L, M, cache)

        abs_err = abs(G_num - G_wigner)
        rel_err = abs_err / (abs(G_wigner) + 1e-14)
        if abs_err > max_abs
            max_abs = abs_err
            max_rel = rel_err
            worst = (l1, m1, l2, m2, L, M, G_wigner, G_num)
        end

        G_grad_num = gradient_gaunt_numeric(l1, m1, l2, m2, L, M, cache)
        G_grad_ana = gradient_gaunt_from_basic(l1, l2, L, G_wigner)
        abs_err_grad = abs(G_grad_num - G_grad_ana)
        rel_err_grad = abs_err_grad / (abs(G_grad_ana) + 1e-14)
        if abs_err_grad > max_abs_grad
            max_abs_grad = abs_err_grad
            max_rel_grad = rel_err_grad
            worst_grad = (l1, m1, l2, m2, L, M, G_grad_ana, G_grad_num)
        end
    end

    println("Gaunt tensor sanity check")
    println("==========================")
    @printf("samples: %d  lmax=%d  lmax_topo=%d  seed=%d\n", ntest, lmax, lmax_t, seed)
    @printf("max |G_num - G_wigner| = %.3e  (rel %.3e)\n", max_abs, max_rel)
    if worst !== nothing
        l1, m1, l2, m2, L, M, Gw, Gn = worst
        @printf("worst G: l1=%d m1=%d l2=%d m2=%d L=%d M=%d  Gw=%.6e  Gn=%.6e\n",
                l1, m1, l2, m2, L, M, Gw, Gn)
    end

    @printf("max |G_grad_num - G_grad_ana| = %.3e  (rel %.3e)\n", max_abs_grad, max_rel_grad)
    if worst_grad !== nothing
        l1, m1, l2, m2, L, M, Ga, Gn = worst_grad
        @printf("worst G_grad: l1=%d m1=%d l2=%d m2=%d L=%d M=%d  Ga=%.6e  Gn=%.6e\n",
                l1, m1, l2, m2, L, M, Ga, Gn)
    end
end

main()
