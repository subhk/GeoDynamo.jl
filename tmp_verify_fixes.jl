using GeoDynamo

println("=== #1: precomputed theta-gradient neighbors must equal mode_index ===")
cfg = GeoDynamo.create_shtnskit_config(lmax=6, mmax=4, nlat=16, nlon=16, nr=8, optimize_decomp=false)
plus, minus = GeoDynamo.solver_build_theta_gradient_neighbors(cfg)
@assert length(plus) == cfg.nlm && length(minus) == cfg.nlm
n1_ok = true
for i in 1:cfg.nlm
    l = cfg.l_values[i]; m = cfg.m_values[i]
    ep = GeoDynamo.mode_index(cfg, l + 1, m)
    em = GeoDynamo.mode_index(cfg, l - 1, m)
    if plus[i] != ep || minus[i] != em
        global n1_ok = false
        println("  MISMATCH i=$i (l=$l,m=$m): plus $(plus[i]) vs $ep ; minus $(minus[i]) vs $em")
    end
end
println(n1_ok ? "  NEIGHBOR_CHECK_OK ($(cfg.nlm) modes)" : "  NEIGHBOR_CHECK_FAIL")

println("=== #2: ERK2 BC spec cache: reuse + matches fresh build ===")
dom = GeoDynamo.create_radial_domain(8)
caches = GeoDynamo.TimestepCaches{Float64}()
fresh = GeoDynamo.build_solver_erk2_scalar_bc(Float64, dom, 1)
got1 = GeoDynamo._get_or_build_erk2_boundary_spec!(caches, :temperature, 1,
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, dom, 1))
# Second call must return the cached object and must NOT invoke the builder.
rebuilt = false
got2 = GeoDynamo._get_or_build_erk2_boundary_spec!(caches, :temperature, 1,
            () -> (global rebuilt = true; GeoDynamo.build_solver_erk2_scalar_bc(Float64, dom, 1)))
println((got1 === got2 && !rebuilt) ? "  BC_CACHE_REUSE_OK (no rebuild on 2nd call)" : "  BC_CACHE_REUSE_FAIL (rebuilt=$rebuilt)")
matches = fresh.inner.type == got1.inner.type && fresh.outer.type == got1.outer.type &&
          fresh.inner.stencil == got1.inner.stencil && fresh.outer.stencil == got1.outer.stencil &&
          fresh.inner.r_inv == got1.inner.r_inv && fresh.outer.r_inv == got1.outer.r_inv
println(matches ? "  BC_CACHE_MATCHES_FRESH_OK" : "  BC_CACHE_MATCHES_FRESH_FAIL")
# Different bc_code must produce a distinct entry.
got_nn = GeoDynamo._get_or_build_erk2_boundary_spec!(caches, :temperature, 4,
            () -> GeoDynamo.build_solver_erk2_scalar_bc(Float64, dom, 4))
println((got_nn !== got1 && length(caches.erk2_boundary_specs) == 2) ? "  BC_CACHE_KEYING_OK (2 distinct entries)" : "  BC_CACHE_KEYING_FAIL")

allok = n1_ok && (got1 === got2 && !rebuilt) && matches && (got_nn !== got1)
println(allok ? "VERIFY_ALL_OK" : "VERIFY_FAILED")
