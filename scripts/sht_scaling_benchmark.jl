# sht_scaling_benchmark.jl
#
# Measure where SHTnsKit's *distributed* scalar transform beats the
# *serial-replicate* transform (every rank does the full transform on
# replicated data), i.e. the crossover that decides whether moving the SHT
# from the serial fraction into the parallel fraction actually strong-scales.
#
# Compares four variants of a scalar analysis+synthesis pair per resolution:
#   serial-planned   : SHTPlan analysis!/synthesis!  (single-thread, 0-alloc) — typical solver inner loop
#   serial-threaded  : cfg-form analysis!/synthesis!  (threaded m-loop, JULIA_NUM_THREADS) — shared-memory lever
#   dist-theta       : θ(latitude)-decomposed PencilArray  — the scaling-friendly distributed layout
#   dist-phi         : φ(longitude)-decomposed PencilArray — the DEFAULT Pencil((nlat,nlon),comm); anti-scaling footgun
#
# WHY both θ and φ: PencilArrays splits the LAST dim by default → φ. The φ path
# Allgathers the full longitude onto every rank and replicates the Legendre
# transform, so it does not strong-scale. θ-decomposition divides the Legendre
# work and only reduces the (lmax+1, mmax+1) spectral matrix. If an earlier
# benchmark used the default pencil, it measured the φ footgun, not the limit.
#
# USAGE
#   # 4 ranks, 1 thread each (pure MPI):
#   mpiexec -n 4 julia --project=. scripts/sht_scaling_benchmark.jl
#   # hybrid 2 ranks x 4 threads:
#   mpiexec -n 2 julia --project=. -t 4 scripts/sht_scaling_benchmark.jl
#   # threads-only scaling (1 rank, vary -t to see the no-communication lever):
#   julia --project=. -t 4 scripts/sht_scaling_benchmark.jl
#   # custom resolutions / iterations / layouts:
#   SHT_LMAX_LIST=64,128,256,512 SHT_ITERS=40 SHT_DECOMPS=theta,phi \
#     mpiexec -n 8 julia --project=. scripts/sht_scaling_benchmark.jl
#
# CLUSTER NOTE: run with the MPI you configured MPI.jl against (system MPI via
# MPIPreferences for multi-node). On-node uses shared-memory transport (fast),
# so the crossover lmax is LOWER on a single node than across a network — run
# this on the real interconnect to get your true crossover.

using MPI
MPI.Init()
using SHTnsKit
using PencilArrays
using PencilFFTs

const comm  = MPI.COMM_WORLD
const rank  = MPI.Comm_rank(comm)
const np    = MPI.Comm_size(comm)
const nthr  = Threads.nthreads()

# ---- config via ENV ----
parse_lmax_list(s) = parse.(Int, split(s, ','))
# SHTnsKit >= 1.2.10 fixed the Plm_row! overflow that NaN'd lmax >= 151 (verified
# finite + roundtrip ~1e-13 at lmax 192/256), so the cap is lifted. High lmax is
# exactly where the transpose-based distributed transform wins, so the default now
# spans into that regime. The correctness gate below still flags any non-finite output.
const LMAXES  = parse_lmax_list(get(ENV, "SHT_LMAX_LIST", "32,64,128,256,384"))
const ITERS   = parse(Int, get(ENV, "SHT_ITERS", "30"))
const DECOMPS = Set(split(get(ENV, "SHT_DECOMPS", "theta,phi"), ','))
const CHECK   = get(ENV, "SHT_CHECK", "1") == "1"

# Collective timer: all ranks call Barrier + Allreduce; returns slowest-rank min time (s).
function timed(f; n=ITERS)
    f()                                   # warmup / compile
    best = Inf
    for _ in 1:n
        MPI.Barrier(comm)
        t = @elapsed f()
        best = min(best, MPI.Allreduce(t, MPI.MAX, comm))
    end
    return best
end

# Fill a PencilArray's local block from a full (nlat,nlon) field using global ranges.
function fill_from!(pa, pen, full)
    r = PencilArrays.range_local(pen)
    p = parent(pa)
    @inbounds for (jl, jg) in enumerate(r[2]), (il, ig) in enumerate(r[1])
        p[il, jl] = full[ig, jg]
    end
    return pa
end

rank == 0 && begin
    println("="^78)
    println("SHTnsKit distributed scaling benchmark")
    println("  ranks=$np  threads/rank=$nthr  host=$(gethostname())")
    try
        b = MPI.Get_library_version()
        println("  MPI: ", first(split(b, '\n')))
    catch; end
    println("  lmax list: ", LMAXES, "  iters=$ITERS  decomps=", collect(DECOMPS))
    any(>=(151), LMAXES) && println("  (lmax >= 151 needs SHTnsKit >= 1.2.10 for the Plm_row! ",
                                    "overflow fix; older versions NaN — the correctness gate flags it.)")
    println("="^78)
    println(rpad("lmax",6), rpad("ser-planned",13), rpad("ser-thread",12),
            rpad("dist-θ",11), rpad("θ-speedup",11),
            rpad("dist-φ",11), rpad("φ-speedup",11), "θ max-err")
end

# crossover tracking (θ beats serial-planned)
theta_cross = -1

for lmax in LMAXES
    nlat = lmax + 2
    nlon = 2 * lmax + 1
    cfg  = create_gauss_config(lmax, nlat; nlon=nlon)

    # band-limited reference field. Use a decaying spectrum (∝ 1/(1+l)) so the
    # field is physical and the synthesis stays well-conditioned at high lmax
    # (a flat spectrum overflows Nlm·P_l^m intermediates around lmax≳150).
    alm0 = zeros(ComplexF64, lmax + 1, lmax + 1)
    for m in 0:lmax, l in m:lmax
        s = 1.0 / (1 + l)
        alm0[l+1, m+1] = m == 0 ? complex(s) : complex(s, 0.5s)
    end
    f_full = SHTnsKit.synthesis(cfg, alm0; real_output=true)

    # ---- serial-planned (0-alloc, single-thread) ----
    plan = SHTPlan(cfg)
    almp = zeros(ComplexF64, lmax + 1, lmax + 1)
    foutp = similar(f_full)
    t_planned = timed() do
        SHTnsKit.analysis!(plan, almp, f_full)
        SHTnsKit.synthesis!(plan, foutp, almp)
    end

    # ---- serial-threaded (cfg-form, threaded m-loop, low-alloc via scratch) ----
    scr = Matrix{ComplexF64}(undef, nlat, nlon)
    almc = zeros(ComplexF64, lmax + 1, lmax + 1)
    foutc = similar(f_full)
    t_thread = timed() do
        SHTnsKit.analysis!(cfg, almc, f_full; fft_scratch=scr)
        SHTnsKit.synthesis!(cfg, foutc, almc; fft_scratch=scr)
    end

    # serial reference coefficients (for the distributed correctness gate)
    alm_ref = SHTnsKit.analysis(cfg, f_full)

    # ---- distributed θ-decomposition ----
    t_theta = NaN; err_theta = NaN
    if "theta" in DECOMPS
        pen_t = Pencil((nlat, nlon), (1,), comm)         # split dim 1 = latitude
        f_t = PencilArray(pen_t, zeros(Float64, PencilArrays.size_local(pen_t)...))
        fill_from!(f_t, pen_t, f_full)
        alm_t = SHTnsKit.dist_analysis(cfg, f_t)
        CHECK && (err_theta = maximum(abs.(alm_t .- alm_ref)))
        t_theta = timed() do
            a = SHTnsKit.dist_analysis(cfg, f_t)
            SHTnsKit.dist_synthesis(cfg, a; prototype_θφ=f_t, real_output=true)
        end
    end

    # ---- distributed φ-decomposition (default pencil; footgun) ----
    t_phi = NaN
    if "phi" in DECOMPS
        pen_p = Pencil((nlat, nlon), comm)               # default: splits last dim = longitude
        f_p = PencilArray(pen_p, zeros(Float64, PencilArrays.size_local(pen_p)...))
        fill_from!(f_p, pen_p, f_full)
        t_phi = timed() do
            a = SHTnsKit.dist_analysis(cfg, f_p)
            SHTnsKit.dist_synthesis(cfg, a; prototype_θφ=f_p, real_output=true)
        end
    end

    sp_theta = isnan(t_theta) ? NaN : t_planned / t_theta
    sp_phi   = isnan(t_phi)   ? NaN : t_planned / t_phi
    if theta_cross < 0 && !isnan(sp_theta) && sp_theta > 1.0
        global theta_cross = lmax
    end

    if rank == 0
        ms(x) = isnan(x) ? "  -  " : string(round(x * 1e3, digits=3))
        sx(x) = isnan(x) ? "  -  " : string(round(x, digits=2), "x")
        println(rpad(lmax,6), rpad(ms(t_planned),13), rpad(ms(t_thread),12),
                rpad(ms(t_theta),11), rpad(sx(sp_theta),11),
                rpad(ms(t_phi),11), rpad(sx(sp_phi),11),
                CHECK ? string(round(err_theta, sigdigits=2)) : "skip")
    end
end

if rank == 0
    println("="^78)
    if np == 1
        println("np=1: distributed columns are the single-rank overhead baseline.")
        println("Thread lever: compare ser-planned (serial) vs ser-thread (",
                nthr, " threads) — that speedup needs NO communication.")
    else
        if theta_cross > 0
            println("θ-decomposition beats serial-replicate from lmax ≈ ", theta_cross,
                    " upward (at $np ranks, this interconnect).")
        else
            println("θ-decomposition did not beat serial-replicate at any tested lmax — ",
                    "raise SHT_LMAX_LIST or check interconnect.")
        end
        println("If dist-φ << dist-θ in speedup, the default Pencil((nlat,nlon),comm) ",
                "footgun is why earlier runs looked like 'no scaling'.")
    end
    println("Hybrid tip: fewer ranks x more threads moves the transform into the ",
            "parallel fraction with zero communication below the distributed crossover.")
    println("="^78)
end

MPI.Finalize()
