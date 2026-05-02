# SphericalShellGrid — spherical shell geometry (fluid between two concentric spheres)
struct SphericalShellGrid
    arch    :: AbstractArchitecture
    lmax    :: Int
    mmax    :: Int
    nlat    :: Int
    nlon    :: Int
    nr      :: Int
    nr_inner :: Int
    r_inner :: Float64
    r_outer :: Float64
end

function SphericalShellGrid(arch::AbstractArchitecture;
        lmax    :: Int,
        mmax    :: Int     = lmax,
        nlat    :: Int     = 3 * lmax ÷ 2,
        nlon    :: Int     = 2 * nlat,
        nr      :: Int,
        nr_inner :: Int    = max(2, nr ÷ 4),
        r_inner :: Float64 = 0.35,
        r_outer :: Float64 = 1.0)
    nlat > 0 || throw(ArgumentError("nlat must be > 0"))
    nlon > 0 || throw(ArgumentError("nlon must be > 0"))
    nr   > 0 || throw(ArgumentError("nr must be > 0"))
    nr_inner > 1 || throw(ArgumentError("nr_inner must be > 1"))
    0.0 < r_inner < r_outer || throw(ArgumentError("need 0 < r_inner < r_outer"))
    return SphericalShellGrid(arch, lmax, mmax, nlat, nlon, nr, nr_inner, r_inner, r_outer)
end

SphericalShellGrid(; arch::AbstractArchitecture=CPU(), kwargs...) =
    SphericalShellGrid(arch; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", g::SphericalShellGrid)
    arch = g.arch isa CPU ? "CPU" : "GPU"
    println(io, "SphericalShellGrid($arch)")
    println(io, "  lmax=$(g.lmax), mmax=$(g.mmax)")
    println(io, "  nlat=$(g.nlat), nlon=$(g.nlon), nr=$(g.nr), nr_inner=$(g.nr_inner)")
    print(io,   "  r_inner=$(g.r_inner), r_outer=$(g.r_outer)")
end

# SphericalBallGrid — full sphere geometry (fluid in a ball)
struct SphericalBallGrid
    arch :: AbstractArchitecture
    lmax :: Int
    mmax :: Int
    nlat :: Int
    nlon :: Int
    nr   :: Int
end

function SphericalBallGrid(arch::AbstractArchitecture;
        lmax :: Int,
        mmax :: Int = lmax,
        nlat :: Int = 3 * lmax ÷ 2,
        nlon :: Int = 2 * nlat,
        nr   :: Int)
    nlat > 0 || throw(ArgumentError("nlat must be > 0"))
    nlon > 0 || throw(ArgumentError("nlon must be > 0"))
    nr   > 0 || throw(ArgumentError("nr must be > 0"))
    return SphericalBallGrid(arch, lmax, mmax, nlat, nlon, nr)
end

SphericalBallGrid(; arch::AbstractArchitecture=CPU(), kwargs...) =
    SphericalBallGrid(arch; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", g::SphericalBallGrid)
    arch = g.arch isa CPU ? "CPU" : "GPU"
    println(io, "SphericalBallGrid($arch)")
    println(io, "  lmax=$(g.lmax), mmax=$(g.mmax)")
    print(io,   "  nlat=$(g.nlat), nlon=$(g.nlon), nr=$(g.nr)")
end
