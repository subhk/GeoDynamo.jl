# Validate the spectral truncation and physical grid sizing shared by both
# geometries. Catches degenerate configurations at construction time with a
# clear message rather than letting them fail cryptically deep in the transform
# setup (e.g. nlat=1 only surfacing later as "nlat must be >= lmax+1").
function _validate_spectral_grid(lmax::Int, mmax::Int, nlat::Int, nlon::Int)
    lmax >= 1 || throw(ArgumentError("lmax must be >= 1 (got $lmax)"))
    0 <= mmax <= lmax ||
        throw(ArgumentError("mmax must satisfy 0 <= mmax <= lmax (got mmax=$mmax, lmax=$lmax)"))
    nlat >= lmax + 1 ||
        throw(ArgumentError("nlat must be >= lmax+1 = $(lmax + 1) for the transform (got nlat=$nlat)"))
    nlon >= 2 * mmax + 1 ||
        throw(ArgumentError("nlon must be >= 2*mmax+1 = $(2 * mmax + 1) (Nyquist) (got nlon=$nlon)"))
    return nothing
end

"""
    SphericalShellGrid(arch=CPU(); lmax, nr, mmax=lmax, nlat=3lmax÷2,
                       nlon=2nlat, nr_inner=max(2, nr÷4),
                       r_inner=0.35, r_outer=1.0)

Spherical shell geometry: fluid contained between two concentric spheres
(`r_inner < r < r_outer`), like Earth's outer core. This is the grid passed to
[`GeodynamoModel`](@ref).

# Arguments
- `arch`: compute architecture, `CPU()` or `GPU` (positional or `arch=`).
- `lmax`: maximum spherical-harmonic degree (required).
- `nr`: number of radial points (required).
- `mmax`: maximum order (defaults to `lmax`).
- `nlat`, `nlon`: latitude/longitude grid sizes (defaults follow the transform's
  anti-aliasing rule).
- `nr_inner`: radial points resolving the inner-core region.
- `r_inner`, `r_outer`: dimensionless shell radii (`0 < r_inner < r_outer`).

# Example
```julia
grid = SphericalShellGrid(lmax = 31, nr = 64)
```
"""
struct SphericalShellGrid
    arch::AbstractArchitecture
    lmax::Int
    mmax::Int
    nlat::Int
    nlon::Int
    nr::Int
    nr_inner::Int
    r_inner::Float64
    r_outer::Float64
end

function SphericalShellGrid(arch::AbstractArchitecture;
        lmax::Int,
        mmax::Int = lmax,
        nlat::Int = 3 * lmax ÷ 2,
        nlon::Int = 2 * nlat,
        nr::Int,
        nr_inner::Int = max(2, nr ÷ 4),
        r_inner::Float64 = 0.35,
        r_outer::Float64 = 1.0)
    _validate_spectral_grid(lmax, mmax, nlat, nlon)
    nr >= 2 || throw(ArgumentError("nr must be >= 2 (got $nr); a single radial point is degenerate"))
    nr_inner > 1 || throw(ArgumentError("nr_inner must be > 1 (got $nr_inner)"))
    nr_inner < nr || throw(ArgumentError("nr_inner must be < nr (got nr_inner=$nr_inner, nr=$nr)"))
    0.0 < r_inner < r_outer || throw(ArgumentError("need 0 < r_inner < r_outer"))
    return SphericalShellGrid(arch, lmax, mmax, nlat, nlon, nr, nr_inner, r_inner, r_outer)
end

function SphericalShellGrid(; arch::AbstractArchitecture = CPU(), kwargs...)
    SphericalShellGrid(arch; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", g::SphericalShellGrid)
    arch = g.arch isa CPU ? "CPU" : "GPU"
    println(io, "SphericalShellGrid($arch)")
    println(io, "  lmax=$(g.lmax), mmax=$(g.mmax)")
    println(io, "  nlat=$(g.nlat), nlon=$(g.nlon), nr=$(g.nr), nr_inner=$(g.nr_inner)")
    print(io, "  r_inner=$(g.r_inner), r_outer=$(g.r_outer)")
end

"""
    SphericalBallGrid(arch=CPU(); lmax, nr, mmax=lmax, nlat=3lmax÷2, nlon=2nlat)

Full-sphere geometry: fluid filling a ball (`0 ≤ r ≤ 1`) with no inner boundary,
like a stellar core or early planetary interior. This is the grid passed to
[`GeodynamoModel`](@ref).

# Arguments
- `arch`: compute architecture, `CPU()` or `GPU` (positional or `arch=`).
- `lmax`: maximum spherical-harmonic degree (required).
- `nr`: number of radial points (required).
- `mmax`: maximum order (defaults to `lmax`).
- `nlat`, `nlon`: latitude/longitude grid sizes (defaults follow the transform's
  anti-aliasing rule).

Unlike [`SphericalShellGrid`](@ref) there is no `r_inner`/`nr_inner`; regularity
at the origin is enforced by the ball solver.

# Example
```julia
grid = SphericalBallGrid(lmax = 31, nr = 48)
```
"""
struct SphericalBallGrid
    arch::AbstractArchitecture
    lmax::Int
    mmax::Int
    nlat::Int
    nlon::Int
    nr::Int
end

function SphericalBallGrid(arch::AbstractArchitecture;
        lmax::Int,
        mmax::Int = lmax,
        nlat::Int = 3 * lmax ÷ 2,
        nlon::Int = 2 * nlat,
        nr::Int)
    _validate_spectral_grid(lmax, mmax, nlat, nlon)
    nr >= 2 || throw(ArgumentError("nr must be >= 2 (got $nr); a single radial point is degenerate"))
    return SphericalBallGrid(arch, lmax, mmax, nlat, nlon, nr)
end

function SphericalBallGrid(; arch::AbstractArchitecture = CPU(), kwargs...)
    SphericalBallGrid(arch; kwargs...)
end

function Base.show(io::IO, ::MIME"text/plain", g::SphericalBallGrid)
    arch = g.arch isa CPU ? "CPU" : "GPU"
    println(io, "SphericalBallGrid($arch)")
    println(io, "  lmax=$(g.lmax), mmax=$(g.mmax)")
    print(io, "  nlat=$(g.nlat), nlon=$(g.nlon), nr=$(g.nr)")
end
