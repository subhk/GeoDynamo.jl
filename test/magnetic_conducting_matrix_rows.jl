using Test, MPI, LinearAlgebra
function _row(bm, i)  # materialize banded row i as dense
    N=bm.size;
    bw=bm.bandwidth;
    r=zeros(eltype(bm.data), N)
    for j in max(1, i - bw):min(N, i + bw)
        ;
        br=bw+1+i-j;
        (1<=br<=2bw+1)&&(r[j]=bm.data[br, j]);
    end;
    r
end
@testset "conducting inner row = (∂/∂r − α)" begin
    if !MPI.Initialized()
        ;
        MPI.Init();
    end
    cfg = GeoDynamo.create_shtnskit_config(
        lmax = 4, mmax = 4, nlat = 12, nlon = 16, nr = 16)
    dom = GeoDynamo.create_radial_domain(16)
    α = Dict(1=>0.7, 2=>1.3, 3=>2.1, 4=>3.0)
    pol = GeoDynamo.create_magnetic_poloidal_matrices(
        cfg, dom, 1.0, 1e-3; theta = 0.5, T = Float64, inner_alpha = α)
    d1 = GeoDynamo.create_derivative_matrix(1, dom)
    for (idx, l) in enumerate(pol.l_values)
        l==0 && continue
        row=_row(pol.system_matrices[idx], 1);
        exp=_row(d1, 1);
        exp[1]-=α[l]
        @test row ≈ exp atol=1e-12
    end
    # default unchanged:
    pol0 = GeoDynamo.create_magnetic_poloidal_matrices(
        cfg, dom, 1.0, 1e-3; theta = 0.5, T = Float64)
    @test pol0 isa typeof(pol)
end
