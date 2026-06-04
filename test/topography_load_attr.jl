using Test
using NCDatasets

# Regression: load_topography_from_file must coerce NetCDF attribute types.
# NetCDF stores integer attributes as 32-bit (Int32) and floats possibly as
# Float32, but TopographyField's constructor requires Int / Float64. Before the
# fix (Int(...) / Float64(...) at the read sites), any topo file carrying an
# `lmax` attribute threw MethodError: no method matching TopographyField(::Int32, ...).
@testset "load_topography_from_file coerces NetCDF attribute types" begin
    tmp = tempname() * ".nc"
    NCDataset(tmp, "c") do ds
        ds.attrib["lmax"] = 8          # round-trips as Int32
        ds.attrib["radius"] = 1.5      # may round-trip as Float32/Float64
        defDim(ds, "nlm", 45)          # nlm for lmax = mmax = 8
        defVar(ds, "h_real", zeros(Float64, 45), ("nlm",))
        defVar(ds, "h_imag", zeros(Float64, 45), ("nlm",))
    end
    field = GeoDynamo.load_topography_from_file(tmp, GeoDynamo.bcs.OUTER_BOUNDARY)
    @test field.lmax == 8
    @test field.lmax isa Int
    @test field.radius ≈ 1.5
    @test field.radius isa Float64
    @test field.nlm == 45
    rm(tmp, force = true)
end
