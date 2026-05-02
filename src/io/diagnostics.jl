# ================================================================================
# Diagnostics Computation
# ================================================================================

"""
    compute_diagnostics(fields, field_info)

Compute scalar diagnostics for available physical and spectral fields.

Physical fields get global mean, standard deviation, minimum, and maximum.
Spectral fields get globally reduced energy, RMS, and maximum coefficient
magnitude, with extra degree-wise diagnostics when SHTns metadata is available.
"""
function compute_diagnostics(fields::Dict{String,Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()
    comm = get_comm()
    nprocs = (comm !== nothing && MPI.Comm_size(comm) > 1) ? MPI.Comm_size(comm) : 1
    use_mpi = nprocs > 1

    # Helper: reduce a scalar across all ranks
    _global_sum(x)  = use_mpi ? MPI.Allreduce(x, MPI.SUM, comm) : x
    _global_max(x)  = use_mpi ? MPI.Allreduce(x, MPI.MAX, comm) : x
    _global_min(x)  = use_mpi ? MPI.Allreduce(x, MPI.MIN, comm) : x

    # Physical-space fields: need global reduction for correct mean/min/max/std
    for (key, prefix) in [("temperature", "temp"), ("composition", "comp")]
        if haskey(fields, key)
            F = fields[key]
            local_n   = Float64(length(F))
            local_sum = Float64(sum(F))
            local_min = Float64(minimum(F))
            local_max = Float64(maximum(F))

            global_n   = _global_sum(local_n)
            global_sum = _global_sum(local_sum)
            global_min = _global_min(local_min)
            global_max = _global_max(local_max)
            global_mean = global_sum / global_n

            # Two-pass variance via parallel algorithm: sum of (x - global_mean)^2
            local_sq_sum = sum(x -> (Float64(x) - global_mean)^2, F)
            global_sq_sum = _global_sum(local_sq_sum)

            diagnostics["$(prefix)_mean"] = global_mean
            diagnostics["$(prefix)_std"]  = sqrt(max(zero(Float64), global_sq_sum / global_n))
            diagnostics["$(prefix)_min"]  = global_min
            diagnostics["$(prefix)_max"]  = global_max
        end
    end

    # Spectral fields: reduce energy, rms, max across ranks
    for component in ["velocity_toroidal", "velocity_poloidal",
                      "magnetic_toroidal", "magnetic_poloidal",
                      "temperature_spectral", "composition_spectral"]
        if haskey(fields, component)
            field_data = fields[component]
            if haskey(field_data, "real") && haskey(field_data, "imag")
                real_part = field_data["real"]
                imag_part = field_data["imag"]

                local_energy = zero(Float64)
                local_energy_comp = zero(Float64)  # Kahan compensator
                local_max_mag = zero(Float64)
                local_count = Float64(length(real_part))
                for i in eachindex(real_part, imag_part)
                    magnitude_sq = Float64(real_part[i])^2 + Float64(imag_part[i])^2
                    # Kahan compensated summation for energy
                    y = magnitude_sq - local_energy_comp
                    t = local_energy + y
                    local_energy_comp = (t - local_energy) - y
                    local_energy = t
                    local_max_mag = max(local_max_mag, sqrt(magnitude_sq))
                end

                global_energy = _global_sum(local_energy)
                global_max_mag = _global_max(local_max_mag)
                global_count = _global_sum(local_count)

                diagnostics["$(component)_energy"] = 0.5 * global_energy
                diagnostics["$(component)_rms"] = sqrt(max(zero(Float64), global_energy / global_count))
                diagnostics["$(component)_max"] = global_max_mag

                if field_info.has_config && !isempty(field_info.l_values)
                    compute_spectral_energy_diagnostics!(diagnostics, component,
                                                        real_part, imag_part, field_info)
                end
            end
        end
    end

    return diagnostics
end

"""
    compute_spectral_energy_diagnostics!(diagnostics, component, real_part, imag_part, field_info)

Add degree-wise spectral energy summaries for one complex spectral component.

Requires `field_info.config` so coefficient rows can be mapped to spherical
harmonic degree `l`. The function mutates `diagnostics` in place.
"""
function compute_spectral_energy_diagnostics!(diagnostics::Dict{String,Float64},
                                            component::String,
                                            real_part::AbstractArray,
                                            imag_part::AbstractArray,
                                            field_info::FieldInfo)
    if !field_info.has_config
        return
    end

    config = field_info.config
    l_values = config.l_values

    l_max = maximum(l_values)
    l_energies = zeros(Float64, l_max + 1)

    for (idx, l) in enumerate(l_values)
        if idx <= size(real_part, 1)
            l_energy = zero(eltype(real_part))
            for j in axes(real_part, 2), k in axes(real_part, 3)
                l_energy += real_part[idx, j, k]^2 + imag_part[idx, j, k]^2
            end
            l_energies[l + 1] += l_energy
        end
    end

    total_energy = sum(l_energies)
    if total_energy > 0
        peak_l = argmax(l_energies) - 1
        diagnostics["$(component)_peak_l"] = Float64(peak_l)

        spectral_centroid = sum((0:l_max) .* l_energies) / total_energy
        diagnostics["$(component)_spectral_centroid"] = spectral_centroid

        low_mode_energy = sum(l_energies[1:min(11, length(l_energies))])
        diagnostics["$(component)_low_mode_fraction"] = low_mode_energy / total_energy
    end
end
