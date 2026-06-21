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
function compute_diagnostics(fields::Dict{String, Any}, field_info::FieldInfo)
    diagnostics = Dict{String, Float64}()
    comm = get_comm()
    nprocs = (comm !== nothing && MPI.Comm_size(comm) > 1) ? MPI.Comm_size(comm) : 1
    use_mpi = nprocs > 1

    # Helper: reduce a scalar across all ranks
    _global_sum(x) = use_mpi ? MPI.Allreduce(x, MPI.SUM, comm) : x
    _global_max(x) = use_mpi ? MPI.Allreduce(x, MPI.MAX, comm) : x
    _global_min(x) = use_mpi ? MPI.Allreduce(x, MPI.MIN, comm) : x

    # Physical-space fields: need global reduction for correct mean/min/max/std
    for (key, prefix) in [("temperature", "temp"), ("composition", "comp")]
        if haskey(fields, key)
            F = fields[key]
            local_n = Float64(length(F))
            local_sum = Float64(sum(F))
            local_min = Float64(minimum(F))
            local_max = Float64(maximum(F))

            global_n = _global_sum(local_n)
            global_sum = _global_sum(local_sum)
            global_min = _global_min(local_min)
            global_max = _global_max(local_max)
            global_mean = global_sum / global_n

            # Two-pass variance via parallel algorithm: sum of (x - global_mean)^2
            local_sq_sum = sum(x -> (Float64(x) - global_mean)^2, F)
            global_sq_sum = _global_sum(local_sq_sum)

            diagnostics["$(prefix)_mean"] = global_mean
            diagnostics["$(prefix)_std"] = sqrt(max(zero(Float64), global_sq_sum /
                                                                   global_n))
            diagnostics["$(prefix)_min"] = global_min
            diagnostics["$(prefix)_max"] = global_max
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
                diagnostics["$(component)_rms"] = sqrt(max(zero(Float64), global_energy /
                                                                          global_count))
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
function compute_spectral_energy_diagnostics!(diagnostics::Dict{String, Float64},
        component::String,
        real_part::AbstractArray,
        imag_part::AbstractArray,
        field_info::FieldInfo)
    if !field_info.has_config
        return
    end

    config = field_info.config

    # Use config.lmax (authoritative) rather than maximum(l_values), which is
    # identical but avoids a redundant reduction over the full nlm list.
    l_max = config.lmax
    l_energies = zeros(Float64, l_max + 1)

    # Map storage slots to global (l, m) via local_spectral_lm_map.  The spec
    # pencil's dim-1 is the l-slot axis and dim-2 is the m-slot axis; lm_map
    # has the same shape as (real_part dim-1, real_part dim-2) so indices
    # align exactly.  Slots with lm_map value 0 are unused padding and are
    # skipped.  This is correct for every Phase (single-rank dense nlm list,
    # Phase-3 (lmax+1,mmax+1,nr) rectangular grid, and any future layout).
    # Toroidal/poloidal components store scalar POTENTIALS; the physical field
    # energy per degree carries an l(l+1) factor (Σ_lm l(l+1)(|T|²+|P|²) is the
    # angular energy). Without it peak_l / spectral_centroid rank coefficient
    # magnitudes rather than energy. Scalars (temperature/composition) carry no
    # such factor.
    is_vector = occursin("toroidal", component) || occursin("poloidal", component)

    lm_map = local_spectral_lm_map(config)
    for i1 in axes(real_part, 1), i2 in axes(real_part, 2)
        # Guard: lm_map may be smaller than real_part if shapes diverge (e.g.
        # a legacy caller passes a differently-shaped array); skip silently.
        (i1 > size(lm_map, 1) || i2 > size(lm_map, 2)) && continue
        mode = lm_map[i1, i2]
        mode == 0 && continue
        l = config.l_values[mode]
        l_weight = is_vector ? Float64(l * (l + 1)) : 1.0
        l_energy = zero(Float64)
        for k in axes(real_part, 3)
            l_energy += Float64(real_part[i1, i2, k])^2 + Float64(imag_part[i1, i2, k])^2
        end
        l_energies[l + 1] += l_weight * l_energy
    end

    # Reduce l_energies across all MPI ranks so that every rank holds the
    # globally-summed per-degree energy.  Each rank only owns a subset of the
    # m-slots (distributed spec pencil), so without this reduction the
    # per-degree sums are partial and peak_l / spectral_centroid are wrong.
    # This call is reached uniformly on all ranks (field_info.has_config is
    # rank-uniform; no rank-divergent early return above), so the collective
    # is safe from deadlock.
    comm = get_comm()
    if comm !== nothing && MPI.Comm_size(comm) > 1
        l_energies = MPI.Allreduce(l_energies, MPI.SUM, comm)
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
