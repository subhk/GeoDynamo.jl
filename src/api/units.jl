# Human-readable formatting helpers (Oceananigans conventions).
#
# `prettytime` is for WALL-CLOCK durations only. Model time in GeoDynamo is
# nondimensional (diffusion-time units), so model times/timesteps print via
# `prettysummary` instead (user decision, see the 2026-06-11 parity spec).

# Printf.@sprintf is in scope via the parent module's `using Printf`.

"""
    prettytime(t)

Format a duration `t` in seconds as a human-readable string, e.g.
`"2.341 seconds"`, `"1.500 days"`, `"100 ns"`. Follows Oceananigans
conventions: picks ns/μs/ms/seconds/minutes/hours/days; integer-valued
quantities drop the decimals; second/minute/hour/day pluralize (sub-second unit symbols do not). NaN returns "NaN"; negative durations format as "-<formatted>".
"""
function prettytime(t::Real)
    t == 0 && return "0 seconds"
    isnan(t) && return "NaN"
    t < 0 && return "-" * prettytime(-t)
    isfinite(t) || return "Inf days"

    if t < 1e-6
        value, units = t * 1e9, "ns"
    elseif t < 1e-3
        value, units = t * 1e6, "μs"
    elseif t < 1
        value, units = t * 1e3, "ms"
    elseif t < 60
        value, units = t, "second"
    elseif t < 3600
        value, units = t / 60, "minute"
    elseif t < 86400
        value, units = t / 3600, "hour"
    else
        value, units = t / 86400, "day"
    end

    if units in ("ns", "μs", "ms")
        body = isinteger(value) ? string(Int(value)) : @sprintf("%.3f", value)
        return string(body, " ", units)
    else
        if isinteger(value)
            n = Int(value)
            return string(n, " ", units, n == 1 ? "" : "s")
        else
            return string(@sprintf("%.3f", value), " ", units, "s")
        end
    end
end

"""
    prettysummary(x)

Compact number formatting for nondimensional quantities (model time, Δt,
physics parameters): integers print without a trailing `.0`, `Inf` and
`typemax(Int)` print as `"Inf"`, everything else uses Julia's shortest
round-trip float printing.
"""
prettysummary(x::Integer) = x == typemax(typeof(x)) ? "Inf" : string(x)
function prettysummary(x::Real)
    isfinite(x) || return isnan(x) ? "NaN" : (Float64(x) > 0 ? "Inf" : "-Inf")
    f = Float64(x)
    isinteger(f) && abs(f) < 1e15 && return string(Int(f))
    return string(x)
end
prettysummary(x) = string(x)
