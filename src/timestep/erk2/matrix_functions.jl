# ================================================================================
# ERK2 Matrix Functions (φ₁ and φ₂)
# ================================================================================

"""
    compute_phi1_function(A, expA)

Compute φ1(A) = (exp(A) - I) / A efficiently with comprehensive error handling.
Uses series expansion for small ||A|| to avoid numerical issues.
"""
function compute_phi1_function(A::Matrix{T}, expA::Matrix{T}) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)

    # Check for NaN or Inf in inputs
    if !all(isfinite.(A)) || !all(isfinite.(expA))
        @warn "Non-finite values detected in φ1 computation, using identity approximation"
        return I_mat
    end

    # Check if A is close to zero matrix - use series expansion
    A_norm = opnorm(A)
    if A_norm < 1e-2
        # Use Taylor series: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)! = I/1! + A/2! + A²/3! + A³/4! + ...
        result = copy(I_mat)  # k=0: A⁰/1! = I/1!
        A_power = copy(I_mat)
        for k in 1:15  # Use enough terms for good accuracy
            A_power = A_power * A  # A^k
            factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
            term = A_power / factorial_k_plus_1
            result += term
            if opnorm(term) < eps(T) * 100
                break
            end
        end
        return result
    end

    # For larger A, use φ1(A) = (exp(A) - I) / A
    diff = expA - I_mat

    # Use lu factorization for stable division by A
    try
        lu_A = lu(A)

        # Check condition number
        if rcond(lu_A) < sqrt(eps(T))
            @warn "Ill-conditioned matrix in φ1 computation (rcond = $(rcond(lu_A))), using series expansion"
            # Fall back to series expansion: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)!
            result = copy(I_mat)  # k=0: A⁰/1!
            A_power = copy(I_mat)
            for k in 1:15
                A_power = A_power * A  # A^k
                factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
                term = A_power / factorial_k_plus_1
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        else
            result = lu_A \ diff
        end

        # Validate result
        if !all(isfinite.(result))
            @warn "Non-finite result in φ1 computation, falling back to series expansion"
            result = I_mat + A/2
            A_power = A * A
            factorial = 6
            for k in 2:15
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
                A_power = A_power * A
                factorial *= (k + 2)
            end
        end

        return result

    catch e
        @warn "LU factorization failed in φ1 computation: $e, using series expansion"
        try
            # Fall back to series expansion: φ1(A) = Σ(k=0 to ∞) A^k/(k+1)!
            result = copy(I_mat)  # k=0: A⁰/1!
            A_power = copy(I_mat)
            for k in 1:15
                A_power = A_power * A  # A^k
                factorial_k_plus_1 = factorial(k + 1)  # (k+1)!
                term = A_power / factorial_k_plus_1
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        catch e2
            @error "Complete failure in φ1 computation: $e2, returning identity"
            return I_mat
        end
    end
end

"""
    Phi2ConditioningMonitor

Global structure for monitoring φ₂ function conditioning during ERK2 integration.
Tracks worst conditioning, series expansion usage, and numerical issues.
"""
mutable struct Phi2ConditioningMonitor
    worst_rcond::Float64
    worst_l::Int
    series_expansion_count::Int
    lu_failure_count::Int
    nonfinite_count::Int
    last_report_step::Int
    enable_monitoring::Bool
end

# Global monitor instance
const PHI2_MONITOR = Phi2ConditioningMonitor(1.0, 0, 0, 0, 0, 0, true)

"""
    reset_phi2_monitor!()

Reset φ₂ conditioning monitor statistics.
"""
function reset_phi2_monitor!()
    PHI2_MONITOR.worst_rcond = 1.0
    PHI2_MONITOR.worst_l = 0
    PHI2_MONITOR.series_expansion_count = 0
    PHI2_MONITOR.lu_failure_count = 0
    PHI2_MONITOR.nonfinite_count = 0
    PHI2_MONITOR.last_report_step = 0
end

"""
    report_phi2_conditioning(step::Int; interval::Int=100)

Report φ₂ conditioning statistics periodically during simulation.
"""
function report_phi2_conditioning(step::Int; interval::Int=100)
    if !PHI2_MONITOR.enable_monitoring
        return
    end

    if step - PHI2_MONITOR.last_report_step >= interval
        if get_rank() == 0
            @info """
            ╔══════════════════════════════════════════════════════════╗
            ║            φ₂ Conditioning Report (Step $step)
            ╠══════════════════════════════════════════════════════════╣
            ║ Worst rcond:             $(PHI2_MONITOR.worst_rcond)
            ║ Worst mode (l):          $(PHI2_MONITOR.worst_l)
            ║ Series expansion used:   $(PHI2_MONITOR.series_expansion_count) times
            ║ LU failures:             $(PHI2_MONITOR.lu_failure_count) times
            ║ Non-finite values:       $(PHI2_MONITOR.nonfinite_count) times
            ╚══════════════════════════════════════════════════════════╝
            """
        end
        PHI2_MONITOR.last_report_step = step
        # Reset counters for next interval
        reset_phi2_monitor!()
    end
end

"""
    compute_phi2_function(A, expA; l=0)

Compute φ2(A) = (exp(A) - I - A) / A² efficiently with comprehensive error handling.
Uses series expansion for small ||A|| to avoid numerical issues.
Tracks conditioning statistics when monitoring is enabled.
"""
function compute_phi2_function(A::Matrix{T}, expA::Matrix{T}; l::Int=0) where T
    nr = size(A, 1)
    I_mat = Matrix{T}(I, nr, nr)

    # Check for NaN or Inf in inputs
    if !all(isfinite.(A)) || !all(isfinite.(expA))
        if PHI2_MONITOR.enable_monitoring
            PHI2_MONITOR.nonfinite_count += 1
        end
        @warn "Non-finite values detected in φ2 computation (l=$l), using zero approximation"
        return zeros(T, nr, nr)
    end

    # Check if A is close to zero matrix - use series expansion
    A_norm = opnorm(A)
    if A_norm < 1e-2
        # Use Taylor series: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)! = I/2! + A/3! + A²/4! + A³/5! + ...
        result = I_mat / 2  # k=0: A⁰/2! = I/2
        A_power = copy(A)   # k=1: A¹/3!
        factorial = 6       # 3! = 6
        result += A_power / factorial

        for k in 2:15  # Use enough terms for good accuracy
            A_power = A_power * A  # A^k
            factorial *= (k + 2)   # (k+2)!
            term = A_power / factorial
            result += term
            if opnorm(term) < eps(T) * 100
                break
            end
        end
        return result
    end

    # For larger A, use φ2(A) = (exp(A) - I - A) / A²
    diff = expA - I_mat - A

    # Need to solve A² * result = diff
    # This is equivalent to solving A * (A * result) = diff
    try
        lu_A = lu(A)

        # Check condition number and track worst conditioning
        rcond_val = rcond(lu_A)
        if PHI2_MONITOR.enable_monitoring
            if rcond_val < PHI2_MONITOR.worst_rcond
                PHI2_MONITOR.worst_rcond = rcond_val
                PHI2_MONITOR.worst_l = l
            end
        end

        if rcond_val < sqrt(eps(T))
            if PHI2_MONITOR.enable_monitoring
                PHI2_MONITOR.series_expansion_count += 1
            end
            @warn "Ill-conditioned matrix in φ2 computation (l=$l, rcond=$rcond_val), using series expansion"
            # Fall back to series expansion: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)!
            result = I_mat / 2  # k=0: A⁰/2! = I/2
            A_power = copy(A)   # k=1: A¹/3!
            factorial = 6       # 3! = 6
            result += A_power / factorial

            for k in 2:15
                A_power = A_power * A  # A^k
                factorial *= (k + 2)   # (k+2)!
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        else
            # Solve A * temp = diff, then A * result = temp
            temp = lu_A \ diff
            result = lu_A \ temp
        end

        # Validate result
        if !all(isfinite.(result))
            @warn "Non-finite result in φ2 computation, falling back to series expansion"
            result = I_mat / 2 + A / 6
            A_power = A * A
            factorial = 24
            for k in 2:15
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
                A_power = A_power * A
                factorial *= (k + 3)
            end
        end

        return result

    catch e
        if PHI2_MONITOR.enable_monitoring
            PHI2_MONITOR.lu_failure_count += 1
        end
        @warn "LU factorization failed in φ2 computation (l=$l): $e, using series expansion"
        try
            # Fall back to series expansion: φ2(A) = Σ(k=0 to ∞) A^k/(k+2)!
            result = I_mat / 2  # k=0: A⁰/2! = I/2
            A_power = copy(A)   # k=1: A¹/3!
            factorial = 6       # 3! = 6
            result += A_power / factorial

            for k in 2:15
                A_power = A_power * A  # A^k
                factorial *= (k + 2)   # (k+2)!
                term = A_power / factorial
                result += term
                if opnorm(term) < eps(T) * 100
                    break
                end
            end
            return result
        catch e2
            @error "Complete failure in φ2 computation: $e2, returning zero matrix"
            return zeros(T, nr, nr)
        end
    end
end
