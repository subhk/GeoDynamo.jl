"""
    sync_spectral_history!(previous::SHTnsSpecField, current::SHTnsSpecField)

Copy spectral coefficients from `current` into `previous`.
This helper is shared by the solver rewrite path and any remaining legacy
timestepping code that keeps one-step nonlinear histories.
"""
function sync_spectral_history!(previous::SHTnsSpecField, current::SHTnsSpecField)
    copyto!(parent(previous.data_real), parent(current.data_real))
    copyto!(parent(previous.data_imag), parent(current.data_imag))
    return previous
end
