# Time-Frequency Analysis of Geodynamo Energies

This directory contains scripts for comprehensive time-frequency analysis of kinetic and magnetic energies from merged Geodynamo.jl simulation outputs.

## Workflow Overview

1. **Merge simulation outputs** → `merge_outputs.jl`
2. **Compute time-frequency spectra** → `time_frequency_spectra.jl`  
3. **Generate publication plots** → `plot_time_frequency_spectra.jl`

## Scripts

### `time_frequency_spectra.jl`
**Purpose**: Compute comprehensive time-frequency analysis of kinetic and magnetic energies

**Features**:
- Time series analysis of total and modal energies
- Short-Time Fourier Transform (STFT) for time-frequency decomposition
- Continuous Wavelet Transform (CWT) for multi-scale analysis
- Power Spectral Density (PSD) computation
- Spherical harmonic mode evolution analysis

**Usage Examples**:
```bash
# Basic analysis with default parameters
julia --project=. scripts/time_frequency_spectra.jl ./output

# Custom time range and analysis parameters  
julia --project=. scripts/time_frequency_spectra.jl ./output \
      --start=1.0 --end=10.0 --window=512 --overlap=0.75 \
      --method=both --l_max=20 --output=results.jld2
```

**Options**:
- `--start=<t>`, `--end=<t>`: Time range for analysis
- `--window=<N>`: STFT window size (default: 256)
- `--overlap=<f>`: Window overlap fraction (default: 0.75) 
- `--method=<str>`: Analysis method - `stft`, `wavelet`, or `both`
- `--l_max=<N>`: Maximum spherical harmonic degree
- `--dt_target=<f>`: Target time spacing for resampling

### `plot_time_frequency_spectra.jl`
**Purpose**: Create publication-ready visualizations of time-frequency analysis results

**Plot Types**:
- **Time series**: Energy evolution over time
- **PSD**: Power spectral density plots
- **STFT**: Short-Time Fourier Transform spectrograms
- **Wavelet**: Continuous wavelet transform spectrograms
- **Modes**: Spherical harmonic mode analysis

**Usage Examples**:
```bash
# Generate all plot types
julia --project=. scripts/plot_time_frequency_spectra.jl results.jld2

# Custom output format and selection
julia --project=. scripts/plot_time_frequency_spectra.jl results.jld2 \
      --outdir=./plots --format=pdf --dpi=300 \
      --plots=timeseries,stft,psd --l_modes=1,2,3
```

**Options**:
- `--outdir=<dir>`: Output directory for plots  
- `--format=<fmt>`: Output format (`png`, `pdf`, `svg`)
- `--plots=<list>`: Select specific plot types
- `--l_modes=<list>`: Highlight specific l modes
- `--log_scale`: Use logarithmic frequency scaling

## Typical Workflow

### 1. Merge Raw Outputs
```bash
# Merge all available times
julia --project=. scripts/merge_outputs.jl ./raw_output --all

# Or merge specific time range
julia --project=. scripts/merge_outputs.jl ./raw_output \
      --time=1.0 --time=2.0 --time=3.0
```

### 2. Compute Time-Frequency Analysis
```bash
# Full analysis with both STFT and wavelets
julia --project=. scripts/time_frequency_spectra.jl ./merged_output \
      --method=both --l_max=15 --output=tf_analysis.jld2
```

### 3. Generate Plots
```bash
# Create all visualization types
julia --project=. scripts/plot_time_frequency_spectra.jl tf_analysis.jld2 \
      --outdir=./figures --format=pdf --dpi=300
```

## Output Data Structure

The `time_frequency_spectra.jl` script saves results in JLD2 format with the following structure:

```julia
results = (
    metadata = Dict(
        analysis_time, method, time_range, n_time_points, dt,
        l_max_actual, window_size, overlap
    ),
    
    time_series = Dict(
        times,           # Time array
        Ek_total,        # Total kinetic energy time series
        Eb_total,        # Total magnetic energy time series  
        Ek_lm,          # Kinetic energy by (l,m) mode
        Eb_lm,          # Magnetic energy by (l,m) mode
        l_values, m_values
    ),
    
    psd = Dict(
        frequencies,     # Frequency array
        psd_Ek,         # Kinetic energy power spectral density
        psd_Eb          # Magnetic energy power spectral density
    ),
    
    stft = Dict(        # Short-Time Fourier Transform
        times,          # STFT time axis
        frequencies,    # STFT frequency axis  
        stft_Ek, stft_Eb,        # Complex STFT coefficients
        psd_Ek, psd_Eb,          # Power spectrograms
        stft_modes, psd_modes,   # Mode-specific analysis
        l_selected              # Analyzed l values
    ),
    
    wavelet = Dict(     # Continuous Wavelet Transform (if available)
        times, frequencies, scales,
        cwt_Ek, cwt_Eb,         # Complex wavelet coefficients
        power_Ek, power_Eb      # Wavelet power spectrograms
    ),
    
    mode_evolution = Dict(
        l_values,               # Unique l degrees
        Ek_by_l, Eb_by_l,      # Energy by l degree
        Ek_fraction_by_l,       # Fractional energy contribution
        Eb_fraction_by_l,
        total_Ek, total_Eb      # Total energy time series
    )
)
```

## Dependencies

**Required**:
- `JLD2.jl`: Data storage
- `NetCDF.jl`: Reading simulation outputs
- `FFTW.jl`: Fast Fourier transforms
- `DSP.jl`: Signal processing (STFT, periodograms)
- `Plots.jl`: Visualization

**Optional**:
- `ContinuousWavelets.jl`: Wavelet analysis (if not available, wavelet analysis is skipped)

## Physical Interpretation

### Energy Components
- **Kinetic Energy**: `Ek = ∫ ½(|u_toroidal|² + |u_poloidal|²) dV`
- **Magnetic Energy**: `Eb = ∫ ½(|B_toroidal|² + |B_poloidal|²) dV`

### Frequency Domain Analysis
- **Power Spectral Density**: Shows dominant frequencies in energy oscillations
- **Time-Frequency Analysis**: Reveals how spectral content evolves over time
- **Modal Analysis**: Energy distribution across spherical harmonic degrees

### Typical Geodynamo Features
- **Fundamental modes** (l=1): Dipole field characteristics
- **Higher-order modes** (l>1): Fine structure and turbulent cascades  
- **Frequency peaks**: Characteristic oscillation modes (MAC waves, torsional oscillations)
- **Broadband features**: Turbulent energy cascades

## Example Results

The analysis typically reveals:
1. **Dominant l=1 magnetic energy** (dipolar field)
2. **Broadband kinetic energy** across multiple l modes
3. **Low-frequency peaks** from large-scale oscillations
4. **Time-varying spectral features** during magnetic reversals or excursions

## Performance Notes

- **Memory usage** scales with number of time points and l_max
- **STFT window size** affects time vs frequency resolution
- **Wavelet analysis** is computationally intensive but provides superior time-frequency localization
- **Parallel processing** is not currently implemented but could be added for large datasets