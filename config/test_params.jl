# Test parameter file with modified values
const i_N   = 128       # Doubled the radial points
const i_L   = 64        # Doubled the spherical harmonic degree
const d_Ra  = 2e6       # Doubled the Rayleigh number
const d_E   = 5e-5      # Modified Ekman number

# Keep other parameters at default values
const i_Nic = 16        
const i_M   = 32        
const i_Th  = 64        
const i_Ph  = 128       
const i_KL  = 4         

const i_L1 = i_L
const i_M1 = i_M
const i_H1 = (i_L + 1) * (i_L + 2) ÷ 2 - 1
const i_pH1 = i_H1
const i_Ma = i_M ÷ 2

const d_rratio = 0.35         
const d_Pr = 1.0              
const d_Pm = 1.0              
const d_Ro = 1e-4             
const d_q = 1.0               

const d_timestep = 1e-4
const d_time = 0.0
const d_implicit = 0.5        
const d_dterr = 1e-8          
const d_courant = 0.5         
const i_maxtstep = 10000      
const i_save_rate2 = 100      
const output_precision = :float64   
const independent_output_files = true   

const i_vel_bc = 1            
const i_tmp_bc = 1            
const i_cmp_bc = 1            

const b_mag_impose = false    
