!*************************************************************************
!
!
!*************************************************************************
 module rotation
!*************************************************************************
   use variables
   use timestep
   implicit none
   save

   double precision :: rot_omega
   double precision :: rot_inertia
   double precision :: rot_velTorq
   double precision :: rot_magTorq

   double precision, private :: velTorq
   double precision, private :: magTorq

   
 contains

!------------------------------------------------------------------------
!  initialise 
!------------------------------------------------------------------------
   subroutine rot_precompute()
      rot_omega   = 0d0
      rot_inertia = (8d0*d_PI/15d0) * mes_oc%r(1,1)**5
      rot_velTorq = 0d0
      rot_magTorq = 0d0
   end subroutine rot_precompute



!------------------------------------------------------------------------
!
!------------------------------------------------------------------------
   subroutine rot_predictor()

      rot_omega = rot_omega  &
         +  (tim_dt/rot_inertia) * (rot_velTorq + rot_magTorq) 

      velTorq = rot_velTorq
      magTorq = rot_magTorq

   end subroutine rot_predictor
   
   
!------------------------------------------------------------------------
!
!------------------------------------------------------------------------
   subroutine rot_corrector()

      rot_omega = rot_omega  &
         + (tim_dt*d_implicit/rot_inertia)  &
         * ((rot_velTorq + rot_magTorq) - ( velTorq + magTorq ) )

      velTorq = rot_velTorq
      magTorq = rot_magTorq

   end subroutine rot_corrector
   

   
!*************************************************************************
 end module rotation
!*************************************************************************
