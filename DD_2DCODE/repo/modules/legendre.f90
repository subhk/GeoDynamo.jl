!**************************************************************************
!
! Public variables:
!
!
!    points
!       th  (j)         theta points, th_j
!       sin (j)		sin(th_j)
!       sin1(j)         1/sin(th_j)
!       gp  (j)         x_j = cos(th_j), gauss points
!       gw  (j)         gauss weights
!
!    spec->phys
!       P     (j,nh)    P_m^l(x_j), nh given L=i_L, M=i_M.
!       P_    (j,nh)    dth P 
!       sin1mP(j,nh)    (1/sin(th_j)) . m . P(j,nh)
!       llP_    (j,nh)  qst; 1/sqrt(l(l+1)) . dth P 
!       llsin1mP(j,nh)       1/sqrt(l(l+1)) . (1/sin(th_j)) . m . P(j,nh)
!
!    phys->spec weights
!       gwP       (j,nh)  gw(j) . (2l+1)/(2(2-d_{m0})) . P(j,nh)
!       gwllP_    (j,nh)  qst weights
!       gwllsin1mP(j,nh)
!
!
! Associated legendre functions are normalised s.t.
!   \int_0^\pi [P_l^m(\cos\theta)]^2 \sin\theta d\theta
!      = 2(2-d_{m0})/(2l+1)
! where d_{m0} is the dirac delta.
!
!**************************************************************************
#include "../parallel.h"
 module legendre
!**************************************************************************
   use mpi
   use parameters
   implicit none
   save

   double precision :: leg_th  (i_Th)
   double precision :: leg_sin (i_Th)
   double precision :: leg_sin1(i_Th)
   double precision :: leg_gp  (i_Th)
   double precision :: leg_gw  (i_Th)
   double precision :: leg_gwP0(i_Th)
   double precision :: leg_P     (i_Th, 0:i_H1)
   double precision :: leg_P_    (i_Th, 0:i_H1)
   double precision :: leg_sin1mP(i_Th, 0:i_H1)
   double precision :: leg_llP_    (i_Th, 0:i_H1)
   double precision :: leg_llsin1mP(i_Th, 0:i_H1)
   double precision :: leg_gwP       (i_Th, 0:i_H1)
   double precision :: leg_gwllP_    (i_Th, 0:i_H1)
   double precision :: leg_gwllsin1mP(i_Th, 0:i_H1)

 contains


!------------------------------------------------------------------------
!  Precompute points, weights
!------------------------------------------------------------------------
   subroutine leg_precompute()
      double precision   ::  pa( (i_L1+1)*(i_L1+2)/2 )
      double precision   :: dpa( (i_L1+1)*(i_L1+2)/2 )
      double precision :: w
      integer :: j, k
      _loop_setlm_vars
   
      					! get Gauss points and weights
      call gauwts(-1d0, 1d0, leg_gp, leg_gw, i_Th)
         				! get theta points
      leg_th = dacos(leg_gp)
         				! get sin, 1/sin points
      leg_sin  = dsin(leg_th)
      leg_sin1 = 1d0 / leg_sin
         				! get Schmidt-normalised 
                                        ! Legendre Functions, derivs,
                                        ! and reorder
      do j = 1, i_Th
         call schnla( pa, dpa, leg_gp(j), i_L1, 1)
         _loop_setlm_begin
               k = l*(l+1)/2+m_+1    
               leg_P     (j,nh) =  pa(k)
               leg_P_    (j,nh) = dpa(k)
               leg_sin1mP(j,nh) =  pa(k) * m_ * leg_sin1(j)
               
               if(l==0) then
                  leg_llP_    (j,nh) = 0d0
                  leg_llsin1mP(j,nh) = 0d0
               else
                  w = 1d0 / dsqrt(dble(l*(l+1)))
                  leg_llP_    (j,nh) = dpa(k) * w
                  leg_llsin1mP(j,nh) =  pa(k) * w * m_ * leg_sin1(j)
               end if
         _loop_setlm_end
         leg_gwP0(j) = pa(1) * leg_gw(j) * 0.5d0
      end do
         				! get P * weights
      _loop_setlm_begin
            do j = 1, i_Th
            
               w = leg_gw(j) * ((2d0*l+1d0)/4d0)
               if(m==0) w = 2d0*w
               leg_gwP(j,nh) = leg_P(j,nh) * w

               if(l==0) then
                  leg_gwllP_    (j,nh) = 0d0
                  leg_gwllsin1mP(j,nh) = 0d0
               else
                  w = w / dsqrt(dble(l*(l+1)))
                  leg_gwllP_    (j,nh) = leg_P_(j,nh) * w
                  leg_gwllsin1mP(j,nh) = leg_P (j,nh) * w*m_*leg_sin1(j)
               end if               
            end do
      _loop_setlm_end
      
   end subroutine leg_precompute


!**************************************************************************
 end module legendre
!**************************************************************************




