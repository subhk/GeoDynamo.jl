!**************************************************************************
! Non-linear terms
!
!
!*************************************************************************
#include "../parallel.h"
 module nonlinear
!*************************************************************************
   use velocity
   use codensity
   use magnetic
   use rotation
   use transform
   implicit none
   save

   type (coll), private :: cq,cs,ct
   type (spec), private :: sq,ss,st
   type (phys), private :: r,th,ph

 contains


!------------------------------------------------------------------------
!  Initialise nonlinear stuff
!------------------------------------------------------------------------
   subroutine non_precompute()
   end subroutine non_precompute


!-------------------------------------------------------------------------
!  nonlinear terms for velocity.  q,s->NPol, t->NTor
!------------------------------------------------------------------------
   subroutine non_velocity()
      double precision, parameter :: qRa = d_q*d_Ra
      double precision :: d(i_N), lsin(i_pTh), lgp(i_pTh)
      integer :: j,n,m,nh

      call var_spec_init(mes_oc,var_H, sq)
      call var_spec_init(mes_oc,var_H, ss)
      call var_spec_init(mes_oc,var_H, st)
      lsin = leg_sin(var_Th%pThi:var_Th%pThi+i_pTh-1)
      lgp  = leg_gp (var_Th%pThi:var_Th%pThi+i_pTh-1)
      
         			! advection  Ro u x curlu
      j = var_Th%pTh
      do n = 1, mes_oc%pN
         do m = 0, i_Ph-1
            r%Re (:j,m,n) = d_Ro*(vel_t%Re(:j,m,n)*vel_curlp%Re(:j,m,n)  &
                                - vel_p%Re(:j,m,n)*vel_curlt%Re(:j,m,n) )
            th%Re(:j,m,n) = d_Ro*(vel_p%Re(:j,m,n)*vel_curlr%Re(:j,m,n)  &
                                - vel_r%Re(:j,m,n)*vel_curlp%Re(:j,m,n) )
            ph%Re(:j,m,n) = d_Ro*(vel_r%Re(:j,m,n)*vel_curlt%Re(:j,m,n)  &
                                - vel_t%Re(:j,m,n)*vel_curlr%Re(:j,m,n) )
         end do
      end do
         			! coriolis  - z x u,  z=(cos,-sin,0)
      do n = 1, mes_oc%pN
         do m = 0, i_Ph-1
            r%Re (:j,m,n) = r%Re (:j,m,n) + lsin*vel_p%Re(:j,m,n)
            th%Re(:j,m,n) = th%Re(:j,m,n) + lgp *vel_p%Re(:j,m,n)
            ph%Re(:j,m,n) = ph%Re(:j,m,n) - (lgp*vel_t%Re(:j,m,n)  &
                                          + lsin*vel_r%Re(:j,m,n))
         end do
      end do

         			! lorentz
      do n = 1, mes_oc%pN
         do m = 0, i_Ph-1
            r%Re(:j,m,n)  = r%Re(:j,m,n)  &
               +  mag_curlt%Re(:j,m,n)*mag_p%Re(:j,m,n)  &
               -  mag_curlp%Re(:j,m,n)*mag_t%Re(:j,m,n) 
            th%Re(:j,m,n) = th%Re(:j,m,n)  &
               +  mag_curlp%Re(:j,m,n)*mag_r%Re(:j,m,n)  &
               -  mag_curlr%Re(:j,m,n)*mag_p%Re(:j,m,n)
            ph%Re(:j,m,n) = ph%Re(:j,m,n)  &
               +  mag_curlr%Re(:j,m,n)*mag_t%Re(:j,m,n)  &
               -  mag_curlt%Re(:j,m,n)*mag_r%Re(:j,m,n)
         end do
      end do

      call tra_rtp2qst(r,th,ph, sq,ss,st)
      call var_spec2coll(sq,cq, s2=ss,c2=cs, s3=st,c3=ct)
         			! codensity
      d = qRa * mes_oc%r(:,1)
      do nh = 0, cq%H%pH1
         do n = 1, mes_oc%N
            cq%Re(n,nh) = cq%Re(n,nh) + d(n)*cod_C%Re(n,nh)
            cq%Im(n,nh) = cq%Im(n,nh) + d(n)*cod_C%Im(n,nh)
         end do
      end do
                                ! r/(l(l+1)) \hat{r} . curl N
      call var_coll_qstllcurlr(ct, vel_NTor)
                                ! r/(l(l+1)) \hat{r} . curl curl N
      call var_coll_qstllcurlcurlr(cq,cs, vel_NPol)

      if(mpi_rnk==0) then
         vel_NTor%Re(:,0) = 0d0
         vel_NTor%Im(:,0) = 0d0
         vel_NPol%Re(:,0) = 0d0
         vel_NPol%Im(:,0) = 0d0
      end if
      
   end subroutine non_velocity
   

!-------------------------------------------------------------------------
!  nonlinear terms for the codensity.  C->NPol
!------------------------------------------------------------------------
   subroutine non_codensity()
      integer :: j, n
   
      call var_spec_init(mes_oc,var_H, ss)

         			! - u . gradC
      j = var_Th%pTh
      n = mes_oc%pN
      r%Re(:j,:,:n) =  &
         - vel_r%Re(:j,:,:n) * cod_gradr%Re(:j,:,:n)  &
         - vel_t%Re(:j,:,:n) * cod_gradt%Re(:j,:,:n)  &
         - vel_p%Re(:j,:,:n) * cod_gradp%Re(:j,:,:n)

      call tra_phys2spec(r, ss)
      call var_spec2coll(ss, cod_NC)
         			! internal source
      if(mpi_rnk==0)  cod_NC%Re(:,0) = cod_NC%Re(:,0) + cod_S
      
   end subroutine non_codensity
   

!-------------------------------------------------------------------------
!  nonlinear terms for magnetic field.  q,s->NTor, t->NPol
!-------------------------------------------------------------------------
   subroutine non_magnetic()
      integer :: nhm, j, n, m
      logical :: magbc
      double precision :: bcRe(0:i_H1), bcIm(0:i_H1), velph(i_pTh)

      call var_spec_init(mes_oc,var_H, sq)
      call var_spec_init(mes_oc,var_H, ss)
      call var_spec_init(mes_oc,var_H, st)

      magbc = (mes_ic%N/=1 .and. i_vel_bc>2)
         			! jump for stress-free cond inner core
      if(magbc) then
         r%Re (:,:,1) = 0d0
         th%Re(:,:,1) = mag_r%Re(:,:,1) * vel_t%Re(:,:,1)
         velph = rot_omega * mes_oc%r(1,1)  &
            * leg_sin(var_Th%pThi:var_Th%pThi+i_pTh-1)
         do m = 0, i_Ph-1 
            ph%Re(:,m,1) = mag_r%Re(:,m,1) * (vel_p%Re(:,m,1) - velph)
         end do
         n = mes_oc%N
         mes_oc%N = 1
         call tra_rtp2qst(r,th,ph, sq,ss,st)
         mes_oc%N = n
         bcRe = st%Re(:,1)
         bcIm = st%Im(:,1)
      end if
                                ! E = u x B
      j = var_Th%pTh
      do n = 1, mes_oc%pN
         do m = 0, i_Ph-1
            r%Re (:j,m,n) = vel_t%Re(:j,m,n) * mag_p%Re(:j,m,n)  &
                          - vel_p%Re(:j,m,n) * mag_t%Re(:j,m,n)
            th%Re(:j,m,n) = vel_p%Re(:j,m,n) * mag_r%Re(:j,m,n)  &
                          - vel_r%Re(:j,m,n) * mag_p%Re(:j,m,n)
            ph%Re(:j,m,n) = vel_r%Re(:j,m,n) * mag_t%Re(:j,m,n)  &
                          - vel_t%Re(:j,m,n) * mag_r%Re(:j,m,n)
         end do
      end do
      call tra_rtp2qst(r,th,ph, sq,ss,st)

      n = modulo(mpi_rnk,_Nr)      
      if(magbc .and. n==0) then  !NB no dr t in curl, curlcurl E
         st%Re(:,1) = bcRe
         st%Im(:,1) = bcIm
      end if
            
      call var_spec2coll(sq,cq, s2=ss,c2=cs, s3=st,c3=ct)
      
                                ! r/(l(l+1) \hat{r} . curl E
      call var_coll_qstllcurlr(ct, mag_NPol)
                                ! r/(l(l+1) \hat{r} . curl curl E
      call var_coll_qstllcurlcurlr(cq,cs, mag_NTor)
      
      if(magbc) then
         mag_bcRe(1,:) = - mag_NPol%Re(1,:)
         mag_bcIm(1,:) = - mag_NPol%Im(1,:)
      end if

               			! inner core
      call var_coll_copy(mag_icTor, ct)
      call var_coll_copy(mag_icPol, cq)
      n = mes_ic%N
      nhm = var_H%pH1
      if(b_mag_impose) then
         ct%Re(1:n,0:nhm) = ct%Re(1:n,0:nhm) + mag_icB0Tor%Re(1:n,0:nhm)
         ct%Im(1:n,0:nhm) = ct%Im(1:n,0:nhm) + mag_icB0Tor%Im(1:n,0:nhm)
         cq%Re(1:n,0:nhm) = cq%Re(1:n,0:nhm) + mag_icB0Pol%Re(1:n,0:nhm)
         cq%Im(1:n,0:nhm) = cq%Im(1:n,0:nhm) + mag_icB0Pol%Im(1:n,0:nhm)
      end if
      call var_coll_dph(ct, mag_icNTor)
      call var_coll_dph(cq, mag_icNPol)
      mag_icNTor%Re(1:n,0:nhm) = - rot_omega * mag_icNTor%Re(1:n,0:nhm)
      mag_icNTor%Im(1:n,0:nhm) = - rot_omega * mag_icNTor%Im(1:n,0:nhm)
      mag_icNPol%Re(1:n,0:nhm) = - rot_omega * mag_icNPol%Re(1:n,0:nhm)
      mag_icNPol%Im(1:n,0:nhm) = - rot_omega * mag_icNPol%Im(1:n,0:nhm)

      if(mpi_rnk==0) then
         mag_NTor%Re(:,0) = 0d0
         mag_NTor%Im(:,0) = 0d0
         mag_NPol%Re(:,0) = 0d0
         mag_NPol%Im(:,0) = 0d0
         mag_icNTor%Re(:,0) = 0d0
         mag_icNTor%Im(:,0) = 0d0
         mag_icNPol%Re(:,0) = 0d0
         mag_icNPol%Im(:,0) = 0d0
      end if

   end subroutine non_magnetic


!-------------------------------------------------------------------------
!  inner core rotation, torques
!------------------------------------------------------------------------
   subroutine non_rotation()
      double precision :: S, vfac, mfac, lsin(i_pTh), d(2)
      integer :: m, n
      
      n = modulo(mpi_rnk,_Nr)
      if(n==0) then
         lsin = leg_sin(var_Th%pThi:var_Th%pThi+i_pTh-1)
         vfac = mes_oc%r(1,3)*d_E/d_Ro
         mfac = mes_oc%r(1,3)/d_Ro      
         			! viscous torque
         if(i_vel_bc>2) then
            rot_velTorq = 0d0
         else
            S = - 2d0/mes_oc%r(1,1)
            do m = 0, i_Ph-1
               r%Re(:,m,1) =  &
                  (S*vel_p%Re(:,m,1)-vel_curlt%Re(:,m,1)) * lsin
            end do
            call tra_surfintegral(r%Re(1,0,1), S)
            rot_velTorq = S * vfac
         end if
         			! lorentz torque
         if(mes_ic%N==1) then
            rot_magTorq = 0d0
         else
            do m = 0, i_Ph-1
               r%Re(:,m,1) = mag_r%Re(:,m,1) * mag_p%Re(:,m,1) * lsin
            end do
            call tra_surfintegral(r%Re(1,0,1), S)
            rot_magTorq = S * mfac
         end if
      end if

#ifdef _MPI
      if(mpi_rnk==0) d(1) = rot_velTorq
      if(mpi_rnk==0) d(2) = rot_magTorq      
      call mpi_bcast(d, 2, mpi_double_precision,  &
         0, mpi_comm_world, mpi_er)
      rot_velTorq = d(1)
      rot_magTorq = d(2)
#endif      
      
   end subroutine non_rotation
   

!------------------------------------------------------------------------
!  get cfl max dt
!------------------------------------------------------------------------
   subroutine non_maxtstep()
      double precision :: dr, mx, r_(0:i_pN+1)
      double precision :: d, p(i_pTh, 0:i_Ph-1)
      integer :: N_,n, n__, NTh

      N_ = mes_oc%pN
      do n = 0, N_+1
         n__ = n+mes_oc%pNi-1
         if(n__>=1 .and. n__<=mes_oc%N)  r_(n) = mes_oc%r(n__,1)
      end do
      
      Nth = var_Th%pTh
      tim_cfl_dt = 1d99
         				! local
      do n = 1, N_

         if(n==1 .and. modulo(mpi_rnk,_Nr)==0) then
            dr = r_(2) - r_(1)
         else if(n==N_ .and. modulo(mpi_rnk,_Nr)==_Nr-1) then
            dr = r_(N_) - r_(N_-1)
         else
            dr = min( r_(n)-r_(n-1), r_(n+1)-r_(n) )
         end if
         d = (d_E+d_Ro)/(2d0*dr)
         d = d*d
         p = mag_r%Re(:,:,n) * mag_r%Re(:,:,n)
         p = p/dsqrt(p*d_Ro+d) + dabs(vel_r%Re(:,:,n)) 
         mx = maxval( p(:Nth,:) )
         if(mx/=0d0) tim_cfl_dt = min( tim_cfl_dt, dr/mx )

         dr = r_(n)/dsqrt(dble(i_L*i_L1))
         d = (d_E+d_Ro)/(2d0*dr)
         d = d*d
         p =   mag_t%Re(:,:,n)*mag_t%Re(:,:,n)  &
            +  mag_p%Re(:,:,n)*mag_p%Re(:,:,n)
         p = p/dsqrt(p*d_Ro+d)  &
            +  dsqrt( vel_t%Re(:,:,n)*vel_t%Re(:,:,n)  &
            +         vel_p%Re(:,:,n)*vel_p%Re(:,:,n) )
         mx = maxval( p(:Nth,:) )
         if(mx/=0d0) tim_cfl_dt = min( tim_cfl_dt, dr/mx )
      end do
         				! global
      tim_cfl_dt = min(tim_cfl_dt, d_Ro)
      tim_cfl_dt = min(tim_cfl_dt, dsqrt(d_E))
                                        
#ifdef _MPI
      call mpi_allreduce(tim_cfl_dt, mx, 1, mpi_double_precision,  &
         mpi_min, mpi_comm_world, mpi_er)
      tim_cfl_dt = mx
#endif

   end subroutine non_maxtstep


!*************************************************************************
 end module nonlinear
!*************************************************************************

