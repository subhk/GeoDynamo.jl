!*************************************************************************
! Transformation using the FFT, and Gauss Pts.
! De-aliased.
! Written for FFTW version 3 -- documentation at  www.fftw.org
! See FFTW reference, sections 2.3, 4.3.3, 4.4.1-2
!
! \sum_{-Ma}^{Ma} == \sum_0^{2Ma-1}
! logical size of DFT n==2Ma=Ph, 
! fftw complex array has n/2+1 elements => Ma+1 elements, or index 0:Ma 
! fftw real array 2*(n/2+1) elts, only first n accessed => index 0:Ph-1
! 
! fft at each theta => howmany=Th
! theta along first index of X,Y => dist=1, stride=Th
!
!*************************************************************************
#include "../parallel.h"
 module transform
!*************************************************************************
   use parameters
   use legendre
   use variables
   implicit none
   save
      					! see fftw3.f
   integer, parameter, private :: FFTW_ESTIMATE=64
   integer, parameter, private :: FFTW_PATIENT =32
   double precision,   private :: a2d_X(i_pTh,0:i_Ph-1)
   double complex,     private :: a2c_Y(i_pTh,0:i_Ma)
   double precision,   private :: a3d_X(i_pTh,0:i_Ph-1,  i_pN,3)
   double complex,     private :: a3c_Y(i_Th, 0:2*_pHc-1,i_pN,3)
   double complex,     private :: a3c_T(i_pTh,0:i_M1,    i_pN,3)
   integer*8,          private :: plan_s2r, plan_r2s


 contains


!------------------------------------------------------------------------
! Setup plans for multiple transforms.  
!    priority = 0   Quick plan, slower ffts -- for only a few ffts
!    priority = 1   Slow plan, quicker ffts -- for many ffts.
!------------------------------------------------------------------------
   subroutine tra_precompute(priority)
      integer, intent(in) :: priority
      integer :: flag, n(1), howmany, inembed(1), onembed(1)
      
      if(i_M>1 .and. modulo(i_M,2)==1) stop 'if(M>1) set M even'
      
      if(priority.eq.0) flag = FFTW_ESTIMATE
      if(priority.eq.1) flag = FFTW_PATIENT

      n = (/i_Ph/)
      howmany = var_Th%pTh
      inembed = (/i_pTh*(i_Ma+1)/)
      onembed = (/i_pTh*i_Ph/)
      call dfftw_plan_many_dft_c2r(plan_s2r, 1, n, howmany,  &
                                   a2c_Y, inembed, i_pTh, 1,  &
                                   a2d_X, onembed, i_pTh, 1,  &
                                   flag)
      n = (/i_Ph/)
      howmany = var_Th%pTh
      inembed = (/i_pTh*i_Ph/)
      onembed = (/i_pTh*(i_Ma+1)/)
      call dfftw_plan_many_dft_r2c(plan_r2s, 1, n, howmany,  &
                                   a2d_X, inembed, i_pTh, 1,  &
                                   a2c_Y, onembed, i_pTh, 1,  &
                                   flag)
   end subroutine tra_precompute


!------------------------------------------------------------------------
!  transpose theta_j and m-modes then fft
!------------------------------------------------------------------------
   subroutine tra_s2p(s,ns)
      type (spec), intent(in) :: s
      integer,     intent(in) :: ns
      integer :: n, m, mp, i
#ifndef _MPI
      do i = 1, ns
         do n = 1, s%D%pN
            do m = 0, var_Th%pM-1
               mp = var_Th%m(m,0)
               a2c_Y(:,mp) = a3c_Y(:,m,n,i)
            end do
            a2c_Y(:,i_M:) = 0d0
            call dfftw_execute(plan_s2r)
            a3d_X(:,:,n,i) = a2d_X
         end do
      end do
#else
      double precision :: bsend(2*i_pN*i_pTh*(2*_pHc)*3,0:_Nth-1)
      double precision :: brecv(2*i_pN*i_pTh*(2*_pHc)*3,0:_Nth-1)
      integer :: stp, dst,src, l,j, rnk,rko
         					! transpose
      rnk = mpi_rnk/_Nr
      rko = modulo(mpi_rnk,_Nr)
      
      do stp = 0, _Nth-1
         src  = modulo(_Nth-stp+rnk, _Nth)         
         mpi_tg = rko + stp*_Nr
         n = 2*s%D%pN*var_Th%pTh*var_Th%pM_(src)*ns
         call mpi_irecv( brecv(1,stp), n, mpi_double_precision,  &
            rko+src*_Nr, mpi_tg, mpi_comm_world, mpi_rq(stp), mpi_er)
      end do

      do stp = 0, _Nth-1
         dst  = modulo(stp+rnk, _Nth)
         src  = modulo(_Nth-stp+rnk, _Nth)         
         l = 1
         do i = 1, ns
           do n = 1, s%D%pN
             do m = 0, var_Th%pM-1
               do j = var_Th%pThi_(dst), var_Th%pThi_(dst)+var_Th%pTh_(dst)-1
                  bsend(l,  stp) =  dble(a3c_Y(j,m,n,i))
                  bsend(l+1,stp) = dimag(a3c_Y(j,m,n,i))
                  l = l + 2
               end do
             end do
           end do
         end do
         mpi_tg = rko + stp*_Nr
         n = 2*s%D%pN*var_Th%pTh_(dst)*var_Th%pM*ns
         call mpi_isend( bsend(1,stp), n, mpi_double_precision,  &
            rko+dst*_Nr, mpi_tg, mpi_comm_world, mpi_rq(_Nth+stp), mpi_er)
      end do
      
      do stp = 0, _Nth-1
         src  = modulo(_Nth-stp+rnk, _Nth)      
         call mpi_wait( mpi_rq(stp), mpi_st, mpi_er)
         l = 1
         do i = 1, ns
           do n = 1, s%D%pN
             do m = 0, var_Th%pM_(src)-1
               mp = var_Th%m(m,src)
               do j = 1, var_Th%pTh
                  a3c_T(j,mp,n,i) = dcmplx(brecv(l,stp),brecv(l+1,stp))
                  l = l + 2
               end do
             end do
           end do
         end do
      end do

      do stp = 0, _Nth-1
         call mpi_wait( mpi_rq(_Nth+stp), mpi_st, mpi_er)
      end do
         					! FFT
      do i = 1, ns
         do n = 1, s%D%pN
            a2c_Y(:,:i_M1) = a3c_T(:,:,n,i)
            a2c_Y(:,i_M:)  = 0d0
            call dfftw_execute(plan_s2r)
            a3d_X(:,:,n,i) = a2d_X
         end do
      end do
#endif      

   end subroutine tra_s2p


!------------------------------------------------------------------------
!  fft then transpose theta_j and m-modes
!------------------------------------------------------------------------
   subroutine tra_p2s(s,ns)
      type (spec), intent(in) :: s
      integer,     intent(in) :: ns
      double precision :: scale
      integer :: n, m, mp, i
         			! scale, FFTW section 4.7.2
#ifndef _MPI
      scale = 1d0 / dble(i_Ph)
      do i = 1, ns
         do n = 1, s%D%pN
            a2d_X = a3d_X(:,:,n,i) * scale
            call dfftw_execute(plan_r2s)
            do m = 0, var_Th%pM-1
               mp = var_Th%m(m,0)
               a3c_Y(:,m,n,i) = a2c_Y(:,mp)            
            end do
         end do
      end do
#else
      double precision :: bsend(2*i_pN*i_pTh*(2*_pHc)*3,0:_Nth-1)
      double precision :: brecv(2*i_pN*i_pTh*(2*_pHc)*3,0:_Nth-1)
      integer :: stp, dst,src, l,j, rnk,rko
         					! FFT
      scale = 1d0 / dble(i_Ph)
      do i = 1, ns
         do n = 1, s%D%pN
            a2d_X = a3d_X(:,:,n,i) * scale
            call dfftw_execute(plan_r2s)
            a3c_T(:,:,n,i) = a2c_Y(:,:i_M1)
         end do
      end do
         					! transpose
      rnk = mpi_rnk/_Nr
      rko = modulo(mpi_rnk,_Nr)
      
      do stp = 0, _Nth-1
         src  = modulo(_Nth-stp+rnk, _Nth)         
         mpi_tg = rko + stp*_Nr
         n = 2*s%D%pN*var_Th%pTh_(src)*var_Th%pM*ns
         call mpi_irecv( brecv(1,stp), n, mpi_double_precision,  &
            rko+src*_Nr, mpi_tg, mpi_comm_world, mpi_rq(stp), mpi_er)
      end do

      do stp = 0, _Nth-1
         dst  = modulo(stp+rnk, _Nth)
         l = 1
         do i = 1, ns
           do n = 1, s%D%pN
             do m = 0, var_Th%pM_(dst)-1
               mp = var_Th%m(m,dst)
               do j = 1, var_Th%pTh
                  bsend(l,  stp) =  dble(a3c_T(j,mp,n,i))
                  bsend(l+1,stp) = dimag(a3c_T(j,mp,n,i))
                  l = l + 2
               end do
             end do
           end do
         end do
         mpi_tg = rko + stp*_Nr
         n = 2*s%D%pN*var_Th%pTh*var_Th%pM_(dst)*ns
         call mpi_isend( bsend(1,stp), n, mpi_double_precision,  &
            rko+dst*_Nr, mpi_tg, mpi_comm_world, mpi_rq(_Nth+stp), mpi_er)
      end do
      
      do stp = 0, _Nth-1
         src  = modulo(_Nth-stp+rnk, _Nth)      
         call mpi_wait( mpi_rq(stp), mpi_st, mpi_er)
         l = 1
         do i = 1, ns
           do n = 1, s%D%pN
             do m = 0, var_Th%pM-1
               do j = var_Th%pThi_(src), var_Th%pThi_(src)+var_Th%pTh_(src)-1
                  a3c_Y(j,m,n,i) = dcmplx(brecv(l,stp),brecv(l+1,stp))
                  l = l + 2
               end do
             end do
           end do
         end do
      end do

      do stp = 0, _Nth-1
         call mpi_wait( mpi_rq(_Nth+stp), mpi_st, mpi_er)
      end do
#endif
         
   end subroutine tra_p2s


!------------------------------------------------------------------------
!  Convert spectral to real space
!------------------------------------------------------------------------
   subroutine tra_spec2phys(s, p)
      type (spec), intent(in)  :: s
      type (phys), intent(out) :: p
      double complex :: c
      integer :: n
      _loop_sumth_vars      
      				! for each x_n ...
      do n = 1, s%D%pN
         			! sum over l at each th_j -> Y_jm
         a3c_Y(:,:,n,1) = 0d0
         _loop_sumth_begin
            c = dcmplx( s%Re(nh,n), s%Im(nh,n) )
            a3c_Y(:,m,n,1) = a3c_Y(:,m,n,1)  +  c * leg_P(:,nh)
         _loop_sumth_end
      end do
            			! fft
      call tra_s2p(s,1)
      p%Re = a3d_X(:,:,:,1)
   
   end subroutine tra_spec2phys


!------------------------------------------------------------------------
!  Convert real to spectral space
!  Ensure output s%L and s%M are set before call.
!------------------------------------------------------------------------
   subroutine tra_phys2spec(p, s)
      type (phys), intent (in  )  :: p
      type (spec), intent (inout) :: s
      integer :: n
      _loop_sumth_vars
            			! fft
      a3d_X(:,:,:,1) = p%Re
      call tra_p2s(s,1)
      				! for each x_n ...
      do n = 1, s%D%pN
            			! integrate over theta, Gauss weights
         _loop_sumth_begin
            s%Re(nh,n) = sum( dble(a3c_Y(:,m,n,1))*leg_gwP(:,nh))
            s%Im(nh,n) = sum(dimag(a3c_Y(:,m,n,1))*leg_gwP(:,nh))
            if(m_==0) s%Im(nh,n) = 0d0
         _loop_sumth_end
      end do
         
   end subroutine tra_phys2spec


!------------------------------------------------------------------------
!  Spectral to real space, qst -> r,th,ph
!------------------------------------------------------------------------
   subroutine tra_qst2rtp(q,s,t, r,th,ph)
      type (spec), intent(in)  :: q,s,t
      type (phys), intent(out) :: r,th,ph
      double complex :: c1,c2
      integer :: n
      _loop_sumth_vars      
         			! Ar = q
      do n = 1, s%D%pN
         a3c_Y(:,:,n,1) = 0d0
         _loop_sumth_begin
            c1 = dcmplx( q%Re(nh,n), q%Im(nh,n) )
            a3c_Y(:,m,n,1) = a3c_Y(:,m,n,1)  +  c1 * leg_P(:,nh)
         _loop_sumth_end
      end do      
                                ! At = s dth - t/sin dph
      do n = 1, s%D%pN
         a3c_Y(:,:,n,2) = 0d0
         _loop_sumth_begin
            c1 = dcmplx( s%Re(nh,n),  s%Im(nh,n) )
            c2 = dcmplx( t%Im(nh,n), -t%Re(nh,n) )
            a3c_Y(:,m,n,2) = a3c_Y(:,m,n,2) + c1 * leg_llP_(:,nh)  &
                                        + c2 * leg_llsin1mP(:,nh)
         _loop_sumth_end
      end do
                                ! Aph = s/sin dph + t dth
      do n = 1, s%D%pN
         a3c_Y(:,:,n,3) = 0d0
         _loop_sumth_begin
            c1 = dcmplx( -s%Im(nh,n), s%Re(nh,n) )
            c2 = dcmplx(  t%Re(nh,n), t%Im(nh,n) )
            a3c_Y(:,m,n,3) = a3c_Y(:,m,n,3) + c1 * leg_llsin1mP(:,nh)  &
                                        + c2 * leg_llP_(:,nh)
         _loop_sumth_end
      end do

      call tra_s2p(s,3)
      r%Re  = a3d_X(:,:,:,1)
      th%Re = a3d_X(:,:,:,2)
      ph%Re = a3d_X(:,:,:,3)
   
   end subroutine tra_qst2rtp


!------------------------------------------------------------------------
!  Real to spectral space, r,th,ph -> q,s,t
!  Ensure output q,s,t%L and q,s,t%M are set before call.
!------------------------------------------------------------------------
   subroutine tra_rtp2qst(r,th,ph, q,s,t)
      type (phys), intent(in)    :: r,th,ph
      type (spec), intent(inout) :: q,s,t
      integer :: n
      _loop_sumth_vars

      a3d_X(:,:,:,1) = r%Re
      a3d_X(:,:,:,2) = th%Re
      a3d_X(:,:,:,3) = ph%Re
      call tra_p2s(s,3)      
         			! q  integrate q.r = q
      do n = 1, s%D%pN
         _loop_sumth_begin
            q%Re(nh,n) = sum( dble(a3c_Y(:,m,n,1))*leg_gwP(:,nh))
            q%Im(nh,n) = sum(dimag(a3c_Y(:,m,n,1))*leg_gwP(:,nh))
            if(m_==0) q%Im(nh,n) = 0d0
         _loop_sumth_end
      end do
         
      do n = 1, s%D%pN
         _loop_sumth_begin
         			! s.th = dth P
            s%Re(nh,n) =  sum( dble(a3c_Y(:,m,n,2))*leg_gwllP_(:,nh))
            s%Im(nh,n) =  sum(dimag(a3c_Y(:,m,n,2))*leg_gwllP_(:,nh))
            			! t.th = - 1/sin(th) dph P
            t%Re(nh,n) = -sum(dimag(a3c_Y(:,m,n,2))*leg_gwllsin1mP(:,nh))
            t%Im(nh,n) =  sum( dble(a3c_Y(:,m,n,2))*leg_gwllsin1mP(:,nh))
         _loop_sumth_end
      end do

      do n = 1, s%D%pN
         _loop_sumth_begin
         			! s.ph = 1/sin(th) dph P
            s%Re(nh,n) =  &
               s%Re(nh,n) + sum(dimag(a3c_Y(:,m,n,3))*leg_gwllsin1mP(:,nh))
            s%Im(nh,n) =  &
               s%Im(nh,n) - sum( dble(a3c_Y(:,m,n,3))*leg_gwllsin1mP(:,nh))
            if(m_==0) s%Im(nh,n) = 0d0
            			! t.ph = dth P
            t%Re(nh,n) =  &
               t%Re(nh,n) + sum( dble(a3c_Y(:,m,n,3))*leg_gwllP_(:,nh))
            t%Im(nh,n) =  &
               t%Im(nh,n) + sum(dimag(a3c_Y(:,m,n,3))*leg_gwllP_(:,nh))
            if(m_==0) t%Im(nh,n) = 0d0
         _loop_sumth_end
      end do

   end subroutine tra_rtp2qst


!------------------------------------------------------------------------
!  Spectral to real space, A -> (grad A)t,p,  th,ph components only
!------------------------------------------------------------------------
   subroutine tra_grad(s, t,p)
      type (spec), intent(in)  :: s
      type (phys), intent(out) :: t,p
      double precision :: r1(s%D%pN)
      double complex :: c
      integer :: n
      _loop_sumth_vars      
      
      do n = 1, s%D%pN
         r1(n) = s%D%r(n+s%D%pNi-1,-1)
      end do
                                ! Ath = 1/r dth a
      do n = 1, s%D%pN
         a3c_Y(:,:,n,1) = 0d0
         _loop_sumth_begin
            c = dcmplx( s%Re(nh,n), s%Im(nh,n) )
            c = c * r1(n)
            a3c_Y(:,m,n,1) = a3c_Y(:,m,n,1)  +  c * leg_P_(:,nh)
         _loop_sumth_end
      end do
                                ! Aph = 1/rsin dph a
      do n = 1, s%D%pN
         a3c_Y(:,:,n,2) = 0d0
         _loop_sumth_begin
            c = dcmplx( -s%Im(nh,n), s%Re(nh,n) )
            c = c * r1(n)
            a3c_Y(:,m,n,2) = a3c_Y(:,m,n,2)  +  c * leg_sin1mP(:,nh)
         _loop_sumth_end
      end do

      call tra_s2p(s,2)
      t%Re = a3d_X(:,:,:,1)
      p%Re = a3d_X(:,:,:,2)     

   end subroutine tra_grad


!------------------------------------------------------------------------
!  theta derivative
!------------------------------------------------------------------------
   subroutine tra_dth(s, p)
      type (spec), intent(in)  :: s
      type (phys), intent(out) :: p
      double complex :: c
      integer :: n
      _loop_sumth_vars      
      
      do n = 1, s%D%pN
         a3c_Y(:,:,n,1) = 0d0
         _loop_sumth_begin
            c = dcmplx( s%Re(nh,n), s%Im(nh,n) )
            a3c_Y(:,m,n,1) = a3c_Y(:,m,n,1)  +  c * leg_P_(:,nh)
         _loop_sumth_end
      end do
      call tra_s2p(s,1)
      p%Re = a3d_X(:,:,:,1)     
   
   end subroutine tra_dth


!------------------------------------------------------------------------
! Surface integral.  \int_0^2pi \int_0^pi A sin(th) dth dph = 4 pi A00
!------------------------------------------------------------------------
   subroutine tra_surfintegral(p, S)
      double precision, intent(in)  :: p(i_pTh, 0:i_Ph-1)
      double precision, intent(out) :: S
      double precision :: Y(i_pTh), S_
      integer :: m, j, r

      j = var_Th%pTh
      Y = 0d0
      do m = 0, i_Ph-1
         Y(:j) = Y(:j) + p(:j,m)
      end do
      S = dot_product( Y(:j) , leg_gwP0(var_Th%pThi:var_Th%pThi+j-1) )
      S = S * 4d0*d_PI / dble(i_Ph)
#ifdef _MPI
      if(mpi_rnk==0) then
         do r = 1, _Nth-1
            mpi_tg = r*_Nr
            call mpi_recv(S_, 1, mpi_double_precision,  &
               r*_Nr, mpi_tg, mpi_comm_world, mpi_st, mpi_er)
            S = S + S_
         end do
      else
         mpi_tg = mpi_rnk
         call mpi_send(S, 1, mpi_double_precision,  &
            0, mpi_tg, mpi_comm_world, mpi_er)
      end if
#endif
   end subroutine tra_surfintegral


!*************************************************************************
 end module transform
!*************************************************************************
