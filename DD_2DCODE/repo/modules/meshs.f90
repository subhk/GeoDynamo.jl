!**************************************************************************
! Definition of mesh-types.
!
! Public variables:
!
!   r(n,k)              radial points r_j^k, j in [1,N]
!
!   dr(p)%M             p-th derivative of f = dr(p)%M . f
!
!   radLap%M            radial part of laplacian  drr + (2/r) dr
!
!
! (mesh)-type matricies are stored s.t.   M(   KL+1+n-j, j) = A(n,j)
! (lumesh)-type matricies are stored s.t. M( 2*KL+1+n-j, j) = A(n,j)
!
! Note that the index n is usually used to denote r_n and the operator on
! the n-th point corresponds to the n-th row of the matrix.  The index
! j usually denotes the column and is therefore in the outer loop below.
! Matrix multiplication  y := M . x  can be achieved accessing M only
! once along the memory by the loop form
!
!   y = 0
!   do j = 1, N
!      do n = max(1,j-KL), min(j+KL,N)
!         y(n) = y(n) + M(KL+1+n-j,j) * x(j)
!
!
!**************************************************************************
#include "../parallel.h"
 module meshs
!**************************************************************************
   use mpi
   use parameters
   implicit none
   save
                                ! M(KL+1+n-j, j) = A(n,j)
   type mesh
      double precision :: M(2*i_KL+1, i_N)
   end type mesh
                                ! M(2*KL+1+n-j, j) = A(n,j)
                                ! see lapack dgbtrf
   type lumesh
      integer          :: ipiv(i_N)
      double precision :: M(3*i_KL+1, i_N)
   end type lumesh

   type iclumesh
      integer          :: ipiv(i_Nic+i_N-1)
      double precision :: M(3*i_KL+1, i_Nic+i_N-1)
   end type iclumesh

   integer, parameter  :: i_rpowmax =  3
   integer, parameter  :: i_rpowmin = -3

   type rdom
      integer          :: N, pNi,pN, pNi_(0:_Nr-1),pN_(0:_Nr-1)
      double precision :: r(i_N,i_rpowmin:i_rpowmax)
      double precision :: intr2dr(i_N)
      double precision :: intr2drh(i_N,0:2*i_KL)
      type (mesh)      :: dr(i_KL)
      type (mesh)      :: radLap
      type (mesh)      :: r1dr   
   end type rdom


   type (rdom) :: mes_oc
   type (rdom) :: mes_ic

 contains


!------------------------------------------------------------------------
!  Precompute the mesh stuff
!------------------------------------------------------------------------
   subroutine mes_precompute()
      double precision :: ri
      integer :: n, N_
      
         			! get collocation pts
                                ! N extrema of T_{N-1}(x)
                                ! Numerical Recipes 5.8.5
                                
                                ! outer core
      if(_Nr>i_N) stop '_Nr>i_N'
      mes_oc%N   = i_N
      do n = 1, i_N
         mes_oc%r(n,1) = 0.5d0 * ( 1d0 + dcos(d_PI*(i_N-n)/dble(i_N-1)) )
      end do      
      ri  = d_rratio/(1d0-d_rratio)
      mes_oc%r(:,1) = mes_oc%r(:,1) + ri
      call mes_rdom_init(mes_oc)
         			! inner core
      mes_ic%N = i_Nic
      N_ = 2*i_Nic
      do n = i_Nic+1, N_
         mes_ic%r(n-i_Nic,1) =  ri * dcos(d_PI*(N_-n)/dble(N_-1))
      end do
      call mes_rdom_init(mes_ic)

   end subroutine mes_precompute


!------------------------------------------------------------------------
! for the radial domain D, first set the number of points D%N and their
! values D%r, then mes_rdom_init() will fill in the rest...
!------------------------------------------------------------------------
   subroutine mes_rdom_init(D)
      type (rdom), intent(inout) :: D
      integer, parameter :: lda = 2*i_KL+1
      double precision   :: A(lda,lda), c,e,w
      integer :: n,i,j,nn, l,r,p
! WD 03/09/2013, as for initialisation of harm type and the distribution 
!over the cores, also the radial domain needs some update
      D%pN_ = 0
      do n = 1, D%N
         r = _Nr - modulo(n-1,_Nr) - 1
         D%pN_(r) = D%pN_(r) + 1
      end do
      D%pNi_(0) = 1
      do r = 1, _Nr-1
         D%pNi_(r) = D%pNi_(r-1) + D%pN_(r-1)
      end do

!      D%pNi_(0) = 1
!      n = modulo(i_N,i_pN)
!      if(n==0) D%pN_(0) = i_pN
!      if(n/=0) D%pN_(0) = n
!      do p = 1, _Nr-1
!         D%pNi_(p) = D%pNi_(p-1) + D%pN_(p-1)
!         D%pN_ (p) = i_pN
!      end do

      p = modulo(mpi_rnk,_Nr)
      D%pN  = D%pN_ (p)
      D%pNi = D%pNi_(p)
         			! get ad_r(:,j) = r^j
      do j = i_rpowmin, i_rpowmax
         if(j==1) cycle
         D%r(1:D%N,j) = D%r(1:D%N,1)**j
      end do

                                ! get finite difference coeffs    
      if(i_KL<2) stop 'KL<2'
      do i = 1, i_KL
         D%dr(i)%M = 0d0
      end do
      D%radLap%M = 0d0
      D%r1dr%M   = 0d0   
      D%intr2dr  = 0d0
      D%intr2drh = 0d0
      if(D%N==1) return

      do n = 1, D%N
         l = max(1,n-i_KL)
         r = min(n+i_KL,D%N)
         nn = r-l+1
         do i = 1, nn
            A(i,1) = 1d0
         end do
         do j = 2, nn
            do i = 1, nn
               A(i,j) = A(i,j-1) * (D%r(l+i-1,1)-D%r(n,1)) / dble(j-1)
            end do
         end do
         call mes_mat_invert(nn,A,lda)
         do i = 1, i_KL
            do j = l, r
               D%dr(i)%M(i_KL+1+n-j,j) = A(i+1,j-l+1)
            end do
         end do
            			! weights for integration r^2 dr
         c = 1d0
         e = 1d0
         do i = 1, nn
            c = c * (D%r(min(n+1,D%N),1)-D%r(n,1)) / dble(i)
            e = e * (D%r(max(  1,n-1),1)-D%r(n,1)) / dble(i)
            do j = l, r
               w = 0.5d0*(c-e)*A(i,j-l+1)*D%r(n,2)
               D%intr2dr (j)     = D%intr2dr (j)     + w
               D%intr2drh(j,i-1) = D%intr2drh(j,i-1) + w
            end do
         end do
      end do
                                ! get  drr + (2/r) dr
      do j = 1, D%N
         l = max(1,j-i_KL)
         r = min(j+i_KL,D%N)
         do n = l, r
            D%radLap%M(i_KL+1+n-j,j)  &
               = D%dr(2)%M(i_KL+1+n-j,j)  &
               + D%dr(1)%M(i_KL+1+n-j,j) * 2d0 * D%r(n,-1)
         end do
      end do
                                ! get  (1/r + dr)
      D%r1dr%M = D%dr(1)%M
      D%r1dr%M(i_KL+1,:) = D%r1dr%M(i_KL+1,:) + D%r(:,-1)

   end subroutine mes_rdom_init



!------------------------------------------------------------------------
!  Replace nxn matrix A by its inverse
!------------------------------------------------------------------------
   subroutine mes_mat_invert(n,A,lda)
      integer, intent(in) :: n, lda
      double precision, intent(inout) :: A(lda,n)
      double precision :: work(n)
      integer :: info, ipiv(n)
   
      call dgetrf(n, n, A, lda, ipiv, info)
      if(info /= 0) stop 'matrix inversion error 1'

      call dgetri(n, A, lda, ipiv, work, n, info)
      if(info /= 0) stop 'matrix inversion error 2'

   end subroutine mes_mat_invert


!------------------------------------------------------------------------
!  Replace A with its LU factorisation
!  uses lapack routine dgbtrf()
!------------------------------------------------------------------------
   subroutine mes_lu_find(D,A)
      type (rdom),   intent(in)    :: D
      type (lumesh), intent(inout) :: A
      integer :: info

      call dgbtrf(D%N, D%N, i_KL, i_KL, A%M, 3*i_KL+1, A%ipiv, info)
!     print *, info
      if(info /= 0) stop 'mes_lu_find'

   end subroutine mes_lu_find


!------------------------------------------------------------------------
!  Replace A with its LU factorisation
!  uses lapack routine dgbtrf()
!------------------------------------------------------------------------
   subroutine mes_iclu_find(A)
      type (iclumesh), intent(inout) :: A
      integer :: info, N_
      
      N_ = mes_ic%N+mes_oc%N-1
      call dgbtrf(N_, N_, i_KL, i_KL, A%M, 3*i_KL+1, A%ipiv, info)
      if(info /= 0) stop 'mes_iclu_find'

   end subroutine mes_iclu_find


!**************************************************************************
 end module meshs
!**************************************************************************
