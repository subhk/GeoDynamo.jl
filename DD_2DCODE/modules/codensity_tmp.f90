!*************************************************************************
!
!
!*************************************************************************
 module codensity
!*************************************************************************
   use variables
   use timestep
   implicit none
   save

   type (phys) :: cod_gradr
   type (phys) :: cod_gradt
   type (phys) :: cod_gradp
   type (coll) :: cod_C
   type (coll) :: cod_NC
   
   double precision :: cod_S(1:i_N)
   double precision :: cod_bcRe(2,0:i_pH1)
   double precision :: cod_bcIm(2,0:i_pH1)

   type (coll), private :: C
   type (coll), private :: NC
   
   type (lumesh), private :: XC(0:i_L1)
   type (mesh),   private :: YC(0:i_L1)

   type (coll), private :: cC
   
 contains

!------------------------------------------------------------------------
!  initialise codensity
!------------------------------------------------------------------------
   subroutine cod_precompute()
      call var_coll_init(mes_oc,var_H, cod_C)
      cod_S    = 0d0
      cod_BCRe = 0d0
      cod_BCIm = 0d0
   end subroutine cod_precompute


!------------------------------------------------------------------------
!  precomputation of codensity timestepping matrices
!------------------------------------------------------------------------
   subroutine cod_matrices()
      integer :: l

      do l = 0, i_L1
         call tim_lumesh_X      ( mes_oc, 1d0, d_q, l, XC(l) )
         call cod_bc_C          ( mes_oc, XC(l), l)
         call mes_lu_find       ( mes_oc, XC(l) )
         call tim_mesh_Y        ( mes_oc, 1d0, d_q, l, YC(l) )
      end do

   end subroutine cod_matrices


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
! i_cod_bc = 1  const C on ICB and CMB
!          = 2  const C on ICB, flux on CMB
!          = 3  const flux on ICB, C on CMB
!          = 4  const flux on ICB and CMB
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

   subroutine cod_bc_C(D,A,l)
      type (rdom),   intent(in)  :: D
      type (lumesh), intent(out) :: A
      integer,       intent(in)  :: l
      integer :: j

      if(i_cod_bc==1 .or. i_cod_bc==2) then	
         A%M(2*i_KL+1,1)   = 1d0
      else
         do j = 1, 1+i_KL
            A%M(2*i_KL+1+1-j,j) = D%dr(1)%M(i_KL+1+1-j,j)
         end do
      end if
      
      if(i_cod_bc==1 .or. i_cod_bc==3) then	
         A%M(2*i_KL+1,D%N) = 1d0
      else
         do j = D%N-i_KL, D%N
            A%M(2*i_KL+1+D%N-j,j) = D%dr(1)%M(i_KL+1+D%N-j,j)
         end do
      end if
! # Rob (14/02/12): Fix temperature so that the code knows what the temperature is when both boundaries are fixed flux
! This is done by setting the average (i.e. l=0) mode as a BC on T rather than on dT/dr
      if((i_cod_bc==4) .and. (l==0)) then
!         A%M(2*i_KL+1,D%N)   = 1d0
         do j = 1, 1+i_KL
            A%M(2*i_KL+1+1-j,j) = 0d0
         end do
         A%M(2*i_KL+1,1)   = 1d0
      endif
! ###
      
   end subroutine cod_bc_C


!-------------------------------------------------------------------------
!  set the RHS for the boundary condition = 0
!-------------------------------------------------------------------------
   subroutine cod_setbc(a)
      type (coll), intent(inout) :: a
      integer :: nh, n
      n = a%D%N
      do nh = 0, a%H%pH1
         a%Re( 1, nh ) = cod_bcRe(1,nh)
         a%Re( n, nh ) = cod_bcRe(2,nh)
         a%Im( 1, nh ) = cod_bcIm(1,nh)
         a%Im( n, nh ) = cod_bcIm(2,nh)
      end do
   end subroutine cod_setbc


!-------------------------------------------------------------------------
!  find flux at outer boundary for no flow:  dr C  evaluated at ro, 
!  where C solves   - q ( drr + 2/r dr ) C = S  plus boundary conditions.
!-------------------------------------------------------------------------
   subroutine cod_cmbflux(q, flx)
      double precision, intent(in)  :: q
      double precision, intent(out) :: flx
      double precision :: S(mes_oc%N)
      type (lumesh) :: A
      integer :: N, j, info
      call tim_lumesh_X ( mes_oc, 0d0, q/d_implicit, 0, A )
      call cod_bc_C     ( mes_oc, A, 0)
      call mes_lu_find  ( mes_oc, A )
      N = mes_oc%N
      S = cod_S
      S(1) = cod_bcRe(1,0)
      S(N) = cod_bcRe(2,0)
      call dgbtrs('N', N, i_KL, i_KL, 1, A%M, 3*i_KL+1,  &
                   A%ipiv, S, N, info )
      if(info/=0) stop 'cod_cmbflux'
      flx = 0d0
      do j = N-i_KL, N
         flx = flx  +  S(j) * mes_oc%dr(1)%M(i_KL+1+N-j,j)
      end do
   end subroutine cod_cmbflux


!------------------------------------------------------------------------
!  cC -> (grad C)r,t,p  phys 
!------------------------------------------------------------------------
   subroutine cod_transform()
      use transform
      type (spec) :: sC, sC_

      call var_coll_meshmult(mes_oc%dr(1),cod_C, cC)
      call var_coll2spec(cod_C,sC, c2=cC,s2=sC_) 

      call tra_spec2phys(sC_, cod_gradr)

      call tra_grad(sC, cod_gradt,cod_gradp)
      
   end subroutine cod_transform


!------------------------------------------------------------------------
!  N  := cN,                         save N at time t
!  C  := Y cC + cN,   C := invX C,   get prediction C*
!  cC := C                           copy prediction
!------------------------------------------------------------------------
   subroutine cod_predictor()

      call var_coll_copy (cod_NC, NC)
      call tim_multY     (.true.,YC,cod_C,cod_NC, C)
      call cod_setbc     (C)
      call tim_invX      (.true.,XC, C)
      call var_coll_copy (C, cod_C)
      
   end subroutine cod_predictor
   
   
!------------------------------------------------------------------------
!  C  := c (cN - N),   	using N* get nlin correction
!  N  := cN		save last N
!  C  := invX C,   	get correction to C, correction has 0 bc
!  cC := cC + C		update correction
!------------------------------------------------------------------------
   subroutine cod_corrector()

      call tim_nlincorr (cod_NC,NC, C)
      call var_coll_copy(cod_NC,NC)
      call tim_zerobc   (C)
      call tim_invX     (.true.,XC, C)
      call tim_addcorr  (C, cod_C)

   end subroutine cod_corrector
   
   
!*************************************************************************
 end module codensity
!*************************************************************************
