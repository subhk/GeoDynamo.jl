!*************************************************************************
!
!
!*************************************************************************
#include "../parallel.h"
 module velocity
!*************************************************************************
   use mpi
   use variables
   use timestep
   implicit none
   save

   type (phys) :: vel_r
   type (phys) :: vel_t
   type (phys) :: vel_p
   type (phys) :: vel_curlr
   type (phys) :: vel_curlt
   type (phys) :: vel_curlp
   type (coll) :: vel_Tor
   type (coll) :: vel_Pol
   type (coll) :: vel_NTor
   type (coll) :: vel_NPol
   type (coll) :: vel_NPre

   type (coll), private :: Tor
   type (coll), private :: Pol
   type (coll), private :: NTor
   type (coll), private :: NPol

   type (lumesh), private :: XTor(0:i_L1)
   type (lumesh), private :: XPol(0:i_L1)
   type (lumesh), private :: XGre(0:i_L1)
   type (mesh),   private :: YTor(0:i_L1)
   type (mesh),   private :: YPol(0:i_L1)

   double precision, private :: Gre (i_N,2,i_L1)
   double precision, private :: invG(  2,2,i_L1)

   type (coll), private :: cq,cs,ct
   type (spec), private :: sq,ss,st
   
 contains

!------------------------------------------------------------------------
!  initialise velocity/pressure field
!------------------------------------------------------------------------
   subroutine vel_precompute()
      call var_coll_init(mes_oc,var_H, vel_Tor)
      call var_coll_init(mes_oc,var_H, vel_Pol)
   end subroutine vel_precompute


!------------------------------------------------------------------------
!  precomputation of velocity/pressure field matrices
!------------------------------------------------------------------------
   subroutine vel_matrices()
      type (harm) :: H
      double precision :: c1, c2
      integer :: N_, l

      c1 = d_Ro
      c2 = d_E

      do l = 1, i_L1
         call tim_lumesh_X      ( mes_oc, c1, c2, l, XTor(l) )
         call vel_bc_Tor        ( mes_oc, XTor(l), l)
         call mes_lu_find       ( mes_oc, XTor(l) )
         call tim_mesh_Y        ( mes_oc, c1, c2, l, YTor(l) )
      end do

      do l = 1, i_L1
         call tim_lumesh_X      ( mes_oc, c1, c2, l, XPol(l) )
         call vel_bc_Pol        ( mes_oc, XPol(l), l)
         call mes_lu_find       ( mes_oc, XPol(l) )
         call tim_mesh_Y        ( mes_oc, c1, c2, l, YPol(l) )
      end do

                                ! greens functions & influence matrix
      c1 = 0d0
      c2 = 1d0/d_implicit
      N_ = mes_oc%N
      do l = 1, i_L1
         call tim_lumesh_X      ( mes_oc, c1, c2, l, XGre(l))
         call vel_bc_Gre        ( mes_oc, XGre(l), l)
         call mes_lu_find       ( mes_oc, XGre(l) )
      end do

      call var_coll_init(mes_oc,H, Pol)
      do l = 1, i_L1
         Pol%H%l(0) = l
         Pol%H%m(0) = 0
         Pol%H%pH0 = 0
         Pol%H%pH1 = 0
         Pol%Re( :,0) = 0d0
         Pol%Im( :,0) = 0d0
         Pol%Re( 1,0) = 1d0
         Pol%Im(N_,0) = 1d0
         call tim_invX(.false.,XGre, Pol)
         call tim_zerobc(Pol)
         call tim_invX(.false.,XPol, Pol)
         Gre(:,1,l) = Pol%Re(:,0)
         Gre(:,2,l) = Pol%Im(:,0)
         invG(1,1,l) = Gre( 1,1,l)
         invG(1,2,l) = Gre( 1,2,l)
         invG(2,1,l) = Gre(N_,1,l)
         invG(2,2,l) = Gre(N_,2,l)
         call mes_mat_invert(2,invG(1,1,l),2)
      end do

   end subroutine vel_matrices


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   subroutine vel_bc_Tor(D,A,l)
      type (rdom),   intent(in)  :: D
      type (lumesh), intent(out) :: A
      integer,       intent(in)  :: l
      integer :: j

      if(i_vel_bc==1 .or. i_vel_bc==2) then	
         A%M(2*i_KL+1,1)   = 1d0
      else
         do j = 1, 1+i_KL
            A%M(2*i_KL+1+1-j,j) = D%dr(1)%M(i_KL+1+1-j,j)
         end do
         A%M(2*i_KL+1,1) = A%M(2*i_KL+1,1) - D%r(1,-1)
      end if
      
      if(i_vel_bc==1 .or. i_vel_bc==3) then	
         A%M(2*i_KL+1,D%N) = 1d0
      else
         do j = D%N-i_KL, D%N
            A%M(2*i_KL+1+D%N-j,j) = D%dr(1)%M(i_KL+1+D%N-j,j)
         end do
         A%M(2*i_KL+1,D%N) = A%M(2*i_KL+1,D%N) - D%r(D%N,-1)
      end if

   end subroutine vel_bc_Tor
   
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   subroutine vel_bc_Pol(D,A,l)
      type (rdom),   intent(in)  :: D
      type (lumesh), intent(out) :: A
      integer,       intent(in)  :: l
      integer :: j

      if(i_vel_bc==1 .or. i_vel_bc==2) then	
         do j = 1, 1+i_KL
            A%M(2*i_KL+1+1-j,j) = D%dr(1)%M(i_KL+1+1-j,j)
         end do
      else
         do j = 1, 1+i_KL
            A%M(2*i_KL+1+1-j,j) = D%dr(2)%M(i_KL+1+1-j,j)
         end do
      end if
      
      if(i_vel_bc==1 .or. i_vel_bc==3) then	
         do j = D%N-i_KL, D%N
            A%M(2*i_KL+1+D%N-j,j) = D%dr(1)%M(i_KL+1+D%N-j,j)
         end do
      else
         do j = D%N-i_KL, D%N
            A%M(2*i_KL+1+D%N-j,j) = D%dr(2)%M(i_KL+1+D%N-j,j)
         end do
      end if

   end subroutine vel_bc_Pol


! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   subroutine vel_bc_Gre(D,A,l)
      type (rdom),   intent(in)  :: D
      type (lumesh), intent(out) :: A
      integer,       intent(in)  :: l
      A%M(2*i_KL+1,1)   = 1d0
      A%M(2*i_KL+1,D%N) = 1d0
   end subroutine vel_bc_Gre


!------------------------------------------------------------------------
!  set boundary condition rotating core on T10 mode.
!------------------------------------------------------------------------
   subroutine vel_setbc_Tor(a)
      use rotation
      type (coll), intent(out) :: a
      integer :: N_
      _loop_lm_vars

      N_ = a%D%N
      _loop_lm_begin(a)
         a%Re(1 ,nh) = 0d0
         a%Re(N_,nh) = 0d0
         a%Im(1 ,nh) = 0d0
         a%Im(N_,nh) = 0d0
         if(i_vel_bc>2) cycle
         if(l/=1 .or. m_/=0) cycle
         a%Re(1,nh) = rot_omega * mes_oc%r(1,nh)
         if(tim_it/=0) a%Re(1,nh) = a%Re(1,nh) - vel_Tor%Re(1,nh)
      _loop_lm_end

   end subroutine vel_setbc_Tor


!------------------------------------------------------------------------
!  Evaluate in physical space  u  and  curl u
!------------------------------------------------------------------------
   subroutine vel_transform()
      use transform

      call var_coll_TorPol2qst(vel_Tor,vel_Pol, cq,cs,ct)
      call var_coll2spec(cq,sq, c2=cs,s2=ss, c3=ct,s3=st)
      call tra_qst2rtp(sq,ss,st, vel_r,vel_t,vel_p)

      call var_coll_qstcurl(cq,cs,ct, cq,cs,ct)
      call var_coll2spec(cq,sq, c2=cs,s2=ss, c3=ct,s3=st)
      call tra_qst2rtp(sq,ss,st, vel_curlr,vel_curlt,vel_curlp)

   end subroutine vel_transform


!------------------------------------------------------------------------
! Tor:
!  N  := vN,            save N at time t
!  T  := Y vV + vN,     rhs
!  vT := invX T,   	get prediction B*
!  vT := T              copy prediction
! Pol:
!  N  := vN,            save N at time t
!  Np := vNp
!  P  := Y vV + vN      rhs
!  Pr := vNp
!  P,Pr -> P            solve -> prediction
!  vP := P              copy pred
!------------------------------------------------------------------------
   subroutine vel_predictor()

      call var_coll_copy (vel_NTor, NTor)
      call tim_multY     (.false.,YTor,vel_Tor,vel_NTor, Tor)
      call vel_setbc_Tor (Tor)
      call tim_invX      (.false.,XTor, Tor)
      call var_coll_copy (Tor, vel_Tor)
 
      call var_coll_copy (vel_NPol, NPol)
      call var_coll_copy (vel_NPol, Pol)
      call tim_zerobc    (Pol)
      call tim_invX      (.false.,XGre, Pol)
      call tim_multY     (.false.,YPol,vel_Pol,Pol, Pol)
      call tim_zerobc    (Pol)
      call tim_invX      (.false.,XPol, Pol)
      call vel_gre_adjust()
      call var_coll_copy (Pol, vel_Pol)

   end subroutine vel_predictor
   
   
!------------------------------------------------------------------------
! Tor:
!  T  := c (vN - N),   	using N* get nlin correction
!  N  := vN
!  T  := invX T,   	get correction to V (includes factor c)
!  vT := vT + T		update correction
! Pol:
!  P  := c (vN - N),   	using N* get nlin correction
!  N  := vN
!  Pr := c (vNp- Np)
!  P,Pr -> P            solve -> correction
!  vP := vP + P		update correction
!------------------------------------------------------------------------
   subroutine vel_corrector()

      call tim_nlincorr  (vel_NTor,NTor, Tor)
      call var_coll_copy (vel_NTor,NTor)
      call vel_setbc_Tor (Tor)
      call tim_invX      (.false.,XTor, Tor)
      call tim_addcorr   (Tor, vel_Tor)

      call tim_nlincorr  (vel_NPol,NPol, Pol)
      call var_coll_copy (vel_NPol,NPol)
      call tim_zerobc    (Pol)
      call tim_invX      (.false.,XGre, Pol)
      call tim_zerobc    (Pol)
      call tim_invX      (.false.,XPol, Pol)
      call vel_gre_adjust()
      call tim_addcorr   (Pol, vel_Pol)

   end subroutine vel_corrector
   

!------------------------------------------------------------------------
!------------------------------------------------------------------------
   subroutine vel_gre_adjust()
      double precision :: a, b, BCi,BCo
      integer :: N_
      _loop_lm_vars

      N_ = Pol%D%N
      _loop_lm_begin(Pol)
         if(l==0) cycle
         BCi = Pol%Re( 1,nh)            
         BCo = Pol%Re(N_,nh)            
         a = invG(1,1,l)*BCi + invG(1,2,l)*BCo
         b = invG(2,1,l)*BCi + invG(2,2,l)*BCo
         Pol%Re(:,nh) =  Pol%Re(:,nh) - a*Gre(:,1,l) - b*Gre(:,2,l)
         BCi = Pol%Im( 1,nh)
         BCo = Pol%Im(N_,nh)
         a = invG(1,1,l)*BCi + invG(1,2,l)*BCo
         b = invG(2,1,l)*BCi + invG(2,2,l)*BCo
         Pol%Im(:,nh) =  Pol%Im(:,nh) - a*Gre(:,1,l) - b*Gre(:,2,l)
      _loop_lm_end

   end subroutine vel_gre_adjust
   
   
   
!*************************************************************************
 end module velocity
!*************************************************************************
