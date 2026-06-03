!*************************************************************************
!  Variables in spectral,collocated and physical space.
!  Simple functions on variables.
!
!     r_n    n in [1,N]
!     th_j   j in [1,Th]
!     ph_i   i in [0,Ph)
!
!  order of harmonics: see parallel.h _loop_setlm
!
!*************************************************************************
#include "../parallel.h"
 module variables
!*************************************************************************
   use mpi
   use parameters
   use meshs
   implicit none
   save

   type harm
      integer              :: l(0:i_H1), m(0:i_H1)
      integer              :: pH0,pH1, pH0_(0:_Nr-1),pH1_(0:_Nr-1)
   end type harm

   type thdom
      integer              :: m(0:2*_pHc-1,0:_Nth-1)
      integer              :: pM, pM_(0:_Nth-1)
      integer              :: pThi,pTh, pThi_(0:_Nth-1),pTh_(0:_Nth-1)
   end type thdom

   type spec
      type (rdom), pointer :: D
      type (harm), pointer :: H
      double precision     :: Re(0:i_H1, i_pN)
      double precision     :: Im(0:i_H1, i_pN)
   end type spec

   type coll
      type (rdom), pointer :: D
      type (harm), pointer :: H
      double precision     :: Re(i_N, 0:i_pH1)
      double precision     :: Im(i_N, 0:i_pH1)
   end type coll

   type phys
      double precision     :: Re(i_pTh, 0:i_Ph-1, i_pN)
   end type phys
   
   type (harm)               :: var_H
   type (thdom)              :: var_Th
   double precision, private :: ad_ll1(0:i_L1)
   double precision, private :: ad_ll11(0:i_L1)
   double precision, private :: ad_rtll1(0:i_L1)
   double precision, private :: ad_rtll11(0:i_L1)
   type (coll),      private :: coll_

 contains


!------------------------------------------------------------------------
!  Initialise stuff for variables
!------------------------------------------------------------------------
   subroutine var_precompute()
      integer :: l
      ad_ll1(0)    = 0d0
      ad_ll11(0)   = 0d0
      ad_rtll1(0)  = 0d0
      ad_rtll11(0) = 0d0
      do l = 1, i_L1
         ad_ll1(l)    = dble(l*(l+1))
         ad_ll11(l)   = 1d0 / ad_ll1(l)
         ad_rtll1(l)  = dsqrt(ad_ll1(l))
         ad_rtll11(l) = 1d0 / ad_rtll1(l)
      end do
      call var_harm_init(var_H)
      call var_thdom_init(var_Th)
   end subroutine var_precompute


!-------------------------------------------------------------------------
!  initialise harmonic indices
!-------------------------------------------------------------------------

   subroutine var_harm_init(H)
      type (harm), intent(out) :: H
      integer :: n,r,p
     _loop_setlm_vars
!WD 5/09/2013, the initialisation and distribution of the harmonic modes
!has changed now, to avoid a segmentaion fault, when iH is smaller than
! _Np*(Np-1) -> see blog.

      H%pH1_ = -1
      do n = 0, i_H1
         r = _Nr - modulo(n,_Nr) - 1
         H%pH1_(r) = H%pH1_(r) + 1
      end do

      H%pH0_(0) = 0
      do r = 1, _Nr-1
         H%pH0_(r) = H%pH0_(r-1) + H%pH1_(r-1) + 1
      end do 
      p = modulo(mpi_rnk,_Nr)
      H%pH0 = H%pH0_(p)
      H%pH1 = H%pH1_(p)

      _loop_setlm_begin
         H%l(nh) = l
         H%m(nh) = m_
      _loop_setlm_end
   end subroutine var_harm_init



!-------------------------------------------------------------------------
!  initialise theta domains
!-------------------------------------------------------------------------
   subroutine var_thdom_init(Th)
      type (thdom), intent(out) :: Th
      integer :: n,p
      _loop_sumth_vars

      Th%pThi_(0) = 1
      n = modulo(i_Th,i_pTh)
      if(n==0) Th%pTh_(0) = i_pTh
      if(n/=0) Th%pTh_(0) = n
      do p = 1, _Nth-1
         Th%pThi_(p) = Th%pThi_(p-1) + Th%pTh_(p-1)
         Th%pTh_ (p) = i_pTh
      end do
      p = mpi_rnk/_Nr
      Th%pThi = Th%pThi_(p)
      Th%pTh  = Th%pTh_ (p)

      _loop_sumth_begin
         if(m_==m__) cycle
         Th%m(m,p) = m_/i_Mp
      _loop_sumth_end
      Th%pM_(p) = m+1
      Th%pM = m+1
#ifdef _MPI
      do p = 0, _Nth-1
         call mpi_bcast(Th%m(0,p), 2*_pHc, mpi_integer,  &
            p*_Nr, mpi_comm_world, mpi_er)
         call mpi_bcast(Th%pM_(p), 1, mpi_integer,  &
            p*_Nr, mpi_comm_world, mpi_er)
      end do
#endif
      
   end subroutine var_thdom_init


!-------------------------------------------------------------------------
!  initialise collocated variable
!-------------------------------------------------------------------------
   subroutine var_coll_init(D,H, a)
      type (rdom), target, intent(in) :: D
      type (harm), target, intent(in) :: H
      type (coll),         intent(out) :: a
      a%D => D
      a%H => H
      a%Re = 0d0
      a%Im = 0d0
   end subroutine var_coll_init


!-------------------------------------------------------------------------
!  initialise spectral variable
!-------------------------------------------------------------------------
   subroutine var_spec_init(D,H, a)
      type (rdom), target, intent(in) :: D
      type (harm), target, intent(in) :: H
      type (spec),         intent(out) :: a
      a%D => D
      a%H => H
   end subroutine var_spec_init


!-------------------------------------------------------------------------
!  Copy a collocated variable
!-------------------------------------------------------------------------
   subroutine var_coll_copy(in, out)
      type (coll), intent(in)  :: in
      type (coll), intent(out) :: out
      integer :: n, nhm
      out%D => in%D
      out%H => in%H
      n   = in%D%N
      nhm = in%H%pH1
      out%Re(1:n,0:nhm) = in%Re(1:n,0:nhm)
      out%Im(1:n,0:nhm) = in%Im(1:n,0:nhm)
   end subroutine var_coll_copy


!-------------------------------------------------------------------------
!  convert collocated -> spectral
!-------------------------------------------------------------------------
#ifndef _MPI
   subroutine var_coll2spec(c,s, c2,s2, c3,s3)
      type (coll), intent(in)  :: c
      type (spec), intent(out) :: s
      type (coll), intent(in),  optional :: c2,c3
      type (spec), intent(out), optional :: s2,s3
      integer :: n, nhm
      s%D => c%D 
      s%H => c%H
      n   = s%D%N
      nhm = s%H%pH1
      s%Re(0:nhm,1:n) = transpose(c%Re(1:n,0:nhm))
      s%Im(0:nhm,1:n) = transpose(c%Im(1:n,0:nhm))
      if(.not. present(c2)) return
      s2%D => c2%D 
      s2%H => c2%H
      s2%Re(0:nhm,1:n) = transpose(c2%Re(1:n,0:nhm))
      s2%Im(0:nhm,1:n) = transpose(c2%Im(1:n,0:nhm))
      if(.not. present(c3)) return
      s3%D => c3%D 
      s3%H => c3%H
      s3%Re(0:nhm,1:n) = transpose(c3%Re(1:n,0:nhm))
      s3%Im(0:nhm,1:n) = transpose(c3%Im(1:n,0:nhm))      
   end subroutine var_coll2spec

#else
   subroutine var_coll2spec(c,s, c2,s2, c3,s3)
      type (coll), intent(in)  :: c
      type (spec), intent(out) :: s
      type (coll), intent(in),  optional :: c2,c3
      type (spec), intent(out), optional :: s2,s3
      double precision :: bsend(2*i_pN*(i_pH1+1)*3,0:_Nr-1)
      double precision :: brecv(2*i_pN*(i_pH1+1)*3,0:_Nr-1)
      integer :: stp, dst,src, n,nh,l, rnk,rko, nc

      nc = 1
      s%D => c%D 
      s%H => c%H

      if(present(c2)) nc = 2
      if(nc==2) s2%D => c2%D
      if(nc==2) s2%H => c2%H
      if(present(c3)) nc = 3
      if(nc==3) s3%D => c3%D
      if(nc==3) s3%H => c3%H

      rnk = modulo(mpi_rnk,_Nr)
      rko = (mpi_rnk/_Nr)*_Nr

      do stp = 0, _Nr-1
         src  = modulo(_Nr-stp+rnk, _Nr)         
         mpi_tg = rko + stp
         call mpi_irecv( brecv(1,stp), 2*c%D%pN*(c%H%pH1_(src)+1)*nc,  &
            mpi_double_precision, rko+src, mpi_tg, mpi_comm_world,  &
            mpi_rq(stp), mpi_er)
      end do


      do stp = 0, _Nr-1
         dst  = modulo(stp+rnk, _Nr)
         l = 1
         do n = c%D%pNi_(dst), c%D%pNi_(dst)+c%D%pN_(dst)-1
            do nh = 0, c%H%pH1
               bsend(l,  stp) = c%Re(n,nh)
               bsend(l+1,stp) = c%Im(n,nh)
               l = l + 2
               if(nc<2) cycle
               bsend(l,  stp) = c2%Re(n,nh)
               bsend(l+1,stp) = c2%Im(n,nh)
               l = l + 2
               if(nc<3) cycle
               bsend(l,  stp) = c3%Re(n,nh)
               bsend(l+1,stp) = c3%Im(n,nh)
               l = l + 2
            end do
         end do
         mpi_tg = rko + stp

         call mpi_isend( bsend(1,stp), 2*c%D%pN_(dst)*(c%H%pH1+1)*nc,  &
            mpi_double_precision, rko+dst, mpi_tg, mpi_comm_world,  &
            mpi_rq(_Nr+stp), mpi_er)
      end do

      do stp = 0, _Nr-1
         src  = modulo(_Nr-stp+rnk, _Nr)      
     
         call mpi_wait( mpi_rq(stp), mpi_st, mpi_er)
         l = 1
       do n = 1, s%D%pN
          do nh = s%H%pH0_(src), s%H%pH0_(src)+s%H%pH1_(src)
               s%Re(nh,n) = brecv(l,  stp)
               s%Im(nh,n) = brecv(l+1,stp)
               l = l + 2
               if(nc<2) cycle
               s2%Re(nh,n) = brecv(l,  stp)
               s2%Im(nh,n) = brecv(l+1,stp)
               l = l + 2
               if(nc<3) cycle
               s3%Re(nh,n) = brecv(l,  stp)
               s3%Im(nh,n) = brecv(l+1,stp)
               l = l + 2
            end do
         end do

      end do      

      do stp = 0, _Nr-1
         call mpi_wait( mpi_rq(_Nr+stp), mpi_st, mpi_er)
      end do

   end subroutine var_coll2spec
#endif
   
!-------------------------------------------------------------------------
!  convert spectral -> collocated
!-------------------------------------------------------------------------
#ifndef _MPI
   subroutine var_spec2coll(s,c, s2,c2, s3,c3)
      type (spec), intent(in)  :: s
      type (coll), intent(out) :: c
      type (spec), intent(in),  optional :: s2,s3
      type (coll), intent(out), optional :: c2,c3
      integer :: n, nhm
      c%D => s%D
      c%H => s%H
      n   = s%D%N
      nhm = s%H%pH1
      c%Re(1:n,0:nhm) = transpose(s%Re(0:nhm,1:n))
      c%Im(1:n,0:nhm) = transpose(s%Im(0:nhm,1:n))
      if(.not. present(s2)) return
      c2%D => s2%D
      c2%H => s2%H
      c2%Re(1:n,0:nhm) = transpose(s2%Re(0:nhm,1:n))
      c2%Im(1:n,0:nhm) = transpose(s2%Im(0:nhm,1:n))
      if(.not. present(s3)) return
      c3%D => s3%D
      c3%H => s3%H
      c3%Re(1:n,0:nhm) = transpose(s3%Re(0:nhm,1:n))
      c3%Im(1:n,0:nhm) = transpose(s3%Im(0:nhm,1:n))
   end subroutine var_spec2coll

#else
   subroutine var_spec2coll(s,c, s2,c2, s3,c3)
      type (spec), intent(in)  :: s
      type (coll), intent(out) :: c
      type (spec), intent(in),  optional :: s2,s3
      type (coll), intent(out), optional :: c2,c3
      double precision :: bsend(2*(i_pH1+1)*i_pN*3,0:_Nr-1)
      double precision :: brecv(2*(i_pH1+1)*i_pN*3,0:_Nr-1)
      integer :: stp, dst,src, n,nh,l, rnk,rko, ns

      ns = 1
      c%D => s%D 
      c%H => s%H
      if(present(s2)) ns = 2
      if(ns==2) c2%D => s2%D
      if(ns==2) c2%H => s2%H
      if(present(s3)) ns = 3
      if(ns==3) c3%D => s3%D
      if(ns==3) c3%H => s3%H

      rnk = modulo(mpi_rnk,_Nr)
      rko = (mpi_rnk/_Nr)*_Nr
      
      do stp = 0, _Nr-1
         src  = modulo(_Nr-stp+rnk, _Nr)
         mpi_tg = rko + stp
         call mpi_irecv( brecv(1,stp), 2*(s%H%pH1+1)*s%D%pN_(src)*ns,  &
            mpi_double_precision, rko+src, mpi_tg, mpi_comm_world,  &
            mpi_rq(stp), mpi_er)
      end do

      do stp = 0, _Nr-1
         dst  = modulo(stp+rnk, _Nr)
         l = 1
         do nh = s%H%pH0_(dst), s%H%pH0_(dst)+s%H%pH1_(dst)
            do n = 1, s%D%pN
               bsend(l,  stp) = s%Re(nh,n)
               bsend(l+1,stp) = s%Im(nh,n)
               l = l + 2
               if(ns<2) cycle
               bsend(l,  stp) = s2%Re(nh,n)
               bsend(l+1,stp) = s2%Im(nh,n)
               l = l + 2
               if(ns<3) cycle
               bsend(l,  stp) = s3%Re(nh,n)
               bsend(l+1,stp) = s3%Im(nh,n)
               l = l + 2
            end do
         end do
         mpi_tg = rko + stp
         call mpi_isend( bsend(1,stp), 2*(s%H%pH1_(dst)+1)*s%D%pN*ns,  &
            mpi_double_precision, rko+dst, mpi_tg, mpi_comm_world,  &
            mpi_rq(_Nr+stp), mpi_er)
      end do
      
      do stp = 0, _Nr-1
         src  = modulo(_Nr-stp+rnk, _Nr)
         call mpi_wait( mpi_rq(stp), mpi_st, mpi_er)
         l = 1
         do nh = 0, c%H%pH1
            do n = c%D%pNi_(src), c%D%pNi_(src)+c%D%pN_(src)-1
               c%Re(n,nh) = brecv(l,  stp)
               c%Im(n,nh) = brecv(l+1,stp)
               l = l + 2
               if(ns<2) cycle
               c2%Re(n,nh) = brecv(l,  stp)
               c2%Im(n,nh) = brecv(l+1,stp)
               l = l + 2
               if(ns<3) cycle
               c3%Re(n,nh) = brecv(l,  stp)
               c3%Im(n,nh) = brecv(l+1,stp)
               l = l + 2
            end do
         end do
      end do

      do stp = 0, _Nr-1
         call mpi_wait( mpi_rq(_Nr+stp), mpi_st, mpi_er)
      end do

   end subroutine var_spec2coll
#endif

!------------------------------------------------------------------------
!  Multiply a collocated variable by a mesh-type
!     out = A.in
!------------------------------------------------------------------------
   subroutine var_coll_meshmult(A,in, out)
      type (mesh), intent(in)  :: A
      type (coll), intent(in)  :: in
      type (coll), intent(out) :: out
      double precision :: re(i_N), im(i_N)
      integer :: N_,nh, n,j,l,r

      out%D => in%D
      out%H => in%H
      N_  = in%D%N

      do nh = 0, in%H%pH1
         re = 0d0
         im = 0d0
         do j = 1, N_
            l = max(1,j-i_KL)
            r = min(j+i_KL,N_)
            do n = l, r
               re(n) = re(n) + A%M(i_KL+1+n-j,j) * in%Re(j,nh)
               im(n) = im(n) + A%M(i_KL+1+n-j,j) * in%Im(j,nh)
            end do
         end do
         out%Re(1:N_,nh) = re(1:N_)
         out%Im(1:N_,nh) = im(1:N_)
      end do

   end subroutine var_coll_meshmult


!------------------------------------------------------------------------
!  change to qst form  LM: Pol->q,s, Tor->t
!------------------------------------------------------------------------
   subroutine var_coll_TorPol2qst(Tor,Pol, q,s,t)
      type (coll), intent(in)  :: Tor,Pol
      type (coll), intent(out) :: q,s,t
      double precision :: a
      integer :: N_,n
      _loop_lm_vars
         			! q = l(l+1)/r Pol
      q%D =>Pol%D
      q%H =>Pol%H
      N_  = Pol%D%N
      _loop_lm_begin(q)
         do n = 1, N_
            a = ad_ll1(l) * q%D%r(n,-1)
            q%Re(n,nh) = Pol%Re(n,nh) * a
            q%Im(n,nh) = Pol%Im(n,nh) * a
         end do
      _loop_lm_end
                                ! s = sqrt(l(l+1)) (1/r+dr) Pol
      call var_coll_meshmult(Pol%D%r1dr,Pol, s)
      _loop_lm_begin(s)
         do n = 1, N_
            s%Re(n,nh) = s%Re(n,nh) * ad_rtll1(l)
            s%Im(n,nh) = s%Im(n,nh) * ad_rtll1(l)
         end do
      _loop_lm_end
                                ! t = - sqrt(l(l+1)) Tor
      t%D =>Tor%D
      t%H =>Tor%H
      _loop_lm_begin(t)
         do n = 1, N_
            t%Re(n,nh) = - Tor%Re(n,nh) * ad_rtll1(l)
            t%Im(n,nh) = - Tor%Im(n,nh) * ad_rtll1(l)
         end do
      _loop_lm_end

   end subroutine var_coll_TorPol2qst


!------------------------------------------------------------------------
!  change qst to Tor,Pol form  LM: q->Pol, t->Tor
!  Only really valid for solenoindal vectors, only q,t required
!------------------------------------------------------------------------
   subroutine var_coll_qst2TorPol(q,t, Tor,Pol)
      type (coll), intent(in)  :: q,t
      type (coll), intent(out) :: Tor,Pol
      double precision :: a
      integer :: N_,n
      _loop_lm_vars

         			! q = l(l+1)/r Pol
      Pol%D =>q%D
      Pol%H =>q%H
      N_    = q%D%N
      _loop_lm_begin(q)
         do n = 1, N_
            a = q%D%r(n,1) * ad_ll11(l)
            Pol%Re(n,nh) = q%Re(n,nh) * a
            Pol%Im(n,nh) = q%Im(n,nh) * a
         end do
      _loop_lm_end
         			! t = - sqrt(l(l+1)) Tor
      Tor%D =>t%D
      Tor%H =>t%H
      _loop_lm_begin(t)
         do n = 1, N_
            Tor%Re(n,nh) = - t%Re(n,nh) * ad_rtll11(l)
            Tor%Im(n,nh) = - t%Im(n,nh) * ad_rtll11(l)
         end do
      _loop_lm_end

   end subroutine var_coll_qst2TorPol


!------------------------------------------------------------------------
!  take the curl of a vector in qst form;
!  change to qst form  LM:  q,s->ot,  t->oq,os
!------------------------------------------------------------------------
   subroutine var_coll_qstcurl(q,s,t, oq,os,ot)
      type (coll), intent(in)  :: q,s,t
      type (coll), intent(out) :: oq,os,ot
      double precision :: a
      integer :: N_,n
      _loop_lm_vars

      call var_coll_copy(t,coll_)
         			! t := (1/r+dr)s - sqrt(l(l+1))/r q
      call var_coll_meshmult(s%D%r1dr,s, ot)
      N_  = ot%D%N
      _loop_lm_begin(ot)
         do n = 1, N_
            a = ad_rtll1(l) * ot%D%r(n,-1)
            ot%Re(n,nh) = ot%Re(n,nh) - a*q%Re(n,nh)
            ot%Im(n,nh) = ot%Im(n,nh) - a*q%Im(n,nh)
         end do
      _loop_lm_end
         			! s := - (1/r+dr)t
      call var_coll_meshmult(coll_%D%r1dr,coll_, os)
      _loop_lm_begin(os)
         do n = 1, N_
            os%Re(n,nh) =  - os%Re(n,nh)
            os%Im(n,nh) =  - os%Im(n,nh)
         end do
      _loop_lm_end
         			! q := - sqrt(l(l+1))/r t
      oq%D =>coll_%D
      oq%H =>coll_%H
      _loop_lm_begin(oq)
         do n = 1, N_
            a = - ad_rtll1(l) * oq%D%r(n,-1)
            oq%Re(n,nh) = coll_%Re(n,nh)*a
            oq%Im(n,nh) = coll_%Im(n,nh)*a
         end do
      _loop_lm_end

   end subroutine var_coll_qstcurl


!------------------------------------------------------------------------
!  r/(l(l+1)) \hat{r} . curl A  where A is in qst form;  only t required
!------------------------------------------------------------------------
   subroutine var_coll_qstllcurlr(t, r)
      use legendre
      type (coll), intent(in)  :: t
      type (coll), intent(out) :: r
      double precision :: a
      integer :: N_,n
      _loop_lm_vars
         			!  - (r/(l(l+1))) * sqrt(l(l+1))/r t
      r%D =>t%D
      r%H =>t%H
      N_  = t%D%N
      _loop_lm_begin(t)
         do n = 1, N_
            r%Re(n,nh) =  - t%Re(n,nh) * ad_rtll11(l)
            r%Im(n,nh) =  - t%Im(n,nh) * ad_rtll11(l)
         end do
      _loop_lm_end

   end subroutine var_coll_qstllcurlr


!------------------------------------------------------------------------
!  r/(l(l+1)) \hat{r} . curl curl A  
!  where A is in qst form;  only q,s required
!------------------------------------------------------------------------
   subroutine var_coll_qstllcurlcurlr(q,s, r)
      use legendre
      type (coll), intent(in)  :: q,s
      type (coll), intent(out) :: r
      integer :: N_,n
      _loop_lm_vars
         			! r/(l(l+1)) * ...
         			! l(l+1)/r^2 q - sqrt(l(l+1))/r (1/r+dr)s
      call var_coll_meshmult(s%D%r1dr,s, r)
      N_  = q%D%N
      _loop_lm_begin(q)
         do n = 1, N_
            r%Re(n,nh) = r%Re(n,nh)*ad_rtll11(l)
            r%Re(n,nh) = q%Re(n,nh)*r%D%r(n,-1) - r%Re(n,nh)
            r%Im(n,nh) = r%Im(n,nh)*ad_rtll11(l)
            r%Im(n,nh) = q%Im(n,nh)*r%D%r(n,-1) - r%Im(n,nh)
         end do
      _loop_lm_end

   end subroutine var_coll_qstllcurlcurlr


!------------------------------------------------------------------------
!  phi-derivative
!------------------------------------------------------------------------
   subroutine var_coll_dph(in, out)
      use legendre
      type (coll), intent(in)  :: in
      type (coll), intent(out) :: out
      double precision :: a
      integer :: N_,n
      _loop_lm_vars

      out%D =>in%D
      out%H =>in%H
      N_  = in%D%N
      _loop_lm_begin(out)
         do n = 1, N_
            a = in%Re(n,nh)
            out%Re(n,nh) =  - in%Im(n,nh) * m_
            out%Im(n,nh) =    a * m_
         end do
      _loop_lm_end

   end subroutine var_coll_dph


!------------------------------------------------------------------------
!  norm  \int a.a dV
!------------------------------------------------------------------------
   subroutine var_coll_norm(a, Eodd,Eeven,El1,Em0)
      type (coll), intent(in) :: a
      double precision, intent(out) :: Eodd, Eeven, El1, Em0
      double precision :: w, f(a%D%N), Fev(a%D%N), Fod(a%D%N)
      double precision :: Fl1(a%D%N), Fm0(a%D%N)
      integer :: N_,n
      _loop_lm_vars
      
      Fod = 0d0
      Fev = 0d0
      Fl1 = 0d0
      Fm0 = 0d0
      N_  = a%D%N
      
      _loop_lm_begin(a)
         w = 4d0 * d_PI / dble(2*l+1)
         if(m_/=0) w = 4d0*w
         f =  w*( a%Re(1:N_,nh)*a%Re(1:N_,nh)  &
                + a%Im(1:N_,nh)*a%Im(1:N_,nh) )
         if(m_==0)  &
            Fm0 = Fm0 + f
         if(l==1)  &
            Fl1 = Fl1 + f
         if(modulo(l-m_,2)==1) then
            Fod = Fod + f
         else
            Fev = Fev + f
         end if
      _loop_lm_end

      Eodd  = dot_product(Fod,a%D%intr2dr(1:N_))
      Eeven = dot_product(Fev,a%D%intr2dr(1:N_))
      El1   = dot_product(Fl1,a%D%intr2dr(1:N_))
      Em0   = dot_product(Fm0,a%D%intr2dr(1:N_))
#ifdef _MPI
      call mpi_allreduce( Eodd,  w, 1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      Eodd  = w
      call mpi_allreduce( Eeven, w, 1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      Eeven = w
      call mpi_allreduce( El1,   w, 1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      El1   = w
      call mpi_allreduce( Em0,   w, 1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      Em0   = w
#endif

   end subroutine var_coll_norm


!------------------------------------------------------------------------
!  norm   \int a.a dV  for each l,m
!------------------------------------------------------------------------
   subroutine var_coll_normlm(a, Eh,El,Em)
      type (coll), intent(in) :: a
      double precision, intent(out) :: Eh(0:2*i_KL)
      double precision, intent(out) :: El(0:i_L1), Em(0:i_M1)
#ifdef _MPI
      double precision :: Eh_(0:2*i_KL)
      double precision :: El_(0:i_L1), Em_(0:i_M1)
#endif
      double precision :: f(a%D%N)
      double precision :: w, e
      integer :: N_,n, h, m
      _loop_lm_vars
      
      Eh = 0d0
      El = 0d0
      Em = 0d0
      N_  = a%D%N
      _loop_lm_begin(a)
         w = 4d0 * d_PI / dble(2*l+1)
         m = m_/i_Mp
         if (m /= 0) w = 4d0*w 
         f =  w * ( a%Re(1:N_,nh)*a%Re(1:N_,nh)  &
                  + a%Im(1:N_,nh)*a%Im(1:N_,nh) )
         e = dot_product(f,a%D%intr2dr(1:N_))
         El(l) = El(l) + e
         Em(m) = Em(m) + e
         do h = 0, 2*i_KL
            Eh(h) = Eh(h) + dot_product(f,a%D%intr2drh(1:N_,h))
         end do
      _loop_lm_end

#ifdef _MPI
      call mpi_allreduce( Eh, Eh_, 1+2*i_KL, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      Eh = Eh_
      call mpi_allreduce( El, El_, 1+i_L1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      El = El_
      call mpi_allreduce( Em, Em_, 1+i_M1, mpi_double_precision,  &
         mpi_sum, mpi_comm_world, mpi_er)
      Em = Em_
#endif
   
   end subroutine var_coll_normlm

!*************************************************************************
 end module variables
!*************************************************************************

