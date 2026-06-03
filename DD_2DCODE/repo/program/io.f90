!**************************************************************************
!  IN/OUT 
!
!**************************************************************************
#include "../parallel.h"
 module io
!**************************************************************************
   use parameters
   use variables
   use transform
   use velocity
   use codensity
   use magnetic
   use rotation
   use netcdf

   implicit none
   save

   character(200)        :: io_statefile, io_codBCfile, io_magB0file
   integer,     private  :: io_save1, io_save2
   integer,     private  :: io_vE, io_mE, io_mIC, io_mMB
   integer,     private  :: io_cNu, io_rTq, io_tdt, io_pts
   type (coll), private  :: cq,cs,ct
   logical,     private  :: do_random_init

 contains
 
 
!--------------------------------------------------------------------------
!  initialiser fn
!--------------------------------------------------------------------------
   subroutine io_precompute()
      io_statefile = 'state.cdf.in'
      io_codBCfile = 'codBC.cdf.in'
      io_magB0file = 'magB0.cdf.in'
      io_save1 = 0
      io_save2 = 0
      io_vE  = 0
      io_mE  = 0
      io_mIC = 0
      io_mMB = 0
      io_cNu = 0
      io_rTq = 0
      io_tdt = 0
      io_pts = 0
      if(b_cod_tstep .and.  &
        (i_cod_bc==1 .or. i_cod_bc==3)) io_cNu = 50
      if(b_vel_tstep)                   io_vE  = 30
      if(b_mag_tstep)                   io_mE  = 40
      if(b_mag_tstep .and. mes_ic%N>1 ) io_mIC = 41
      if(b_mag_tstep)                   io_mMB = 42
      if(b_rot_tstep)                   io_rTq = 60
      if(d_timestep<0d0)                io_tdt = 70
                                        io_pts = 80
   end subroutine io_precompute
 
 

!--------------------------------------------------------------------------
!  Open files written to every ... steps runtime
!--------------------------------------------------------------------------
   subroutine io_openfiles()
      if(mpi_rnk/=0) return
      if(io_vE /=0)  open(io_vE,  status='unknown', file='vel_energy.dat')
      if(io_mE /=0)  open(io_mE,  status='unknown', file='mag_energy.dat')
      if(io_mIC/=0)  open(io_mIC, status='unknown', file='mag_ic.dat')
      if(io_mMB/=0)  open(io_mMB, status='unknown', file='mag_cmb.dat')
      if(io_cNu/=0)  open(io_cNu, status='unknown', file='cod_nusselt.dat')
      if(io_rTq/=0)  open(io_rTq, status='unknown', file='rot_torque.dat')
      if(io_tdt/=0)  open(io_tdt, status='unknown', file='tim_step.dat'  )
      if(io_pts/=0)  open(io_pts, status='unknown', file='pts_bench.dat'  )
   end subroutine io_openfiles


!--------------------------------------------------------------------------
!  Close files written to during runtime
!--------------------------------------------------------------------------
   subroutine io_closefiles()
      if(mpi_rnk/=0) return
      if(io_vE /=0) close(io_vE)
      if(io_mE /=0) close(io_mE)
      if(io_mIC/=0) close(io_mIC)
      if(io_mMB/=0) close(io_mMB)
      if(io_cNu/=0) close(io_cNu)
      if(io_rTq/=0) close(io_rTq)
      if(io_tdt/=0) close(io_tdt)
      if(io_pts/=0) close(io_pts)
   end subroutine io_closefiles


!--------------------------------------------------------------------------
!  Write to files
!--------------------------------------------------------------------------
   subroutine io_write2files()
      double precision :: KE,VD, ME,OD

      KE = 0.5d0
      VD = d_E/d_Ro
      ME = 0.5d0/d_Ro
      OD = 1d0/d_Ro

      if(modulo(tim_step,i_save_rate1)==0) then
         if(tim_step>0) call io_save_state()
         call io_save_spectrum ('vel',vel_Tor,vel_Pol,KE)
         call io_save_spectrum ('mag',mag_Tor,mag_Pol,ME)
         call io_save_spectrum1('cod',cod_C,1d0)
         if(b_mag_impose)  &
            call io_save_spectrum('mB0',mag_B0Tor,mag_B0Pol,ME)
         io_save1 = io_save1+1
      endif

      if(modulo(tim_step,i_save_rate2)==0) then
         if(io_vE /=0) call io_write_energy(io_vE, vel_Tor,  vel_Pol,  KE,VD)
         if(io_mE /=0) call io_write_energy(io_mE, mag_Tor,  mag_Pol,  ME,OD)
         if(io_mIC/=0) call io_write_energy(io_mIC,mag_icTor,mag_icPol,ME,OD)
         if(io_mMB/=0) call io_write_cmb   (io_mMB,mag_Tor,  mag_Pol,  ME)
         if(io_cNu/=0) call io_write_nussel(io_cNu,cod_C)
         if(io_rTq/=0) call io_write_torque(io_rTq)
         if(io_pts/=0) call io_write_bench (io_pts)
         io_save2 = io_save2+1
      end if

      if(tim_new_dt) call io_write_timestep(io_tdt)
   
   end subroutine io_write2files



!--------------------------------------------------------------------------
!  Load state - start from previous solution
!--------------------------------------------------------------------------
   subroutine io_load_state()
      integer :: e, f, i, rd, icrd
      integer :: n, No,Ni, No_,Ni_
      double precision :: d
      integer, parameter :: lda = 2*i_KL+1
      double precision :: Ao(lda,lda,mes_oc%N)
      double precision :: Ai(lda,lda,mes_ic%N)
      double precision, allocatable :: roc(:), ric(:)
      logical :: into, inti

      e=nf90_open(io_statefile, nf90_nowrite, f)      
      if(e/=nf90_noerr) then
         if(mpi_rnk==0) open(99, file='PRECOMPUTING')
         if(mpi_rnk==0) close(99, status='delete')
         stop 'state file not found!'
      end if
!  ## Piotr ##
!  initialize random numbers generator
!  (to perturb modes not present in the initail state file
!   with a random noise of small amplitude)

      call random_seed
      e=nf90_get_att(f, nf90_global,  't', d)
      if(d_time<0d0) tim_t = d

      No = mes_oc%N
      Ni = mes_ic%N
      e=nf90_inq_dimid(f,    'r', rd)
      e=nf90_inq_dimid(f,  'icr', icrd)
      e=nf90_inquire_dimension(f,   rd, len=No_)
      e=nf90_inquire_dimension(f, icrd, len=Ni_)

      allocate(roc(No_))
      allocate(ric(Ni_))
      e=nf90_inq_varid(f, 'r', i)
      e=nf90_get_var(f, i, roc)
      e=nf90_inq_varid(f, 'icr', i)
      e=nf90_get_var(f, i, ric)
      into = (No_/=No)
      if(.not.into) into = (maxval(dabs(mes_oc%r(1:No,1)-roc))>1d-8)
      inti = (Ni_/=Ni)
      if(.not.inti) inti = (maxval(dabs(mes_ic%r(1:Ni,1)-ric))>1d-8)
      if(into) then
         if(mpi_rnk==0) print*,' N  :', No_, ' --> ',No 
         call io_interp_wts(No_,roc,No,mes_oc%r(1,1), Ao)
      end if
      if(inti .and. Ni/=1 .and. Ni_/=1) then
         if(mpi_rnk==0) print*,' Nic:', Ni_, ' --> ',Ni 
         call io_interp_wts(Ni_,ric,Ni,mes_ic%r(1,1), Ai)
      end if

      e=nf90_inq_varid(f, 'dt', i)
      e=nf90_get_var(f, i, d)
      if(d_timestep<0d0) tim_dt = d

      e=nf90_inq_varid(f, 'icOmega', i)
      e=nf90_get_var(f, i, rot_omega)
      do_random_init = .true.
      call io_load_coll(f,'uT',into,No_,roc,Ao, vel_Tor)
      call io_load_coll(f,'uP',into,No_,roc,Ao, vel_Pol)
      call io_load_coll(f, 'C',into,No_,roc,Ao, cod_C)
      call io_load_coll(f,'BT',into,No_,roc,Ao, mag_Tor)
      call io_load_coll(f,'BP',into,No_,roc,Ao, mag_Pol)
      if(inti .and. Ni_==1) then
         call io_ic_extend('icBT',mag_Tor,mag_icTor)
         call io_ic_extend('icBP',mag_Pol,mag_icPol)
      else if(Ni/=1) then
         call io_load_coll(f,'icBT',inti,Ni_,ric,Ai, mag_icTor)
         call io_load_coll(f,'icBP',inti,Ni_,ric,Ai, mag_icPol)      
      end if
      deallocate(roc)
      deallocate(ric)

      e=nf90_close(f)
#ifdef _MPI
      call mpi_barrier(mpi_comm_world, mpi_er)
#endif

   end subroutine io_load_state


!--------------------------------------------------------------------------
!  Load imposed field
!--------------------------------------------------------------------------
   subroutine io_load_magB0()
      integer :: e, f, i, rd, icrd
      integer :: n, No,Ni, No_,Ni_
      integer, parameter :: lda = 2*i_KL+1
      double precision :: Ao(lda,lda,mes_oc%N)
      double precision :: Ai(lda,lda,mes_ic%N)
      double precision, allocatable :: roc(:), ric(:), fn(:)
      logical :: into, inti

      e=nf90_open(io_magB0file, nf90_nowrite, f)      
      if(e/=nf90_noerr) then
         if(mpi_rnk==0) open(99, file='PRECOMPUTING')
         if(mpi_rnk==0) close(99, status='delete')
         stop 'magB0 file not found!'
      end if

      No = mes_oc%N
      Ni = mes_ic%N
      e=nf90_inq_dimid(f,    'r', rd)
      e=nf90_inq_dimid(f,  'icr', icrd)
      e=nf90_inquire_dimension(f,   rd, len=No_)
      e=nf90_inquire_dimension(f, icrd, len=Ni_)

      allocate(fn (No_))
      allocate(roc(No_))
      allocate(ric(Ni_))
      e=nf90_inq_varid(f, 'r', i)
      e=nf90_get_var(f, i, roc)
      e=nf90_inq_varid(f, 'icr', i)
      e=nf90_get_var(f, i, ric)
      into = (No_/=No)
      if(.not.into) into = (maxval(dabs(mes_oc%r(1:No,1)-roc))>1d-8)
      inti = (Ni_/=Ni)
      if(.not.inti) inti = (maxval(dabs(mes_ic%r(1:Ni,1)-ric))>1d-8)
      if(into) then
         if(mpi_rnk==0) print*,' N  :', No_, ' --> ',No 
         call io_interp_wts(No_,roc,No,mes_oc%r(1,1), Ao)
      end if
      if(inti .and. Ni/=1 .and. Ni_/=1) then
         if(mpi_rnk==0) print*,' Nic:', Ni_, ' --> ',Ni 
         call io_interp_wts(Ni_,ric,Ni,mes_ic%r(1,1), Ai)
      end if

      do_random_init = .false.
      call io_load_coll(f,'B0T',into,No_,roc,Ao, mag_B0Tor)
      call io_load_coll(f,'B0P',into,No_,roc,Ao, mag_B0Pol)
      if(Ni/=1) then
         call io_load_coll(f,'icB0T',inti,Ni_,ric,Ai, mag_icB0Tor)
         call io_load_coll(f,'icB0P',inti,Ni_,ric,Ai, mag_icB0Pol)      
      end if

      deallocate(fn )
      deallocate(roc)
      deallocate(ric)

      e=nf90_close(f)
#ifdef _MPI
      call mpi_barrier(mpi_comm_world, mpi_er)
#endif

   end subroutine io_load_magB0


!--------------------------------------------------------------------------
!  Load codensity boundary condition and source info
!--------------------------------------------------------------------------
   subroutine io_load_codBC()
      integer :: e, f, i, rd, i9,j9
      integer :: n, No, No_
      double precision :: d
      integer, parameter :: lda = 2*i_KL+1
      double precision :: Ao(lda,lda,mes_oc%N)
      double precision, allocatable :: roc(:), fn(:)
      logical :: into

      e=nf90_open(io_codBCfile, nf90_nowrite, f)      
      if(e/=nf90_noerr) then
         if(mpi_rnk==0) then 
            cod_bcRe(1,0) = 1d0
            if (i_cod_bc==2 .or. i_cod_bc==4) cod_bcRe(2,0) = 1d0
            print*, 'WARNING! codBC file not found, set values:'
            print*, '   cod_S          : S = 0'
            print*, '   codBC l=0 inner:', real(cod_bcRe(1,0))
            print*, '   codBC l=0 outer:', real(cod_bcRe(2,0))
         end if
         return
      end if

      No = mes_oc%N
      e=nf90_inq_dimid(f,    'r', rd)
      e=nf90_inquire_dimension(f,   rd, len=No_)

      allocate(fn (No_))
      allocate(roc(No_))
      e=nf90_inq_varid(f, 'r', i)
      e=nf90_get_var(f, i, roc)
      into = (No_/=No)
      if(.not.into) into = (maxval(dabs(mes_oc%r(1:No,1)-roc))>1d-8)
      if(into) then
         if(mpi_rnk==0) print*,' N  :', No_, ' --> ',No 
         call io_interp_wts(No_,roc,No,mes_oc%r(1,1), Ao)
      end if

      fn = 0d0
      e=nf90_inq_varid(f, 'S', i)
      e=nf90_get_var(f, i, fn)
      if(into) then
         call io_interp(No_,roc,fn,Ao,No,mes_oc%r(1,1), cod_S)
      else
         cod_S = fn
      end if

      do_random_init = .false.
      call var_coll_init(mes_oc,var_H, cq)
      call io_load_coll(f,'Cbc',.false.,2,roc,Ao, cq)
      cod_bcRe = cq%Re(:2,:)
      cod_bcIm = cq%Im(:2,:)

      if(mpi_rnk==0) then
!            print*, '   codBC l=0 inner:', real(cod_bcRe(1,0))
!            print*, '   codBC l=0 outer:', real(cod_bcRe(2,0))
!            print*, '   cod_S min      :', real(minval(cod_S))
!            print*, '   cod_S max      :', real(maxval(cod_S))
      end if

      deallocate(fn )
      deallocate(roc)

      e=nf90_close(f)
#ifdef _MPI
      call mpi_barrier(mpi_comm_world, mpi_er)
#endif

   end subroutine io_load_codBC


!--------------------------------------------------------------------------
!  index of l,m mode in state file
!--------------------------------------------------------------------------
   subroutine io_state_nh(l,m_,iL,iM,iMp, nh)
      integer, intent(in)  :: l,m_,iL,iM,iMp
      integer, intent(out) :: nh
      integer :: m
      if(modulo(m_,iMp)/=0 .or. l>=iL .or. m_>=iM*iMp) then
         nh = -1
         return
      end if
      m  = m_/iMp
      nh = iL*m - iMp*(m-1)*m/2 + l-m_!+1 - 1
   end subroutine io_state_nh


!--------------------------------------------------------------------------
!  Load coll variable
!--------------------------------------------------------------------------



   subroutine io_load_coll(f,nm,interp,N__,r,W, a)
      type (coll),      intent(inout) :: a
      integer,          intent(in) :: f
      character(*),     intent(in) :: nm
      logical,          intent(in) :: interp
      integer,          intent(in) :: N__
      double precision, intent(in) :: r(N__)
      double precision, intent(in) :: W(2*i_KL+1,2*i_KL+1,*)
      double precision :: dRe(N__,i_pH1+1), dIm(N__,i_pH1+1)
      integer :: nh_,nh__,n
      integer :: L__, M__, Mp__      
      integer :: e,i,j
      _loop_lm_vars
         
      e=nf90_inq_varid(f,nm, i)
      e=nf90_get_att(f,i, 'L',  L__)
      e=nf90_get_att(f,i, 'M',  M__)
      e=nf90_get_att(f,i,'Mp', Mp__)
      if(mpi_rnk==0) then
         if(L__/=i_L) print*, nm, ' L :', L__,' --> ',i_L 
         if(M__/=i_M) print*, nm, ' M :', M__,' --> ',i_M 
         if(Mp__/=i_Mp .and. M__/=1)  &
                      print*, nm, ' Mp:',Mp__,' --> ',i_Mp     
      end if

      a%Re = 0d0
      a%Im = 0d0
      n = 1

      _loop_lm_begin(a)      
         call io_state_nh(l,m_,L__,M__,Mp__, nh__)
         if(nh__==-1) then
            ! mode is not in state file
            if(do_random_init .and. d_init_pert>0d0) then
               ! add noise to this mode
               call random_number(a%Re(:,nh))
               call random_number(a%Im(:,nh))
               a%Re(:,nh)=2d0*d_init_pert*(a%Re(:,nh)-0.5d0)
               a%Im(:,nh)=2d0*d_init_pert*(a%Im(:,nh)-0.5d0)
            end if
            n = 1
         else
            ! mode is in state file
            if(nh>=a%H%pH1) nh_ = -1
            if(nh <a%H%pH1) call io_state_nh(  &
               a%H%l(a%H%pH0+nh+1),a%H%m(a%H%pH0+nh+1),L__,M__,Mp__, nh_)
            if(nh__+1==nh_) then
               ! mode is contiguous with next mode in state file
               n = n + 1
            else
               ! last n modes were contiguous, load them!
               if(.not.interp) then
                  j = nh__+1 - (n-1)
                  e=nf90_get_var(f,i, a%Re(1:N__,nh-n+1:nh),start=(/1,j,1/))
                  e=nf90_get_var(f,i, a%Im(1:N__,nh-n+1:nh),start=(/1,j,2/))
               else
                  j = nh__+1 - (n-1)
                  e=nf90_get_var(f,i, dRe(1:N__,1:n),start=(/1,j,1/))
                  e=nf90_get_var(f,i, dIm(1:N__,1:n),start=(/1,j,2/))
                  do j = 1, n
                     call io_interp(N__,r,dRe(1,j),W,  &
                        a%D%N,a%D%r(1,1), a%Re(1,nh-n+j))
                     call io_interp(N__,r,dIm(1,j),W,  &
                        a%D%N,a%D%r(1,1), a%Im(1,nh-n+j))
                  end do
               end if
               n = 1
            end if
         end if
      _loop_lm_end      
   end subroutine io_load_coll


!--------------------------------------------------------------------------
!  interpolate; map both onto xrange [0,1] and return weights
!--------------------------------------------------------------------------
   subroutine io_interp_wts(ni,xi,no,xo, A)
      integer, parameter :: lda = 2*i_KL+1
      integer,          intent(in)  :: ni, no
      double precision, intent(in)  :: xi(ni), xo(no)
      double precision, intent(out) :: A(lda,lda,no) 
      double precision :: xi1(ni), xo1(no)
      integer :: n,nn,i,j,l,r
         			! map both onto [0,1] for convenience
      xi1 = (xi-xi(1))/(xi(ni)-xi(1))
      xo1 = (xo-xo(1))/(xo(no)-xo(1))
      
      do n = 1, no
         j = 1
         do while(xi1(j) < xo1(n)-1d-8)
           j = j+1
         end do
         l = max(1,j-i_KL)
         r = min(j+i_KL,ni)
         nn = r-l+1
         do i = 1, nn
            A(i,1,n) = 1d0
         end do
         do j = 2, nn
            do i = 1, nn
               A(i,j,n) = A(i,j-1,n) * (xi1(l+i-1)-xo1(n)) / dble(j-1) 
            end do
         end do
         call mes_mat_invert(nn,A(1,1,n),lda)
      end do

   end subroutine io_interp_wts


!--------------------------------------------------------------------------
!  interpolate, given weights from io_interp_wts()
!--------------------------------------------------------------------------
   subroutine io_interp(ni,xi,fi,A,no,xo, fo)
      integer, parameter :: lda = 2*i_KL+1
      integer,          intent(in)  :: ni, no
      double precision, intent(in)  :: xi(ni), fi(ni), xo(no)
      double precision, intent(in)  :: A(lda,lda,no) 
      double precision, intent(out) :: fo(no)
      double precision :: xi1(ni), xo1(no)
      integer :: n,nn,i,j,l,r
         			! map both onto [0,1] for convenience
      xi1 = (xi-xi(1))/(xi(ni)-xi(1))
      xo1 = (xo-xo(1))/(xo(no)-xo(1))
      
      do n = 1, no
         j = 1
         do while(xi1(j) < xo1(n)-1d-8)
           j = j+1
         end do
         l = max(1,j-i_KL)
         r = min(j+i_KL,ni)
         nn = r-l+1
         fo(n) = dot_product(A(1,1:nn,n),fi(l:r))
      end do

   end subroutine io_interp


!-------------------------------------------------------------------------
! extend into inner core:  f = a r^3 + b r^2  
! f=f'=0 at r=0, f,f' cts at icb
!-------------------------------------------------------------------------
   subroutine io_ic_extend(nm,oc,ic)
      character(*), intent(in)  :: nm
      type (coll),  intent(in)  :: oc
      type (coll),  intent(out) :: ic
      double precision :: ri,ri2,ri3, a,b,f,f1
      integer :: nhm,nh,j

      if(mpi_rnk==0)  print*, nm, ' extending B onto inner core'
      ri  = mes_oc%r(1,1)
      ri2 = mes_oc%r(1,2)
      ri3 = mes_oc%r(1,3)
      nhm = oc%H%pH1
      
      do nh = 0, nhm
         f = oc%Re(1,nh)
         f1 = 0d0
         do j = 1, 1+i_KL
            f1 = f1 + mes_oc%dr(1)%M(i_KL+1+1-j,j)*oc%Re(j,nh)
         end do
         a = ( f1*ri - 2d0*f ) / ri3
         b = ( f - a*ri3) / ri2
         ic%Re(:,nh) = a*mes_ic%r(:,3) + b*mes_ic%r(:,2)
      end do
      do nh = 0, nhm
         f = oc%Im(1,nh)
         f1 = 0d0
         do j = 1, 1+i_KL
            f1 = f1 + mes_oc%dr(1)%M(i_KL+1+1-j,j)*oc%Im(j,nh)
         end do
         a = ( f1*ri - 2d0*f ) / ri3
         b = ( f - a*ri3) / ri2
         ic%Im(:,nh) = a*mes_ic%r(:,3) + b*mes_ic%r(:,2)
      end do

   end subroutine io_ic_extend


!--------------------------------------------------------------------------
!  Open a netcdf file with parallel access
!--------------------------------------------------------------------------
#ifdef _PNCDF
   subroutine io_create_par(fname, f)
      character(*), intent(in)  :: fname
      integer,      intent(out) :: f
      integer :: e,info,cmode
      info =mpi_info_null
      cmode=ior(nf90_mpiio,nf90_clobber)
      cmode=ior(cmode,nf90_netcdf4)
      e=nf90_create_par(fname, cmode, mpi_comm_world, info, f)
      if(e/=nf90_noerr)  print*, nf90_strerror(e)
      if(e/=nf90_noerr)  stop
   end subroutine io_create_par
#endif

!--------------------------------------------------------------------------
!  Save state
!--------------------------------------------------------------------------
   subroutine io_save_state()
      character(4) :: cnum
      integer :: e, f
      integer :: H
      integer :: rd,icrd, Hd, ReImd, dims(3)
      integer :: dt, r,icr, omega, uT,uP,C,BT,BP,icBT,icBP
      integer :: info,cmode

      H = i_L*i_M - i_Mp*(i_M-1)*i_M/2 !- 1
            
      write(cnum,'(I4.4)') io_save1

      if(mpi_rnk==0) then
         print*, ' saving state'//cnum//'  t=', tim_t
#ifdef _PNCDF
      end if
      call io_create_par('state'//cnum//'.cdf.dat', f)
#else
         e=nf90_create('state'//cnum//'.cdf.dat', nf90_clobber, f)
#endif
         e=nf90_put_att(f, nf90_global,  't', tim_t)
         e=nf90_put_att(f, nf90_global, 'Ro', d_Ro)
         e=nf90_put_att(f, nf90_global,  'E', d_E)
         e=nf90_put_att(f, nf90_global, 'Ra', d_Ra)
         e=nf90_put_att(f, nf90_global,  'q', d_q)
         e=nf90_put_att(f, nf90_global, 'riro', d_rratio)

         e=nf90_def_dim(f,   'r', mes_oc%N, rd)
         e=nf90_def_dim(f, 'icr', mes_ic%N, icrd)
         e=nf90_def_dim(f,   'H', H, Hd)
         e=nf90_def_dim(f, 'ReIm',  2, ReImd)

         e=nf90_def_var(f,   'r', nf90_double, (/rd/), r)
         e=nf90_def_var(f, 'icr', nf90_double, (/icrd/), icr)
         e=nf90_def_var(f,       'dt', nf90_double, dt)
         e=nf90_def_var(f,  'icOmega', nf90_double, omega)

         dims = (/rd,Hd,ReImd/)
         call io_define_coll(f,  'uT',dims,vel_Tor, uT)
         call io_define_coll(f,  'uP',dims,vel_Pol, uP)
         call io_define_coll(f,   'C',dims,cod_C,   C)
         call io_define_coll(f,  'BT',dims,mag_Tor, BT)
         call io_define_coll(f,  'BP',dims,mag_Pol, BP)
         dims = (/icrd,Hd,ReImd/)
         call io_define_coll(f,'icBT',dims,mag_icTor, icBT)
         call io_define_coll(f,'icBP',dims,mag_icPol, icBP)

         e=nf90_enddef(f)
#ifdef _PNCDF
      if(mpi_rnk==0) then
#endif
         e=nf90_put_var(f,     r, mes_oc%r(1:i_N,1))
         e=nf90_put_var(f,   icr, mes_ic%r(1:i_Nic,1))
         e=nf90_put_var(f,    dt, tim_dt)
         e=nf90_put_var(f, omega, rot_omega)
      end if

      call io_save_coll(f, uT,i_N,vel_Tor)
      call io_save_coll(f, uP,i_N,vel_Pol)
      call io_save_coll(f,  C,i_N,cod_C)
      call io_save_coll(f, BT,i_N,mag_Tor)
      call io_save_coll(f, BP,i_N,mag_Pol)
      call io_save_coll(f, icBT,i_Nic,mag_icTor)
      call io_save_coll(f, icBP,i_Nic,mag_icPol)

#ifndef _PNCDF
      if(mpi_rnk==0)  &
#endif
         e=nf90_close(f)

   end subroutine io_save_state
 

!--------------------------------------------------------------------------
!  Save imposed field
!--------------------------------------------------------------------------
   subroutine io_save_magB0()
      integer :: e, f
      integer :: H
      integer :: rd,icrd, BTHd,BPHd, ReImd
      integer :: r,icr, BT,BP,icBT,icBP

      H = i_L*i_M - i_Mp*(i_M-1)*i_M/2 !- 1
            
      if(_Np/=1) stop 'io_save_magB0: set _Np 1'
      if(mpi_rnk==0) then
         print*, ' saving magB0.cdf.dat'
         e=nf90_create('magB0.cdf.dat', nf90_clobber, f)

         e=nf90_def_dim(f,   'r', mes_oc%N, rd)
         e=nf90_def_dim(f, 'icr', mes_ic%N, icrd)
         e=nf90_def_dim(f, 'BTH', H, BTHd)
         e=nf90_def_dim(f, 'BPH', H, BPHd)
         e=nf90_def_dim(f, 'ReIm',  2, ReImd)

         e=nf90_def_var(f,   'r', nf90_double, (/rd/), r)
         e=nf90_def_var(f, 'icr', nf90_double, (/icrd/), icr)

         call io_define_coll(f,  'B0T',(/rd,BTHd,ReImd/),mag_B0Tor, BT)
         call io_define_coll(f,  'B0P',(/rd,BPHd,ReImd/),mag_B0Pol, BP)
         call io_define_coll(f,'icB0T',(/rd,BTHd,ReImd/),mag_icB0Tor, icBT)
         call io_define_coll(f,'icB0P',(/rd,BPHd,ReImd/),mag_icB0Pol, icBP)

         e=nf90_enddef(f)

         e=nf90_put_var(f,     r, mes_oc%r(1:i_N,1))
         e=nf90_put_var(f,   icr, mes_ic%r(1:i_Nic,1))
      end if

      call io_save_coll(f, BT,i_N,mag_B0Tor)
      call io_save_coll(f, BP,i_N,mag_B0Pol)
      call io_save_coll(f, icBT,i_Nic,mag_icB0Tor)
      call io_save_coll(f, icBP,i_Nic,mag_icB0Pol)

      if(mpi_rnk==0)  &
         e=nf90_close(f)

   end subroutine io_save_magB0
 

!--------------------------------------------------------------------------
!  Save codensity boundary condition and source info
!--------------------------------------------------------------------------
   subroutine io_save_codBC()
      integer :: e, f
      integer :: H
      integer :: rd, uPHd, ReImd, rirod
      integer :: r, S,Cbc

      H = i_L*i_M - i_Mp*(i_M-1)*i_M/2 !- 1
            
      if(_Np/=1) stop 'io_save_codBC: set _Np 1'
      if(mpi_rnk==0) then
         print*, ' saving codBC.cdf.dat'
         e=nf90_create('codBC.cdf.dat', nf90_clobber, f)

         e=nf90_def_dim(f,   'r', mes_oc%N, rd)
         e=nf90_def_dim(f, 'uPH', H, uPHd)
         e=nf90_def_dim(f, 'ReIm',  2, ReImd)
         e=nf90_def_dim(f, 'riro',  2, rirod)

         e=nf90_def_var(f,   'r', nf90_double, (/rd/), r)
         e=nf90_def_var(f,   'S', nf90_double, (/rd/), S)

         call io_define_coll(f, 'Cbc',(/rirod,uPHd,ReImd/),cod_C, Cbc)

         e=nf90_enddef(f)

         e=nf90_put_var(f,     r, mes_oc%r(1:i_N,1))
         e=nf90_put_var(f,     S, cod_S(1:i_N))
      end if

      call var_coll_init(mes_oc,var_H, cq)
      cq%Re(:2,:) = cod_bcRe
      cq%Im(:2,:) = cod_bcIm
      call io_save_coll(f,Cbc,2,cq)

      if(mpi_rnk==0)  &
         e=nf90_close(f)

   end subroutine io_save_codBC
 

!--------------------------------------------------------------------------
!  Save coll variable
!--------------------------------------------------------------------------
   subroutine io_define_coll(f,nm,dims,a, id)
      integer,      intent(in) :: f, dims(3)
      character(*), intent(in) :: nm
      type (coll),  intent(in) :: a
      integer, intent(out) :: id
      integer :: e
      e=nf90_def_var(f, nm, nf90_double, dims, id)
      e=nf90_put_att(f, id,  'L', i_L)      
      e=nf90_put_att(f, id,  'M', i_M)
      e=nf90_put_att(f, id, 'Mp', i_Mp)
   end subroutine io_define_coll

!- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   subroutine io_save_coll(f,id,N__,a)
      integer,     intent(in) :: f, id, N__
      type (coll), intent(in) :: a
      integer :: e, n,j,nh_,nh__
      integer :: r,rr, pH0,pH1
      integer :: nh, l(0:i_pH1), m_(0:i_pH1)

#ifdef _PNCDF
      logical :: nf90coll
      nf90coll = .true.
      call mpi_barrier(mpi_comm_world, mpi_er)
      e=nf90_var_par_access(f,id, nf90_collective)
      if(e/=nf90_noerr)  print*, nf90_strerror(e)
      if(e/=nf90_noerr)  stop
      pH0 = a%H%pH0
      pH1 = a%H%pH1
      ct%Re = a%Re
      ct%Im = a%Im
      l (0:pH1) = a%H%l(pH0:pH0+pH1)
      m_(0:pH1) = a%H%m(pH0:pH0+pH1)
#else
      if(mpi_rnk==0) then
         do r = 0, mpi_sze-1
            rr  = modulo(r,_Nr)
            pH0 = a%H%pH0_(rr)
            pH1 = a%H%pH1_(rr)
            if(r==0) then
               ct%Re = a%Re
               ct%Im = a%Im
               l  = a%H%l(0:i_pH1)
               m_ = a%H%m(0:i_pH1)
#ifdef _MPI
            else
               mpi_tg = r
               call mpi_recv( ct%Re, i_N*(pH1+1), mpi_double_precision,  &
                  r, mpi_tg, mpi_comm_world, mpi_st, mpi_er)
               call mpi_recv( ct%Im, i_N*(pH1+1), mpi_double_precision,  &
                  r, mpi_tg, mpi_comm_world, mpi_st, mpi_er)
               call mpi_recv( l,  pH1+1, mpi_integer,  &
                  r, mpi_tg, mpi_comm_world, mpi_st, mpi_er)
               call mpi_recv( m_, pH1+1, mpi_integer,  &
                  r, mpi_tg, mpi_comm_world, mpi_st, mpi_er)
#endif
            end if
#endif
            n = 1
            do nh = 0, pH1
               call io_state_nh(l(nh),m_(nh),i_L,i_M,i_Mp, nh__)
               if(nh==pH1) nh_ = -1
               if(nh <pH1)  &
                  call io_state_nh(l(nh+1),m_(nh+1),i_L,i_M,i_Mp, nh_)
               if(nh__+1==nh_) then  ! mode contiguous with next in file
                  n = n + 1
               else                  ! last n modes ctgs, save them!
                  j = nh__+1 - (n-1)
                  e=nf90_put_var(f,id, ct%Re(1:N__,nh-n+1:nh),start=(/1,j,1/))
                  e=nf90_put_var(f,id, ct%Im(1:N__,nh-n+1:nh),start=(/1,j,2/))
                  n = 1
#ifdef _PNCDF
                  if(nf90coll) e=nf90_var_par_access(f,id, nf90_independent)
                  nf90coll = .false.
#endif _PNCDF
               end if
            end do
#ifndef _PNCDF
         end do
#ifdef _MPI
      else
         mpi_tg = mpi_rnk
         call mpi_send( a%Re, i_N*(a%H%pH1+1), mpi_double_precision,  &
            0, mpi_tg, mpi_comm_world, mpi_er)
         call mpi_send( a%Im, i_N*(a%H%pH1+1), mpi_double_precision,  &
            0, mpi_tg, mpi_comm_world, mpi_er)
         call mpi_send( a%H%l(a%H%pH0), a%H%pH1+1, mpi_integer,  &
            0, mpi_tg, mpi_comm_world, mpi_er)
         call mpi_send( a%H%m(a%H%pH0), a%H%pH1+1, mpi_integer,  &
            0, mpi_tg, mpi_comm_world, mpi_er)
#endif
      end if
#endif

   end subroutine io_save_coll


!--------------------------------------------------------------------------
!  write to energy file
!--------------------------------------------------------------------------
   subroutine io_write_energy(ifile,Tor,Pol,Efac,Dfac)
      integer, intent(in) :: ifile
      type (coll), intent(in) :: Tor, Pol
      double precision, intent(in) :: Efac, Dfac
      double precision :: Ee,Eo,E1,E0, E,ETor,ESym,El1,ETm0, EDis

      call var_coll_TorPol2qst(Tor,Pol, cq,cs,ct)
      call var_coll_norm(cq, Eo,Ee,E1,E0)
      E    = Eo + Ee
      ESym = Ee
      El1  = E1
      call var_coll_norm(cs, Eo,Ee,E1,E0)
      E    = E    + Eo + Ee
      ESym = ESym + Ee
      El1  = El1  + E1
      call var_coll_norm(ct, Eo,Ee,E1,E0)
      ETor = Eo + Ee
      E    = E  + ETor
      ESym = ESym + Eo
      El1  = El1  + E1
      ETm0 = E0
      
      call var_coll_qstcurl(cq,cs,ct, cq,cs,ct)
      call var_coll_norm(cq, Eo,Ee,E1,E0)
      EDis = Eo + Ee
      call var_coll_norm(cs, Eo,Ee,E1,E0)
      EDis = EDis + Eo + Ee
      call var_coll_norm(ct, Eo,Ee,E1,E0)
      EDis = EDis + Eo + Ee
            
      E    = Efac * E 
      ETor = Efac * ETor
      ESym = Efac * ESym
      El1  = Efac * El1
      ETm0 = Efac * ETm0
      EDis = Dfac * EDis
      
      if(mpi_rnk/=0) return
      write(ifile,'(7e20.12)')  tim_t, E, ETor, ESym, El1, ETm0, EDis

   end subroutine io_write_energy


!--------------------------------------------------------------------------
!
!--------------------------------------------------------------------------
   subroutine io_write_cmb(ifile,Tor,Pol,Efac)
      integer, intent(in) :: ifile
      type (coll), intent(in) :: Tor, Pol
      double precision, intent(in) :: Efac
      double precision :: Eo,Ee,E1,E0, EPol,ESym,El1, g10,g11,h11,th,ph
      integer :: N_, L_, r,rt,nh_
      _loop_lm_vars
      
      call var_coll_TorPol2qst(Tor,Pol, cq,cs,ct)
      N_ = mes_oc%N
      cq%Re(:N_-1,:) = 0d0
      cq%Im(:N_-1,:) = 0d0
      cs%Re(:N_-1,:) = 0d0
      cs%Im(:N_-1,:) = 0d0
      call var_coll_norm(cq, Eo,Ee,E1,E0)
      EPol = Eo + Ee
      ESym = Ee
      El1  = E1
      call var_coll_norm(cs, Eo,Ee,E1,E0)
      EPol = EPol + Eo + Ee
      ESym = ESym + Ee
      El1  = El1  + E1
      EPol = Efac * EPol * mes_oc%r(N_,2) / mes_oc%intr2dr(N_)
      ESym = Efac * ESym * mes_oc%r(N_,2) / mes_oc%intr2dr(N_)
      El1  = Efac * El1  * mes_oc%r(N_,2) / mes_oc%intr2dr(N_)
      
      r  = 0
      nh_= 0
      _loop_lm_begin(Pol)
         if(l==1 .and. m_==0) then
            r  = mpi_rnk
            nh_= nh
         end if
      _loop_lm_end
      g10 = Pol%Re(N_,nh_)
#ifdef _MPI
      call mpi_allreduce(r, rt, 1, mpi_integer,  &
         mpi_sum, mpi_comm_world, mpi_er)
      call mpi_bcast(g10, 1, mpi_double_precision,  &
         rt, mpi_comm_world, mpi_er)
#endif
      
      if(i_Mp==1 .and. i_M1>0) then
         r  = 0
         nh_= 0
         _loop_lm_begin(Pol)
            if(l==1 .and. m_==1) then
               r  = mpi_rnk
               nh_= nh
            end if
         _loop_lm_end
         g11 = -2d0*Pol%Re(N_,nh_)
         h11 =  2d0*Pol%Im(N_,nh_)
#ifdef _MPI
         call mpi_allreduce(r, rt, 1, mpi_integer,  &
            mpi_sum, mpi_comm_world, mpi_er)
         call mpi_bcast(g11, 1, mpi_double_precision,  &
            rt, mpi_comm_world, mpi_er)
         call mpi_bcast(h11, 1, mpi_double_precision,  &
            rt, mpi_comm_world, mpi_er)
#endif            
         th = dsqrt(g10*g10 + g11*g11 + h11*h11)
         if(th/=0d0)  th = dacos( g10 / max(th,dabs(g10)) )
         if(dabs(g11)<1d-99) then
            if(h11>=0d0) ph = 0.5d0*d_Pi
            if(h11 <0d0) ph = 1.5d0*d_Pi
         else
            ph = datan( h11/g11 )
            if(g11<0d0) ph = ph + d_Pi
            if(ph <0d0) ph = ph + 2d0*d_Pi
         endif
      else
         if(g10>=0d0) th = 0d0
         if(g10< 0d0) th = d_Pi
         ph = 0d0
      end if
          
      if(mpi_rnk/=0) return
      write(ifile,'(6e20.12)') tim_t, EPol, ESym, El1, th, ph

   end subroutine io_write_cmb


!--------------------------------------------------------------------------
!  write the Nusselt number:  actual flux / flux for no flow  at cmb
!--------------------------------------------------------------------------
   subroutine io_write_nussel(ifile, C)
      integer,     intent(in) :: ifile
      type (coll), intent(in) :: C
      double precision, save  :: cflx = 0d0
      double precision :: flx
      integer :: N, j
      if(mpi_rnk/=0) return
      if(C%H%pH0>0) stop 'io_write_nussel'
      if(cflx==0d0) call cod_cmbflux(d_q, cflx)
      N = C%D%N
      flx = 0d0
      do j = N-i_KL, N
         flx = flx  +  C%Re(j,0) * C%D%dr(1)%M(i_KL+1+N-j,j)
      end do
      write(ifile,'(2e20.12)') tim_t, flx/cflx
   end subroutine io_write_nussel


!--------------------------------------------------------------------------
!  write to torque file
!--------------------------------------------------------------------------
   subroutine io_write_torque(ifile)
      integer, intent(in) :: ifile
      if(mpi_rnk/=0) return
      write(ifile,'(4e20.12)') tim_t, rot_omega, rot_velTorq, rot_magTorq
   end subroutine io_write_torque


!--------------------------------------------------------------------------
!  write to torque file
!--------------------------------------------------------------------------
   subroutine io_write_bench(ifile)
      integer, intent(in) :: ifile
      double precision :: dat(6)
      double precision, save :: t=-1d0,t_
      double complex,   save :: c_, c=0d0
      integer :: Nr,Nth, r,r_,rr,rth
      type (spec) :: sC
      type (phys) :: pC
      
      call var_coll2spec(cod_C, sC)
      call tra_spec2phys(sC, pC)
      call var_coll2spec(vel_Pol, sC)
      Nr  = (mes_oc%N+1)/2
      Nth = (i_Th+1)/2
      do r = 0, mpi_sze-1
         rr  = modulo(r,_Nr)
         rth = r/_Nr
         if(mes_oc%pNi_(rr)<=Nr .and.  &
            mes_oc%pNi_(rr)+mes_oc%pN_(rr)-1>=Nr .and.  &
            var_Th%pThi_(rth)<=Nth .and.  &
            var_Th%pThi_(rth)+var_Th%pTh_(rth)-1>=Nth )  &
            r_ = r
      end do
      t_   = t
      t    = tim_t
      if(r_==mpi_rnk) then
         Nr = Nr - mes_oc%pNi + 1
         Nth= Nth- var_Th%pThi + 1
         dat(1) = vel_r%Re(Nth,0,Nr)	! u_r
         dat(2) = vel_p%Re(Nth,0,Nr)	! u_ph
         dat(3) =    pC%Re(Nth,0,Nr)	! C
         dat(4) = mag_t%Re(Nth,0,Nr)	! B_th
!         c_   = c
!         c    = cmplx( sC%Re(i_L,Nr), sC%Im(i_L,Nr) )
!         if(abs(c_)/=0d0)  dat(5) =  real(log(c/c_))/(t-t_)  ! growth
!         if(abs(c_)/=0d0)  dat(6) = aimag(log(c/c_))/(t-t_)  ! drift*Mp
      end if
      if(t_==-1d0) return
#ifdef _MPI
      call mpi_bcast(dat, 6, mpi_double_precision,  &
                     r_, mpi_comm_world, mpi_er)
#endif
      if(mpi_rnk/=0) return
      write(ifile,'(7e20.12)') tim_t, (dat(r), r=1,4) !6)
   end subroutine io_write_bench


!--------------------------------------------------------------------------
!  write to timestep file
!--------------------------------------------------------------------------
   subroutine io_write_timestep(ifile)
      integer, intent(in) :: ifile
      if(mpi_rnk/=0) return
      write(ifile,'(4e20.12)') tim_t, tim_dt, tim_corr_dt, tim_cfl_dt
   end subroutine io_write_timestep


!--------------------------------------------------------------------------
!  save spectrum
!--------------------------------------------------------------------------
   subroutine io_save_spectrum(prefix,Tor,Pol,Efac)
      character(3), intent(in) :: prefix
      type (coll), intent(in) :: Tor, Pol
      double precision, intent(in) :: Efac
      double precision :: ETh(0:2*i_KL), ETl(0:i_L1), ETm(0:i_M1)
      double precision :: EPh(0:2*i_KL), EPl(0:i_L1), EPm(0:i_M1)
      integer :: i
   10 format(i4,1e20.12)

      if(mpi_rnk==0)  &
         open(11, status='unknown', file=prefix//'_spectrum.dat')

      call var_coll_TorPol2qst(Tor,Pol, cq,cs,ct)

      if(mpi_rnk==0)  &
         write(11,*) '# t = ', tim_t

      call var_coll_normlm(ct, ETh,ETl,ETm)
      call var_coll_normlm(cq, EPh,EPl,EPm)
      ETl = ETl + EPl
      ETm = ETm + EPm
      call var_coll_normlm(cs, EPh,EPl,EPm)
      ETl = Efac * (ETl + EPl)
      ETm = Efac * (ETm + EPm)

      if(mpi_rnk==0) then
         write(11,*) '# l'
         do i = 1, i_L1
            write(11,10) i, ETl(i)
         end do
         write(11,*)
         write(11,*) '# m'
         do i = 0, i_M1
            write(11,10) i*i_Mp, ETm(i)
         end do
         close(11)
      end if

   end subroutine io_save_spectrum


   subroutine io_save_spectrum1(prefix,a,Efac)
      character(3), intent(in) :: prefix
      type (coll),  intent(in) :: a
      double precision, intent(in) :: Efac
      double precision :: Eh(0:2*i_KL), El(0:i_L1), Em(0:i_M1)
      integer :: i
   10 format(i4,1e20.12)

      if(mpi_rnk==0)  &
         open(11, status='unknown', file=prefix//'_spectrum.dat')

      if(mpi_rnk==0)  &
         write(11,*) '# t = ', tim_t

      call var_coll_normlm(a, Eh,El,Em)
      El = Efac * El
      Em = Efac * Em

      if(mpi_rnk==0) then
         write(11,*) '# l'
         do i = 1, i_L1
            write(11,10) i, El(i)
         end do
         write(11,*)
         write(11,*) '# m'
         do i = 0, i_M1
            write(11,10) i*i_Mp, Em(i)
         end do
         close(11)
      end if

   end subroutine io_save_spectrum1


!**************************************************************************
 end module io
!**************************************************************************

