!*************************************************************************
! main.f90 (executable)
!*************************************************************************
!
!
!*************************************************************************
#include "../parallel.h"
 PROGRAM MAIN
!*************************************************************************
   use mpi
   use meshs
   use legendre
   use transform
   use velocity
   use temperature
   use composition
   use magnetic
   use rotation
   use nonlinear
   use timestep
   use io
   implicit none
   real :: d_start, d_stop


   call initialise()
  
   
      							! main loop
   do while(tim_it/=0 .or. .not. terminate())
      if(b_vel_tstep) call vel_transform()
      if(b_tmp_tstep) call tmp_transform()
      if(b_cmp_tstep) call cmp_transform()
      if(b_mag_tstep) call mag_transform()
      if(b_rot_tstep) call non_rotation()
      if(b_vel_tstep) call non_velocity()
      if(b_tmp_tstep) call non_temperature()
      if(b_cmp_tstep) call non_composition()
      if(b_mag_tstep) call non_magnetic()


      if(tim_it==0) then
         if(d_timestep<0d0) then
            call non_maxtstep()
            call tim_new_tstep()
            if(tim_new_dt) then
               call vel_matrices()
               call tmp_matrices()
               call cmp_matrices()
               call mag_matrices()
            endif  
         end if 
      
         if(b_rot_tstep) call rot_predictor()
         if(b_vel_tstep) call vel_predictor()
         if(b_tmp_tstep) call tmp_predictor()
         if(b_cmp_tstep) call cmp_predictor()
         if(b_mag_tstep) call mag_predictor()
         tim_it = 1
      else
         if(b_rot_tstep) call rot_corrector()
         if(b_vel_tstep) call vel_corrector()
         if(b_tmp_tstep) call tmp_corrector()
         if(b_cmp_tstep) call cmp_corrector()
         if(b_mag_tstep) call mag_corrector()
         call tim_check_cgce()
      end if
     
 
      if(tim_it==0) then
         tim_t    = tim_t    + tim_dt
         tim_step = tim_step + 1
         call io_write2files()
      end if
   end do
      							! end main loop   

   call cleanup()
   stop

 contains


!-------------------------------------------------------------------------
!  Termaination conditions
!-------------------------------------------------------------------------
   logical function terminate()
      logical :: file_exist
            
      if(mpi_rnk==0) then
         terminate = .false.
      
         if(tim_step==i_maxtstep) then
            terminate = .true.
            print*, 'maxtstep reached!'
         end if

         call cpu_time(d_stop)
         if(dble(d_stop-d_start)/36d2 >= d_cpuhours) then
            terminate = .true.
            print*, 'cpuhours reached!'
         end if

         if( modulo(tim_step,i_save_rate2)==0) then
            inquire(file='RUNNING', exist=file_exist)
            if(.not. file_exist) then
               terminate = .true.
               print*, 'RUNNING deleted !'
            end if
         end if
      end if

#ifdef _MPI
      call mpi_bcast(terminate,1,mpi_logical, 0,mpi_comm_world,mpi_er)
#endif

   end function terminate


!-------------------------------------------------------------------------
!  Initialisation
!-------------------------------------------------------------------------
   subroutine initialise()
      logical :: file_exist

      call mpi_precompute()
      if(mpi_rnk==0) then
         call system('touch PRECOMPUTING')
         call system('echo $HOSTNAME > HOST')
      end if

      if(mpi_rnk==0)  print*, 'precomputing function requisites...'
      call par_precompute()
      call mes_precompute()
      call leg_precompute()
      call var_precompute()
      call tra_precompute(1)
      call tim_precompute()
      call rot_precompute()
      call vel_precompute()
      call tmp_precompute()
      call cmp_precompute()
      call mag_precompute()
      call non_precompute()
      call io_precompute()
   
      if(mpi_rnk==0)  print*,    'loading tmpBC...'
      call io_load_tmpBC()
      if(mpi_rnk==0)  print*,    'loading cmpBC...'
      call io_load_cmpBC()
      if(b_mag_impose) then
         if(mpi_rnk==0)  print*, 'loading magB0...'
         call io_load_magB0()
      end if

      if(mpi_rnk==0)  print*,    'loading state...'
      call io_load_state()
      call vel_transform()
      call tmp_transform()
      call cmp_transform()
      call mag_transform()
      call non_rotation()
      call non_maxtstep()

      if(mpi_rnk==0)  print*, 'computing timestepping matrices...'
      call vel_matrices()
      call tmp_matrices()
      call cmp_matrices()
      call mag_matrices()

      if(mpi_rnk==0)  print*, 'initialising output files...'
      call io_openfiles()
      call io_write2files()

      if(mpi_rnk==0) then
         open(99, file='PRECOMPUTING')
         close(99, status='delete')

         open(99, file='RUNNING')
         write(99,*) 'This file indicates a job running in this directory.'
         write(99,*) 'Delete this file to cleanly terminate the process.'
         close(99)
         print*, 'timestepping.....'
      end if
      
      call cpu_time(d_start)
   
   end subroutine initialise

!-------------------------------------------------------------------------
!  Program closure
!-------------------------------------------------------------------------
   subroutine cleanup()
      logical :: file_exist
   
      if(mpi_rnk==0) then
         print*, 'cleanup...'
         call cpu_time(d_stop)
         print*, ' sec/step = ', (d_stop-d_start)/real(tim_step)
         print*, ' cpu time = ', int((d_stop-d_start)/6d1), ' mins.'
      end if
      
      call io_save_state()
      call io_closefiles()

      if(mpi_rnk==0) then
         inquire(file='RUNNING', exist=file_exist)
         if(file_exist) open(99, file='RUNNING')
         if(file_exist) close(99, status='delete')
      end if      

#ifdef _MPI
      call mpi_barrier(mpi_comm_world, mpi_er)
      call mpi_finalize(mpi_er)
#endif
      if(mpi_rnk==0) print*, '...done!'

   end subroutine cleanup


         
!*************************************************************************
 END PROGRAM MAIN
!*************************************************************************

