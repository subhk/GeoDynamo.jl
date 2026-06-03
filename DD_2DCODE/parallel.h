#ifndef parallel_h
#define parallel_h

!/***************************************************************/
#define _Nr  48
#define _Nth 45
!/* #define _PAR_IO */
!/***************************************************************/

!/*-------------------------------------------------------------*/
#define _Np (_Nr*_Nth)
#if _Np != (1*1)
#define _MPI
#endif

#if defined(_MPI) && defined(_PAR_IO)
#define _PNCDF
#endif
!/*-------------------------------------------------------------*/
#define _loop_lm_vars  \
   integer :: nh,l,m_
   
#define _loop_lm_begin(a)  \
   do nh = 0, a%H%pH1;  \
      l  = a%H%l(a%H%pH0+nh);  \
      m_ = a%H%m(a%H%pH0+nh);
          
#define _loop_lm_end  \
   end do

!/*-------------------------------------------------------------*/
#define _Hc  ((i_M+1)/2)
#define _Hr  (i_H/_Hc)
#define _pHc (_Hc/_Nth)

#define _loop_setlm_vars  \
   integer :: nh,l,m,m_,i__,j__,jo__

#define _loop_setlm_begin  \
   nh = -1;  \
   jo__ = (mpi_rnk/_Nr)*_pHc;  \
   do j__ = jo__+1, jo__+_pHc;  \
      do i__ = 1, _Hr;  \
         m = j__-1;  \
         l = m*i_Mp+i__-1;  \
         if(l>=i_L)  m = 2*_Hc-j__;  \
         if(l>=i_L)  l = i_L-_Hr+i__-1;  \
         m_ = m*i_Mp;  \
         nh = nh + 1;
   
#define _loop_setlm_end  \
     end do;  \
   end do

!/*-------------------------------------------------------------*/
#define _loop_sumth_vars  \
   integer :: nh,m,m_,m__

#define _loop_sumth_begin  \
   m  = -1;  \
   m_ = -1;  \
   do nh = 0, i_H1;  \
      m__ = m_;  \
      m_  = var_H%m(nh);  \
      if(m_/=m__) m = m + 1;  \
          
#define _loop_sumth_end  \
   end do

!/*-------------------------------------------------------------*/
#endif /* ndef parallel_h */

