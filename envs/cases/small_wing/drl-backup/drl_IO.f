c==============================================
c I/O for saving/loading relevent file for opposition control  
c Yuning Wang 
c==============================================

c******************************************************
c           STATE/OBSERVATION 
c******************************************************

c-----------------------------------------------------------------------
cc YW: WRITE the Reward for agents
            subroutine drl_reward_out
c=============================================
c       Define variable
c=============================================
            implicit none 
            include 'SIZE'
            ! include 'INPUT'
            include 'DRL'       
            include 'TSTEP'  
            include 'NEKUSE'  
            include 'mpif.h'
!     arguments
            integer npts              ! local number of points
            integer parent_comm,ierr
      !     local variables
            real send_buff(totctrl) ! buffer for snapshots
            real tmlist(drlsnap)       ! snapshot time
            integer istcount          ! step interval for data collection
            integer il,jl,kl        ! For iteration
            integer icl             ! call counter for control 
            character*3 prefix
            integer isize   
c=============================================
c       Function
c=============================================
            call MPI_COMM_GET_PARENT(parent_comm,ierr)
            if(NUMCTRL.ne.0) then 
            call copy(send_buff(1),vwall(1),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_DOUBLE, 
     $                  0, NID+80000, 
     $                  parent_comm, ierr)
            else ! 
            call copy(send_buff(1),vwall(1),TOTCTRL)
            endif ! if(NUMCTRL.ne.0)
      
            return 
            end subroutine drl_reward_out
c-----------------------------------------------------------------------




c-----------------------------------------------------------------------
!     Output points for statistics point time history
!     To reduce number of wirting to the disc I collect some number of 
!     time snapshots
!     and later on write the whole set to the disc
      subroutine drl_state_out()
c=============================================
c       Define variable
c=============================================
      implicit none
      include 'SIZE'
      include 'TSTEP'
      include 'INPUT'
      include "DRL"
      include 'mpif.h'
!     arguments
      integer npts              ! local number of points
      integer parent_comm,ierr
!     local variables
      real send_buff(totctrl) ! buffer for snapshots
      real c_time             ! Current time 
      integer istcount          ! step interval for data collection
      integer il,jl,kl        ! For iteration
      integer icl             ! call counter for control 
      character*3 prefix
      integer isize             !
      ! parameter (isize = NFLDC*totctrl)

!     collect data
!     count calls
c=============================================
c       Function
c=============================================
cc NOTE: This is a ONE-BATCH file without accumulating!
      ! USE MPI4PY Get parent here 
      call MPI_COMM_GET_PARENT(parent_comm,ierr)
      
      ! Hand-shake, sending the current time step 
      c_time = TIME 
      if (NID.eq.0) then 
        call MPI_SEND(c_time, 1, MPI_DOUBLE, 
     $                  0, 1998, 
     $                  parent_comm, ierr)
      else 
        c_time = TIME 
      endif 

      call nekgsync()
      if (NID.eq.0) print *, "[STATE] TIME GET",c_time


      if(NUMCTRL.ne.0) then 
      do il=1,NFLDC 
      
      call copy(send_buff(1),vctl(il,:),TOTCTRL)

      call MPI_SEND(send_buff, TOTCTRL, MPI_DOUBLE, 
     $                  0, NID*il+70000, 
     $                  parent_comm, ierr)

      ! call mpi_barrier(MPI_COMM_WORLD,ierr)

      ! print *,"[STATE] TRANSFER",NID
      enddo 
      else ! if numctrl .ne.0
      call copy(send_buff(1),vctl(il,:),TOTCTRL)
            
      endif 
      ! call mpi_barrier(MPI_COMM_WORLD,ierr)
      if (NID.eq.0) print *, "[STATE] BUFFER TRANSFERRED"

      end subroutine drl_state_out
!---------------------------------------------------------------------------


c-----------------------------------------------------------------------
cc YW: WRITE THE NID, IGLLID, IFACE for sorting DRL
            subroutine drl_info_out
c=============================================
c       Define variable
c=============================================
            implicit none 
            include 'SIZE'
            ! include 'INPUT'
            include 'DRL'         
            include 'mpif.h'
            integer ierr,parent_comm
            integer k,il,jl        ! Iteration
            integer len,recctrl ! Flag for counting 
cc YW: Test for MPI_ISEND & MPI_IRecv 
            integer request_send, request_recv, status(mpi_status_size)
            integer send_buff(totctrl) ! The pair to send and recevie
            integer node_list(LP)
            save node_list
            real send_r_buff(TOTCTRL)

            integer recv_buff(totctrl) ! The pair to send and recevie
            
            logical ifexist
            integer tot_ctrl,iglsum
            character*13 fNAME
            parameter(fNAME="NODE_INFO.dat")
c=============================================
c       Function
c=============================================
            ! tot_ctrl = iglsum(NUMCTRL,1)
            ! len  = nfeat*(totctrl) ! We send and recv the full array
            ! ierr = 0  
            
            ! USE MPI4PY Get parent here 
            call MPI_COMM_GET_PARENT(parent_comm,ierr)
            
            call count_total_ctrlpts(node_list)
! #ifdef YWDEBUG
!             if (NID.eq.0) print *,node_list
! #endif
            ! print *, node_list
            if (NID.eq.0) then 
              call MPI_SEND(node_list,LP,MPI_INTEGER,
     $                     0,1996,
     $                     parent_comm,ierr)
            print *, "[NEK] SEND HAND-SHAKE"
            else
            endif 
            
            ! call nekgsync()
            
!--------------------GET CONTROL INFO------------------------------
            if (NUMCTRL.ne.0) then
            ! NUMCTRL
            k=0
            call MPI_SEND(NUMCTRL, 1, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)
            ! GLOB-ID
            k=k+1
            call icopy(send_buff(1),grdwall(k,:),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)
            
            ! faceid 
            k=k+1 
            call icopy(send_buff(1),grdwall(k,:),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)

            ! ix
            k=k+1 
            call icopy(send_buff(1),grdwall(k,:),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)
            
            ! iy
            k=k+1 
            call icopy(send_buff(1),grdwall(k,:),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)

            ! iz
            k=k+1 
            call icopy(send_buff(1),grdwall(k,:),TOTCTRL)
            call MPI_SEND(send_buff, TOTCTRL, MPI_INTEGER, 
     $                  0, NID+10000*(k+1), 
     $                  parent_comm, ierr)


!--------------------GET COORINATE INFO------------------------------
            
            k=1
            call copy(send_r_buff(1),crdwall(k,:),TOTCTRL)
            call MPI_SEND(send_r_buff, TOTCTRL, MPI_DOUBLE, 
     $                  0, NID+100000*k, 
     $                  parent_comm, ierr)
            k=k+1
            call copy(send_r_buff(1),crdwall(k,:),TOTCTRL)
            call MPI_SEND(send_r_buff, TOTCTRL, MPI_DOUBLE, 
     $                  0, NID+100000*k, 
     $                  parent_comm, ierr)

            k=k+1 
            call copy(send_r_buff(1),crdwall(k,:),TOTCTRL)
            call MPI_SEND(send_r_buff, TOTCTRL, MPI_DOUBLE, 
     $                  0, NID+100000*k, 
     $                  parent_comm, ierr)
            endif 
            
            call nekgsync()
            
            if (NID.eq.0) print *,"[NEK] Finish SEND NODE INFO"
            
            end subroutine drl_info_out
c-----------------------------------------------------------------------




c-----------------------------------------------------------------------
      subroutine count_total_ctrlpts(listp1)
c=============================================
c       Define variable
c=============================================
      implicit none

cc MA:      include 'SIZE_DEF'        ! missing definitions in include files
      include 'SIZE'
cc MA:      include 'RESTART_DEF'
      include 'RESTART'
cc MA:      include 'PARALLEL_DEF'
      include 'PARALLEL'
cc MA:      include 'TSTEP_DEF'
      include 'TSTEP'
      include 'DRL'          ! Common block for opposition control 

      integer listp1(LP)           ! for Summup the control points
      integer listp2(LP)           ! For summup the control points
      integer j 
c=============================================
c     Function
c=============================================
      ! call nekgsync()
      
      call izero(listp1,LP)
      call izero(listp2,LP)
      
      ! Inquire num of control 
      ! print *, NUMCTRL
      listp1(NID+1)=(NUMCTRL)
      
      ! Global operation, sum everything up 
      call igop(listp1,listp2,"+  ",LP) 

      return
      end subroutine count_total_ctrlpts
c-----------------------------------------------------------------------