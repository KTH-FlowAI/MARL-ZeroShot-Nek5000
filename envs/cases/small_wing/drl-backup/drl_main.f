c==============================================
c DRL subroutines for capusling everying 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================


c----------------------------------------------
        subroutine DRL_main
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "INPUT"
        include "TSTEP"
        include 'PARALLEL'
        include 'mpif.h'
        logical, save :: evolving = .false.
        integer, save :: i_evolv
        integer ndrl,nst,it
        integer drl_step
        character*5 request 
        integer parent_comm,ierr,my_nid
c=============================================
c       Function
c=============================================
        drl_step = int(PARAM(89))
        if (NID.eq.0) print *, "[NEK] DRL STEP",drl_step
        call mpi_comm_rank(MPI_COMM_WORLD,my_nid,ierr)
        call MPI_COMM_GET_PARENT(parent_comm,ierr)

  !----------------------
  ! SET-UP DRL (MUST)
  !----------------------
        if (ISTEP.eq.0) then 
        call drl_init
        else ! We will let Nek run 1 step without control 
  !----------------------
  ! INIFITY  MPI LOOP 
  !----------------------
        mpi_loop: do
          if (NID.eq.0) then 
            print *, "============================"
            print *, "[NEK] INTO MPI LOOP!"
            print *, "============================"
          endif 
          ! FORWARD : EVOLV
          if ((evolving).and.(i_evolv.ne.drl_step)) then
  !---------------------------
            i_evolv = i_evolv+1
  !---------------------------
          ! print *, "ISTEP",ISTEP

            if (NID.eq.0) then
              print *, '[NEK] IEVLOV:',i_evolv,'drl_step:',drl_step
            end if
            
            exit mpi_loop
  !---------------------------
          else
            ! If meets the update limit, stop evolving and refresh the counter
            evolving = .false.
            i_evolv = 1
          endif 
  
          !---------------------------
          ! Broadcast the request FROM STB3
          !---------------------------
          if(NID.eq.0) then 
            call MPI_RECV(request,5,MPI_CHARACTER,0,22,
     &              parent_comm,MPI_STATUS_IGNORE,ierr)
            print *, "============================="
            print *, "[NEK] RECV REQUEST: ",request
            print *, "============================="
            call MPI_BCAST(request,5,MPI_CHARACTER,
     &              0,MPI_COMM_WORLD,ierr)
          
          else
            ! print *, "Waiting BCAST"
            ! call MPI_BARRIER(MPI_COMM_WORLD,ierr)
            call MPI_BCAST(request,5,MPI_CHARACTER,
     &              0,MPI_COMM_WORLD,ierr)
            ! print *,"NID;request",NID,request
          endif ! NID.eq.0

          ! endif ! ((evolving).and.(i_evolv.ne.2))
  !--------------------------------
          !---------------------------
          ! Execute based on Request
          !---------------------------
          select case (request)
            case ('STATE')
                  call drl_state
                  call nekgsync()
            case ('CNTRL')
                  call drl_action
                  call nekgsync()
            case ('TERMN')
                  call stop_simulation
                  call nekgsync()
            case ('EVOLV')
                  evolving = .true.
                  exit mpi_loop
            case ('INTAL')
                  call nekgsync
                  call drl_info_out
          end select
          
          !----------------------------------------------------
        enddo mpi_loop

        call nekgsync()
        if (NID.eq.0) then 
          print *, "============================"
          print *, "[NEK] OUT MPI LOOP!"
          print *, "============================"
        endif
!-----------------------------
        if (evolving) then 
        if (NID.eq.0) then 
          print *, "============================"
          print *, "[NEK] INTO EVLOV!"
          print *, "============================"
        endif
        !! NOTE: In Simson implementation, it requires keeping calling the action subroutines to modify the B.C
        !! HOWEVER, in NEK, as we are using common block to store the actions, this is not required anymore 
        !! See More details in:/scratch/guastoni/PhD/024-DRL_Channel3D/simson/bla/mpi_drl3d.f90
        ! call drl_action

        call drl_reward
        ! call moving_smooth_action(i_evolv,drl_step)

        endif 
!------------------------------
        endif ! If ISTEP.eq.0

        end subroutine DRL_main
c----------------------------------------------


c----------------------------------------------
        subroutine stop_simulation 
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "INPUT"
        include "TSTEP"
        include 'PARALLEL'
        include 'mpif.h'
        ! include "CHKPOINT"
        character*5 termn, request
        integer parent_comm,ierr,my_nid
        parameter(termn="TERMN")
        logical is_exist
c=============================================
c       Function
c=============================================
        call mpi_comm_rank(MPI_COMM_WORLD,my_nid,ierr)
        call MPI_COMM_GET_PARENT(parent_comm,ierr)
  
        if(NID.eq.0) print *, "--------------STOP-----------------"
        if(NID.eq.0) print *, "[TERMN] !!!!! LAST STEP !!!!!"
        ! call checkpoint_save_pert(CHKPTSTEP)
        if(NID.eq.0) print *, "[TERMN] CKPT SAVED!"
        ! if(NID.eq.0) print *, "[TERMN] STOP SIMULATION HERE!"
        if(NID.eq.0) print *, "--------------STOP-----------------"
        call mpi_barrier(MPI_COMM_WORLD,ierr)
        ! print *, NID,"DISCONNET"
        ! THIS IS THE WAY TO Go! 
        if(NID.eq.0) print *, "[TERMN] DISCONNECT!"
        
        call MPI_COMM_DISCONNECT(parent_comm, ierr)
        call exitt0
        ! ! endif 

        ! endif 


        end subroutine stop_simulation
c----------------------------------------------