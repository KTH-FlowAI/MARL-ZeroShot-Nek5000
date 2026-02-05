c==============================================
c DRL subroutines for ACTION 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================



c------------------------------------------------------------------
        subroutine drl_action
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include 'PARALLEL'
c=============================================
c       Function
c=============================================
        
      !   if (ISTEP.eq.0) then
cc STEP 1: Do nothing 
            if (NID.eq.0) then 
                  print *,"-------------------------"
                  print *,"[ACTION] STANDING BY"
                  print *,"-------------------------"
            endif 
cc STEP 3: Computing Fluctuation and Actuating on the wall 
c-----------------------------------------------
      !   else ! When the simulation is running: 
            ! call wall_aft_vel               ! TEST the actutation on the wall 
            call recv_Actions() 
            call nekgsync()
            ! YW Comment here,since we use the unique action, the Weighted average will explode
            ! call znmf_avg()
      !   endif ! if (ISTEP.eq.0)

        return 
        end subroutine drl_action
c------------------------------------------------------------------



c------------------------------------------------------------------
      subroutine recv_Actions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'PARALLEL'
      include 'DRL'         
      include 'mpif.h'
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      !-----------------------
      integer totLine,ntot
      ! integer gllel
      integer glbid,fceid,nidid,nididnek,lclid
      integer ix,iy,iz
      real    jetval
      character*4 str, str1
      !-----------------------
      
      !--------------------------
      ! MPI 
      integer parent_comm, ierr
      integer idx_buffer(totctrl)
      integer fce_buffer(totctrl)
      real    act_buffer(totctrl)
      integer sendLen 
      integer request_send, request_recv, status(mpi_status_size)
      !--------------------------
      logical ifexist
      character*13 fNAME

c=============================================
c       Function
c=============================================
      call MPI_COMM_GET_PARENT(parent_comm,ierr)

      if (NUMCTRL.ne.0) then 

      call MPI_RECV(act_buffer,TOTCTRL,MPI_DOUBLE,
     &            0,NID+90000,parent_comm,
     &            MPI_STATUS_IGNORE,ierr)

      ! print *,"[ACTION] UPDATE NID=",NID
      
      else
      ! print *,"[ACTION] UPDATE NID=",NID
      endif
      ! call nekgsync()
! Update 
!----------------------------------------
      ntot=LX1*LY1*LZ1*LELT
      ! INIT THE buffer with dumi value 
      ! call cfill(ACTIONS(1,1,1,1),dumi,ntot)
      ! Update 
      do il=1,NUMCTRL 
            glbid=grdwall(1,il)
            lclid=gllel(glbid)
            fceid=grdwall(2,il)
            ix=grdwall(3,il)
            iy=grdwall(4,il)
            iz=grdwall(5,il)
            ! ACTIONS(ix,iy,iz,fceid,lclid)=act_buffer(il)
            ! YW Modified Nov 4th 2024, NO NEED of FACE!
            ACTIONS(ix,iy,iz,lclid)=act_buffer(il)
      enddo 
      
      ! totLine=LX1*LY1*LZ1*6*LELT
      ! if(ISTEP.eq.1) call copy(old_ctrl_val,ctrl_val,totLine)
c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      if (ISTEP.le.20) then 
      if (numctrl.gt.0) then 
      write(str,"(i4.4)") NID
      write(str1,"(i4.4)") ISTEP
      open(10001,file="RECV-ACTION.txt"//str//str1)
      write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, "
      do il = 1,numctrl
      lclid=grdwall(1,il)
      lclid=gllel(lclid)
      ix=grdwall(3,il)
      iy=grdwall(4,il)
      iz=grdwall(5,il)
      write(10001,*) lclid,
     $      (crdwall(jl,il), jl=1,NDIM),
     $       ACTIONS(ix,iy,iz,lclid) 
      enddo
      close(10001)
      endif
#endif 
      endif
!----------------------------------------

      if (NID.eq.0) then
            print *, "-------------------------"
            print *, "[ACTION] UPDATED"
            print *, "-------------------------"
      endif
      
      end subroutine recv_Actions 
c------------------------------------------------------------------

c------------------------------------------------------------------
      subroutine ACTUATE_JET(vf,isfind,ix,iy,iz,iside,eg)
! Used for userbc: Impose velocity 
! c=============================================
! c       Define variable
! c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,iside,eg,iel
      integer ierr,IO_STATUS,rface
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      real vf
      logical isfind
! c=============================================
! c       Function
! c=============================================
      iel = gllel(eg)
      vf =0.0
      isfind =.FALSE.
      
      if (NUMCTRL.ne.0) then 
            ! We know there is only one face! 
            rface=grdwall(2,1)
            vf=ACTIONS(ix,iy,iz,iel)
            if (vf.ne.dumi .and. iside.eq.rface) then 
                  isfind=.TRUE.
            endif
      endif 

      end subroutine ACTUATE_JET
c--------------------------------------------------


c--------------------------------------------------

      subroutine znmf_avg() 
c Subroutine for average the Actions for ZNMF conditions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real velV(LX1,LY1,LZ1,LELT)
      real actV(LX1,LY1,LZ1,LELT)
      real avgVZ(LX1,LY1,xnel,ynel)
      real avgVX(LX1,LY1,ynel,znel)
      real vf,va,vo
      
      ! Handlers for gop average
      integer igs_x,igs_z
      save igs_x,igs_z
      ! For test 
      integer ilx, ily
      character*4 str, str1
c=============================================
c       Functions
c=============================================
      ntot=LX1*LY1*LZ1*LELT

      ! STEP 1: Spatial Average based on X- Z-dir 
      ! NOTE: Again this should be modified if we move to NEK-V17 
      ! if (ISTEP.eq.1) then 
      ! call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
      ! call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
      ! if (NID.eq.0) print *, "[ACTION] AVG HANDLE INIT!"
      ! endif 

      call copy(velV(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      call copy(actV(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      if (rwd_zavg) then 
            call z_averaging(velV,avgVZ)
            call z_avg_reshape(velV,avgVZ)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] Z-AVG!"
#endif
            endif ! if (rwd_zavg)

            ! Do streamwise average if it allowed/defined
            if (rwd_xavg) then 
            call z_averaging(velV,avgVX)
            call z_avg_reshape(velV,avgVX)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] X-AVG!"
#endif
            endif 

      ! Step 2: Subtract the mean of the action 
            do ilx = 1,numctrl
            iel=grdwall(1,ilx)
            iel=gllel(iel)
            ix=grdwall(3,ilx)
            iy=grdwall(4,ilx)
            iz=grdwall(5,ilx)
            va=velV(ix,iy,iz,iel)
            vo=actV(ix,iy,iz,iel)  
            ACTIONS(ix,iy,iz,iel)=vo-va
            enddo
      
c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      if (NID.eq.0) print *, "[ACTION] ZNMF AVERAGED"
      if (ISTEP.eq.1) then 
            if (numctrl.gt.0) then 
            write(str,"(i4.4)") NID
            write(str1,"(i4.4)") ISTEP
            open(10001,file="ZNMF-ACTION.txt"//str//str1)
            write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, ","MEAN, ","BEFORE" 
            do ilx = 1,numctrl
            iel=grdwall(1,ilx)
            iel=gllel(iel)
            ix=grdwall(3,ilx)
            iy=grdwall(4,ilx)
            iz=grdwall(5,ilx)
            write(10001,*) iel,
     $      (crdwall(ily,ilx), ily=1,NDIM),
     $       ACTIONS(ix,iy,iz,iel), 
     $       velV(ix,iy,iz,iel),
     $       actV(ix,iy,iz,iel)
            enddo
            close(10001)
            endif
      endif 
#endif 

      end subroutine znmf_avg
c--------------------------------------------------



c--------------------------------------------------
      subroutine moving_smooth_action(i_evolv,ndrl)
c Subourtine for Moving average for smoothing the action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real ctrl_val(LX1,LY1,LZ1,LELT)
      real old_ctrl_val(LX1,LY1,LZ1,LELT)
      real vf,va,vo,reduce_r
      common /ctrl_cache/ ctrl_val,old_ctrl_val
      ! save ctrl_val,old_ctrl_val
c=============================================
c       Function
c=============================================
      ntot = LX1*LY1*LZ1*LELT

      ! if (NID.eq.0) print *, "[ACTION] SMOOTHING",i_evolv,ndrl

      ! Copy for the worker array
      call copy(ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      
      ! Initialize the reference, 
      ! which means the first NDRL we do not decay the action
      if (ISTEP.eq.1) then 
      call copy(old_ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      endif 
      ! endif


      do il=1,NUMCTRL
      iel=grdwall(1,il)
      iel=gllel(iel)
      ix=grdwall(3,il)
      iy=grdwall(4,il)
      iz=grdwall(5,il)
      vo = old_ctrl_val(ix,iy,iz,iel)
      vf = ctrl_val(ix,iy,iz,iel)

#ifdef YWDEBUG
      if (NUMCTRL.ne.0 .and. il.le.5) then 
      print *,'VO AND VF',vo,vf
      endif 
#endif

      ! ------------ Update Action using moving average ------------
      ! if (vf.ne.vo) then
      reduce_r = real(i_evolv)/real(ndrl)
      vf=vo+(vf-vo)*reduce_r
      ctrl_val(ix,iy,iz,iel)=vf
      ! endif 
      !------------------------------------------------------

#ifdef YWDEBUG
      if (NUMCTRL.ne.0 .and. il.le.5) then 
      print *,'Smooth VO from VF',vo,vf,reduce_r
      endif 
#endif 
      
      enddo ! do il=1,NUMCTRL
      

      ! At the end of evolv also update reference 
      if (i_evolv.eq.ndrl) then 
            call copy(old_ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      endif 

      ! ! Update actions 
      call copy(ACTIONS(1,1,1,1),ctrl_val(1,1,1,1),ntot)
      

      end subroutine moving_smooth_action
c--------------------------------------------------


c--------------------------------------------------
      subroutine exp_smooth_action(i_evolv,ndrl)
c Exponentially Smoothing the Action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel
      integer ntot,il
      integer i_evolv,ndrl
      real ctrl_val(LX1,LY1,LZ1,LELT)
      real vf,va,vo,reduce_r
      ! save ctrl_val,old_ctrl_val
c=============================================
c       Function
c=============================================
      ntot = LX1*LY1*LZ1*LELT

      ! if (NID.eq.0) print *, "[ACTION] SMOOTHING",i_evolv,ndrl

      ! Copy for the worker array
      call copy(ctrl_val(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      
      do il=1,NUMCTRL
            iel=grdwall(1,il)
            iel=gllel(iel)
            ix=grdwall(3,il)
            iy=grdwall(4,il)
            iz=grdwall(5,il)
            vf = ctrl_val(ix,iy,iz,iel)
            vo = vf
            ! ------- UPDATE ACTION -------------
            reduce_r = real(i_evolv)/real(ndrl)
            call smooth_step(vf,reduce_r)
            ctrl_val(ix,iy,iz,iel) = vf
            !------------------------------------
#ifdef YWDEBUG
            if (NUMCTRL.ne.0 .and. il.le.5) then 
            print *,NID,'Smooth VO from VF',vo,vf,reduce_r
            endif 
#endif 
      enddo 
      ! ! Update actions 
      call copy(ACTIONS(1,1,1,1),ctrl_val(1,1,1,1),ntot)
      

      end subroutine exp_smooth_action
c--------------------------------------------------





c--------------------------------------------------
      subroutine smooth_step(step,x)
c
c     Smooth step function:
c     x<=0 : step(x) = 0
c     x>=1 : step(x) = 1
c     Non-continuous derivatives at x=0.02 and x=0.98
c
      implicit none

      real x,step

      if (x.le.0.02) then
            step = 0.0
      else
            if (x.le.0.98) then
            step = 1./( 1. + exp(1./(x - 1.) + 1./x) )
            else
            step = 1.
            end if
      end if

      return  
      end subroutine smooth_step
c--------------------------------------------------


! c------------------------------------------------------------------
!       subroutine assign_Actions(totLine,idx_buffer,
!      $                           fce_buffer,act_buffer)
! c=============================================
! c       Define variable
! c=============================================
!       implicit none 
!       include 'SIZE'
!       include 'TOTAL'
!       include 'DRL'         
!       ! include 'mpif.h'
!       integer ierr,IO_STATUS
!       integer k,il,jl        ! Iteration
!       integer len,recctrl ! Flag for counting 
      
!       !-----------------------
!       integer totLine
!       integer glbid,fceid,nidid,nididnek,lclid
!       real    jetval
!       !-----------------------
      
!       !--------------------------
!       ! MPI 
!       integer idx_buffer(LP,totctrl)
!       integer fce_buffer(LP,totctrl)
!       real    act_buffer(LP,totctrl)
!       real    dummy(6,LELT)
!       ! integer sendLen 
!       ! integer request_send, request_recv, status(mpi_status_size)
      
!       ! integer send_buff(2,totctrl) ! The pair to send and recevie
!       ! integer recv_buff(2,totctrl) ! The pair to send and recevie
!       ! real send_act(totctrl)
!       ! real recv_act(totctrl)
!       ! !--------------------------

!       logical ifexist
!       character*13 fNAME
!       character*6 HEAD1, HEAD2, HEAD3, HEAD4
!       parameter(fNAME="JET_INPUT.dat")
! c=============================================
! c       Function
! c=============================================
!       len = LELT*6
!       do il=1,totLine
!       ! Read global id and convert into local
!       glbid=idx_buffer(NID,il)
!       lclid=gllel(glbid)
!       ! Read face id 
!       fceid=fce_buffer(NID,il)
!       ! Read action
!       jetval=act_buffer(NID,il)
!       ! dummy(fceid,lclid)=jetval
!       print *, "[DRL] NID",NID,"RECV:",glbid,fceid
!       enddo 
!       ! call copy(ACTIONS,Act_dummy,len)
      
! !       call nekgsync()
! !       if (NID.eq.0) then 
! !             !-------------------------------------------------------    
! !             ! NOW SEND THEM 
! !             do il=1,LP-1
! !             ! NEK Global Synethic
! !             call icopy(send_buff(1,1),idx_buffer(il,1),totctrl)
! !             call icopy(send_buff(2,1),fce_buffer(il,1),totctrl)
! !             call copy(send_act(1),act_buffer(il,1),totctrl)
! !             call MPI_SEND(send_buff, sendLen, MPI_INTEGER, 
! !      $                  il, il+20000, 
! !      $                  MPI_COMM_WORLD, ierr)
            
! !             call MPI_SEND(send_act, TOTCTRL, MPI_FLOAT, 
! !      $                  il, il+30000, 
! !      $                  MPI_COMM_WORLD, ierr)
            
! !             enddo ! do il=1,LP-1 
! !       !-------------------------------------------------------    

! !       else ! if NID.ne.0
            
! !             print *, "[DRL] AT NID",NID,"WAITING FOR RECV"
! !             call MPI_RECV(recv_buff, sendLen, MPI_INTEGER, 
! !      $                  0, NID+20000, 
! !      $                  MPI_COMM_WORLD, ierr)

! !             call MPI_RECV(recv_act, TOTCTRL, MPI_FLOAT, 
! !      $                  0, NID+30000, 
! !      $                  MPI_COMM_WORLD, ierr)
! !             print *, "[DRL] AT NID",NID,"ACTION RECV"
            
! !       ! Loop to Assign the ACTION
! !       do il=1,TOTCTRL
! !       ! Read global id and convert into local
! !       glbid=recv_buff(1,il)
! !       lclid=gllel(glbid)
! !       ! Read face id 
! !       fceid=recv_buff(2,il)
! !       ! Read action
! !       jetval=recv_act(il)
! !       Vnfluct(fceid,lclid)=jetval
! !       print *, "[DRL] NID",NID,"RECV:",glbid,fceid,Vnfluct(fceid,lclid)
! !       enddo 
! !       endif ! if (nid.eq.0)

!       print*, "[DRL] NID=",NID," GET ACTION INTO ENV"
!       end subroutine assign_Actions
c------------------------------------------------------------------
