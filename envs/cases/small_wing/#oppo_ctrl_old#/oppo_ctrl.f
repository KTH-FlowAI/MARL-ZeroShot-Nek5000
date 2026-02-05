!=======================================================================
! Name        : time_series For Opposition Ctrl
! Author      : Adam Peplinski
! Version     : last modification 2015.05.22
! Copyright   : GPL
! Description : This is a set of routines to generate time series for 
!     wing simulations. It is just a slight modification of hpts. It 
!     is written to save some memory and directly couple to arrays 
!     generated in statistics.
!=======================================================================
cc YW: Initialisation for ctrl points and wall points, We will just sample the sensing point value but should build one-to-one correspondence 
cc Date Nov28 2023 
!     Initialise points for statistics point time history
!     I read 2D distribution and generate 3D on
cc YW; We can just initialize the point once and then send to the rest of the processors 
c--------------------------------------------------------------------------------------
cc YW: A subroutine for everything with the opposition control, we do it independen to to the statistics 
            subroutine ctrl_all 
            
            implicit none 

            include 'SIZE'
            include 'TSTEP'
            include 'INPUT'
            include 'OPPO_CTL'
            integer itl1,itl2
            character*2 str, str1
            
c================================================
            if (ISTEP.eq.0) then 
               call ctrl_init
               call send_ctrl_pts
               call recv_ctrl_pts
               icount = 0 ! Refresh the counter 
               ! if(NID.eq.0) then 
c$$$!     testing
c$$               write(str,'(i2.2)') NID
c$$               open(unit=10001,file='gradwall_fnl.txt'//str)
c$$               do itl1=1,totctrl
c$$                  write(10001,*) itl1,
c$$     $           (grdwall(itl2,itl1),itl2 = 1,nfeat),
c$$     $           (crdwall(itl2,itl1),itl2 = 1,3)
c$$               enddo
c$$               close(10001)
c$$$!     testing end
               ! endif 
            else
               ! icount =0
               ! print *, "NID=",NID,"icount:",icount
               ! icount = 0 ! Refresh counter 
               if (mod(ISTEP,ctrl_comp).eq.0) then 
                  ! icount = 0
                  call ctrl_sample
               endif

               ! if (ISTEP.gt.1) then 

            endif 
            return 
            end subroutine ctrl_all 



c-------------------------------------------------------------------------------------
cc YW:  Add an extra function for sampling the contrl and wall point instantaneously 
cc Date: Nov 22 2023 
            subroutine ctrl_sample
            implicit none 
            
            include 'SIZE'
            include 'SOLN'
            include 'TSTEP'
            include 'INPUT'           ! if3d
            include 'OPPO_CTL'
            real ltim_init
            integer npcts              ! position in STAT_RUAVG
            real alpha, beta,dtime    ! time averaging parameters
            integer lnvar             ! count number of variables
            integer i, itl1                 ! loop index
            integer itmp              ! dummy variable
            real rtmp                 ! dummy variable
            character*2 str, str1
            real ctlx,ctly,ctlz
            integer rkid,ix,iy,iz,iel,f
            real ctrlV(LX1,LY1,LZ1,LELT,LDIM) ! The veclocity fields
c================================================

            npcts = LX1*LY1*LZ1*LELT

            if (NIO.eq.0) write(6,*) 'YW: Compute ctrl sampling points'

            if (if3d) then            ! #2D
cc  UNCOMM FOR STATS

                  call copy(ctrlV(1,1,1,1,1),vx,npcts)
                  call copy(ctrlV(1,1,1,1,2),vy,npcts)
                  call copy(ctrlV(1,1,1,1,3),vz,npcts)
                  
                  ! if (ISTEP.gt.1) call wall_pts_sensing(ctrlV)
                  
                  call ctrl_pts_sensing(ctrlV)  !UNCOMM 

                  call comp_ctrl_fluct
                  
                  
                  
            endif
            
            return
            end subroutine ctrl_sample
c-------------------------------------------------------------------------------------

c-------------------------------------------------------------------------------------

cc YW: A new subroutine for collecting the velocity at sensing points s
         !  subroutine ctrl_pts_sensing(ctrlV)
          subroutine ctrl_pts_sensing(ctrlV)
         
          implicit none
          include 'SIZE'
          include 'INPUT'
          include 'TSTEP'
          include 'OPPO_CTL'

          real ctrlV(LX1,LY1,LZ1,LELT,LDIM) ! velocity array
          integer ntot, nxyz        ! array sizes
          integer i                 ! loop index
          integer ifld              ! field number
          character*2 str, str1

c================================================
          nxyz  = NX1*NY1*NZ1
          ntot  = nxyz*NELV
          if(NIO.eq.0) write(6,*) 'YW: Sensing control points'
c TEST
c$$          !  print *, "NID=",NID
c$$         !  print *, "rcode:",rcode(1:5)
c$$         !  print *, "proc:",proc(1:5)
c$$         !  print *, "dist:",dist(1:5)
c$$         !  print *, "rst:",rst(1:5)
c TESTEND
!     evaluate fields start
!     VX
          ifld = 1
          call findpts_eval(inth_hpts,vctl(ifld,1),nfldc,
     &                       rcode,1,
     &                       proc,1,
     &                       elid,1,
     &                       rst,NDIM,totctrl,
     &                       ctrlV(1,1,1,1,1))
         !  print*,"FIND VX"
!     VY
          ifld = ifld +1
          call findpts_eval(inth_hpts,vctl(ifld,1),nfldc,
     &                       rcode,1,
     &                       proc,1,
     &                       elid,1,
     &                       rst,NDIM,totctrl,
     &                       ctrlV(1,1,1,1,2))
         !  print *, "FIND VY"

!     VZ
          if (IF3D) then
             ifld = ifld +1
             call findpts_eval(inth_hpts,vctl(ifld,1),nfldc,
     &                       rcode,1,
     &                       proc,1,
     &                       elid,1,
     &                       rst,NDIM,totctrl,
     &                       ctrlV(1,1,1,1,3))
            !  print *, "FIND VZ"
          endif

          if (NID.eq.0) write(6,*) "YW: Control sampling ended"
! TEST 
c$$          if (NID.eq.0) then 
c$$          write(str,'(i2.2)') NID
c$$          write(str1,'(i2.2)') ISTEP
c$$          open(1000001, file='vctl.txt'//str//str1)
c$$         
c$$          do i=1,totctrl
c$$!             write(1000001,*) 
c$$!      $          crdctl(1,i), crdctl(2,i),crdctl(3,i)
c$$            write(1000001,*) 
c$$     $          vctl(1,i), vctl(2,i),vctl(3,i)
c$$          enddo 
c$$          close(1000001)  
c$$          endif
! TEST END
          return
          end
c-------------------------------------------------------------------------------------

cc YW: A new subroutine JUST for check the result 
         subroutine wall_pts_sensing(ctrlV)
                  
         implicit none
         include 'SIZE'
         include 'INPUT'
         include 'TSTEP'
         include 'OPPO_CTL'
         real ctrlV(LX1,LY1,LZ1,LELT,LDIM) ! velocity array
         integer ntot, nxyz        ! array sizes
         integer i                 ! loop index
         integer ifld              ! field number
         character*2 str, str1

         real wctl(nfldc,totctrl)
         integer inth_wall
         integer wrcode(totctrl)
         integer welid(totctrl)
         integer wproc(totctrl)
         real wdist(totctrl)
         real wrst(totctrl*LDIM)

         common /stat_wallsi/ inth_wall
         common /stat_walliv/ wrcode, welid, wproc
         common /stat_wallrv/ wdist, wrst
         
       
c================================================
         nxyz  = NX1*NY1*NZ1
         ntot  = nxyz*NELV

            if (NID.eq.0) write(6,*) "YW: Wall sampling ended"
            ! if (NID.eq.0) then 
            write(str,'(i2.2)') NID
            write(str1,'(i2.2)') ISTEP
            open(1000001, file='vWall.txt'//str//str1)
            do i=1,totctrl
            write(1000001,*) 
     $           i, "Sensing VEL:",vctl(1,i), vctl(2,i),vctl(3,i)
            ! if (vwall)
            write(1000001,*) 
     $           i, "Wall VEL:",vwall(1,i), vwall(2,i),vwall(3,i)
            
            write(1000001,*) 
     $           i, "Wall COORD REAL:",
     $           crdwall(1,i),crdwall(2,i),crdwall(3,i)
                   
            write(1000001,*) 
     $           i, "Wall COORD REAL:",
     $           rwall(1,i), rwall(2,i),rwall(3,i)
            enddo 
                         
            close(1000001)  
            ! endif

            return
            end

c-----------------------------------------------------------------------

c-----------------------------------------------------------------------
cc YW: Compute the spatial mean in Z-dir 
cc We only compute it on the NID=0  and then we communicate with the rest 
cc Date: 26 Nov 2023 
            subroutine comp_ctrl_fluct
            implicit none 
            
            include 'SIZE'
            include 'INPUT'
            include 'OPPO_CTL'
            include 'TSTEP'

            integer nptctl, nzl
            integer  il, jl, kl ! Iteration 
            real    vm(nfldc, totctrl) ! mean value of each variable along z direction 
            real    flt, smm

            character*2 str, str1
            integer i 
c================================================
            nzl = uniqz
            nptctl = int(totctrl/nzl)
            
            do kl = 1,nfldc ! For each variable
               do jl = 1,nptctl !  Sum For each Z-plane 
                  smm = 0
                  do il = 1, nzl 
                     smm = smm +  vctl(kl,(jl-1)*nzl+il)           
                  enddo
                  vm(kl,jl) = smm/nzl  ! Do avgerage
               enddo
            enddo  

cc YW: Decomposition 
            do kl = 1,nfldc 
               do jl = 1,nptctl
                  do il = 1,nzl
                     flt = vctl(kl,(jl-1)*nzl + il) - vm(kl,jl)
                     vctl(kl,(jl-1)*nzl+il) = flt
                  enddo
               enddo
            enddo

            if (NID.eq.0) write(6,*) "YW: Get Spatial Fluctuation"
cc TEST! 
c$$            if (NID.eq.0) then 
c$$            write(str,'(i2.2)') NID
c$$            write(str1,'(i2.2)') ISTEP
c$$            open(1000001, file='vFluct.txt'//str//str1)
c$$     
c$$            do i=1,totctrl
c$$              write(1000001,*) 
c$$     $          vctl(1,i), vctl(2,i),vctl(3,i)
c$$            enddo 
c$$            close(1000001)  
c$$            endif
cc TEST END!    
            
            return 
            end


c-----------------------------------------------------------------------
cc YW: Sort out the GRID wall point and the corresponding RANK at different processor 
cc Date Nov 29 2023 

c-----------------------------------------------------------------------
cc YW: After we find the corresponding grid wall point, We need communicate 
cc Date Nov 29 2023 
            subroutine send_ctrl_pts
           
            implicit none 
            
            include 'SIZE'
            ! include 'INPUT'
            include 'OPPO_CTL'         
            include 'mpif.h'
            integer ierr
            integer k,il,jl        ! Iteration
            integer len ! Flag for counting 
cc YW: Test for MPI_ISEND & MPI_IRecv 
            integer request_send, request_recv, status(mpi_status_size)
            ! integer send_buff(nfeat,totctrl), recv_buff(nfeat,totctrl) ! The pair to send and recevie
            integer recv_buff(nfeat,totctrl) ! The pair to send and recevie
c===========================================================
            ! Initialising parameters
            ! print *, "Start Sending ELEM INFO"
            len  = nfeat*totctrl ! We send and recv the full array
            ierr = 0  
            if (NID.ne.0) then ! Except #0, the rest processors sent their array to it
            ! call copy(send_buff,grdwall,len)

            call MPI_SEND(grdwall, len, MPI_INTEGER, 
     $                  0, NID, 
     $                  MPI_COMM_WORLD, ierr)
            ! print *, "Finish SEND ELEM INFO"
            else              ! if NID.eq.0
               do k = 1,LP-1  ! We recv from all the rest processor 

                  call MPI_RECV(recv_buff, len, MPI_INTEGER, 
     $                  k, k,
     $                  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                  do jl =1,nfeat ! And we sort out those usable elements from array 
                     do il=1,totctrl
                        if (recv_buff(jl,il).ne.infty) then 
                             grdwall(jl,il) = recv_buff(jl,il)
                        endif 
                     enddo
                  enddo  ! jl = 1, nfldm
              
               enddo ! k=1,LP
               ! print *,"FINISH recv"
            endif 
            return 
            end 
c-----------------------------------------------------------------------

            subroutine recv_ctrl_pts 

            implicit none 

            include 'SIZE'
            ! include 'INPUT'
            include 'OPPO_CTL'
            include 'mpif.h'
            
            integer ierr, mtype
            integer k        ! Iteration
            integer len
            integer iflag    ! Flag for counting 
cc YW: Test for MPI_ISEND & MPI_IRecv 
            integer request_send, request_recv, status(mpi_status_size)
            ! integer send_buff(nfeat,totctrl), recv_buff(nfeat,totctrl) ! The pair to send and recev
            ! integer recv_buff(nfeat,totctrl) ! The pair to send and recev
c===========================================
! Initialising parameters
            len  = nfeat*totctrl
            ierr = 0 
            iflag = 0
            if (NID.eq.0) then ! Let #0 send the array to all the rest processors 
            ! call copy(send_buff,grdwall,len)
                  do k=1,LP-1
                        iflag = k*3
                        call MPI_SEND(grdwall, len, MPI_INTEGER, 
     $                  k, iflag, 
     $                  MPI_COMM_WORLD, ierr)
                  enddo

                  else ! IF NID.ne.0
                     iflag = NID*3
                     call MPI_RECV(grdwall, len, MPI_INTEGER, 
     $                  0, iflag,
     $                  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)

            endif 
            
            ! print *,"FINISH SHARE PTS"
                  
            return 
            end   

c-----------------------------------------------------------------------
cc YW: 
cc Then we can leverage the indices information to check the B.C
            subroutine  sensing_Vel(ix,iy,iz,iel,rkid,f,
   !   $                                   snx,sny,snz,
     $                                   ux,uy,uz,iflag,iloc)

            implicit none 
            
            include 'SIZE'
            include 'OPPO_CTL' 
            include 'TSTEP'

            integer iel, ieg, rkid,f
            integer ix,iy,iz             ! Input info
            integer wx,wy,wz,wel,wid,wf    ! Written info 
            real Vx,Vy,Vz
            real ux, uy, uz
            real snx,sny,snz
            integer il ! Iteration 
            real wax,way,waz ! Test
            real ampl ! Amplitude for control
            parameter(ampl = 1.0)
            logical iflag
            integer iloc ! At which loc we find the data?
c================================================            
            ! print *,rkid,ix,iy,iz,iel
            do il = 1, totctrl
                  wid = grdwall(1,il)
                  wx  = grdwall(2,il)
                  wy  = grdwall(3,il)
                  wz  = grdwall(4,il)
                  wel = grdwall(5,il)
                  wf  = grdwall(6,il)
                  if(wid.eq.rkid .and.
     $               wx.eq.ix    .and.
     $               wy.eq.iy    .and.
     $               wz.eq.iz    .and.
     $               wel.eq.iel  .and.
     $               wf.eq.f) then  
                              Vx = vctl(1,il)
                              Vy = vctl(2,il)
                              Vz = vctl(3,il)

cc: We follow the control function: 
cc  V(x,yw,z) = -ALPHA*v(x,ys,z) where V is velocity and v' is spatial fluctucation

                              ux = ampl*Vx
                              uy = ampl*Vy
                              uz = ampl*Vz
                              iflag = .TRUE.
                              iloc = il
                              goto 909
! Test                        
c$                              print *,NID,"ISCTRL!"
c$                              wax = crdwall(1,il)
c$                              way = crdwall(2,il)
c$                              waz = crdwall(3,il)
c$                              print *,"wax",wax,'way',way,'waz',waz
! Testend
                              ! goto 909
                  endif
            enddo

909         return 
            end



c-----------------------------------------------------------------------
! ! MA: substituted by intp_setup in Nek5000/core/intp.f
cc MA: copied from nek1093_dong/trunk/nek/postpro.f
c-----------------------------------------------------------------------
! !       subroutine intpts_setup(tolin,ih)
! ! c
! ! c setup routine for interpolation tool
! ! c tolin ... stop point seach interation if 1-norm of the step in (r,s,t) 
! ! c           is smaller than tolin 
! ! c
! !       include 'SIZE'
! !       include 'GEOM'
! ! 
! !       common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal
! ! 
! !       tol = tolin
! !       if (tolin.lt.0) tol = 1e-13 ! default tolerance 
! ! 
! !       n       = lx1*ly1*lz1*lelt 
! !       npt_max = 256
! !       nxf     = 2*nx1 ! fine mesh for bb-test
! !       nyf     = 2*ny1
! !       nzf     = 2*nz1
! !       bb_t    = 0.1 ! relative size to expand bounding boxes by
! ! c
! !       if(nidd.eq.0) write(6,*) 'initializing intpts(), tol=', tol
! !       call findpts_setup(ih,nekcomm,npp,ndim,
! !      &                     xm1,ym1,zm1,nx1,ny1,nz1,
! !      &                     nelt,nxf,nyf,nzf,bb_t,n,n,
! !      &                     npt_max,tol)
! ! c       
! !       return
! !       end
! ! c-----------------------------------------------------------------------



! c----------------------------------------------------------
! cc YW: The subroutine for sampling 
!           subroutine ctrl_pts_out(fieldouts,nflds)
!           implicit none 
          
!           include 'SIZE'
!           include 'TSTEP'
!           include 'INPUT'
!           include 'OPPO_CTL'
!           integer lfldm             ! max number of fields
!           parameter(lfldm=2*LDIM+1)
!           !     arguments
!           real fieldouts(lfldm,LHIS)
!           integer nflds             ! number of fields
          
        
!           integer itl1, itl2, itl3
!           character*2 str
! cc YW for sorting out points 
!          !  real cxp, cyp, czp
!          !  real xtp,ytp, ztp
!           integer xgp
!           integer cgp
!           logical isfind
!           integer fcount

    
! ! #########################################
! cc YW: Modification Nov 21 2023: We sort out the velocity 
!          ! We first fill arrays into inf 
!           do itl2 = 1, ttp
!             do itl3 = 1,lfldm
!                vctl(itl3,itl2) = infty
!             enddo 
!           enddo 
!           fcount = 1
!           do itl1 = 1, npts ! Loop along all the grid points
!             xgp = ipts(itl1)
!             do itl2 = 1, ttp ! Loop for all control points
!                ! cxp = crdctl(1,itl2)
!                ! cyp = crdctl(2,itl2)
!                ! czp = crdctl(3,itl2)
!                cgp = iptctl(itl2)
!                 if (xgp.eq.cgp) then
!                    do itl3 = 1,lfldm
!                       vctl(itl3,itl2) = fieldouts(itl3,itl1)
!                    enddo 
!                    fcount = 1
!                 endif
!             enddo

!           enddo
!          !  write(*,*) "YW: At NID, fcount", NID, fcount
!           return
!           end


! c-----------------------------------------------------------------------
! cc YW: An attempt to commnuicate the data for sampled points 
! cc Date Nov 23 2023 
! cc The idea is to send all 
!             subroutine comm_ctrl_pts
           
!             implicit none 
            
!             include 'SIZE'
!             include 'INPUT'
!             include 'OPPO_CTL'         
!             include 'mpif.h'
!             integer ierr
!             integer k,il,jl        ! Iteration
!             integer len ! Flag for counting 
! cc YW: Test for MPI_ISEND & MPI_IRecv 
!             integer request_send, request_recv, status(mpi_status_size)
!             real send_buff(nfldm,ttp), recv_buff(nfldm,ttp) ! The pair to send and recevie

!             ! write(6,*) "YW: COMM TEST"
! ! Initialising parameters
!             len  = nfldm*ttp ! We send and recv the full array
!             ierr = 0 
! cc ################################################
! cc YW: Make a test on how to send a value 42 to NID#0
! cc Date Nov 23 2023 
! cc YW: Fixed it 
! cc Update Date Nov 24 2023  
!             if (NID.ne.0) then ! Except #0, the rest processors sent their array to it
!             call copy(send_buff,vctl,len)

!             call MPI_SEND(send_buff, len, MPI_DOUBLE, 
!      $                  0, NID, 
!      $                  MPI_COMM_WORLD, ierr)

!             else              ! if NID.eq.0
!                do k = 1,LP-1  ! We recv from all the rest processor 

!                   call MPI_RECV(recv_buff, len, MPI_DOUBLE, 
!      $                  k, k,
!      $                  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
!                   do jl =1,nfldm ! And we sort out those usable elements from array 
!                      do il=1,ttp
!                         if (recv_buff(jl,il).ne.infty) then 
!                              call copy(vctl(jl,il), recv_buff(jl,il),1)
!                         endif 
!                      enddo
!                   enddo  ! jl = 1, nfldm
              
!                enddo ! k=1,LP
!             endif 

! ! Test
!             ! print *, "YW ,NID",NID, "vbuff"
!             ! do k=1,ttp
!             !    print *, vctl(1,k)
!             ! enddo
! ! end test
            
!             return
!             end 

! c-----------------------------------------------------------------------
! cc YW
! cc After everything is on the #0 processor, Now we need to send it to all processor 
! cc Date  24 Nov 2023 
!             subroutine share_ctrl_pts 

!             implicit none 
!             include 'SIZE'
!             include 'INPUT'
!             include 'PTSTAT'
!             include 'OPPO_CTL'
!             include 'mpif.h'
!             integer ierr, mtype
!             integer k        ! Iteration
!             integer len
!             integer iflag    ! Flag for counting 
! cc YW: Test for MPI_ISEND & MPI_IRecv 
!             integer request_send, request_recv, status(mpi_status_size)
!             real send_buff(nfldm,ttp), recv_buff(nfldm,ttp) ! The pair to send and recevie

!             ! write(6,*) "YW: COMM Share"
! ! Initialising parameters
!             len  = nfldm*ttp
!             ierr = 0 
!             iflag = 0
!             if (NID.eq.0) then ! Let #0 send the array to all the rest processors 
!             call copy(send_buff,vctl,len)
!             do k=1,LP-1
!                   iflag = k*3
!                   call MPI_SEND(send_buff, len, MPI_DOUBLE, 
!      $                  k, iflag, 
!      $                  MPI_COMM_WORLD, ierr)
!             enddo
            
!             else ! IF NID.ne.0
!                iflag = NID*3
!                call MPI_RECV(recv_buff, len, MPI_DOUBLE, 
!      $                  0, iflag,
!      $                  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                  
!                call copy(vctl,recv_buff,len)

         
!             endif 
! ! test             
!             ! print *, "YW ,NID",NID, "vbuff"

!             ! do k=1,5
!             !    print *, vctl(1,k)
!             ! enddo
! ! test end       
!             return 
!             end 