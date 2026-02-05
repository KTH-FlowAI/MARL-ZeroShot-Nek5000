
c------------------------------------------------------------------
        subroutine drl_init
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
        
        
                if (NID.eq.0) then
                        print *,"=============================="
                        print *, "[DRL] STATE Initialisation"
                        print *,"=============================="
                endif 
cc STEP 1: Preprocessing for Finding the Wall Points by coordinates and B.C type
c---------------------------------------------
                call register_wall_pts          ! FIND WALL Points by B.C 
c---------------------------------------------

c---------------------------------------------
c      Bubble Sorting for coordinates
c--------------------------------------------- 
c: YW It is redundant in practise 
c: BUT gives a clean results to check the implementations 
                call sort_wall_pts
c----------------------------------------------

cc STEP 2: Locating the sensing plane (y+=15)
c----------------------------------------------
                call interp_sensing_plane       ! Use u_tau for calculating y+, find the sensing plane
                call register_ctrl_pts          ! Interpolating the sensing plane on the mesh grid
                ! call nekgsync
                ! call interp_wall_pts          ! Interpolating the sensing plane on the mesh grid
c----------------------------------------------
                if (NID.eq.0) then
                        print *,"---------------------------"
                        print *, "[DRL] NODE INFO READY"
                        print *,"---------------------------"
                endif 
        
                if (NID.eq.0) then
                        print *,"=============================="
                        print *, "[DRL] FINISH INIT"
                        print *,"=============================="
                endif 
        return 
        end subroutine drl_init
c------------------------------------------------------------------


c------------------------------------------------------------------
        subroutine register_wall_pts
cc YW: A subroutine for finding the wall points 
cc We only care about the wall points with wall boundary conditions! 
c=============================================
c       Define variable
c=============================================

        implicit none 
        include "SIZE"
        include "TOTAL"
        include "NEKUSE"
cc YW:
        include "DRL"

        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        
        integer ie,iface,ix,iy,iz,lgi ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer idx
        real xf,yf,zf
        real xr,yr,zr
        character*3 bcb

        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================
        nfaces = 2*NDIM
        NEL    = NELFLD(IFIELD)
        ! From MA: Face midpoints on each direction
        im = (1+nx1)/2
        jm = (1+ny1)/2
        km = (1+nz1)/2
        ! print *, "im,jm,km=",im,jm,kmz
        ! From MA: Grid-point number within element corresponding to each face
        fmid(4) =   1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x = -1
        fmid(2) = nx1 + nx1*( jm-1) + nx1*ny1*( km-1) !  x =  1
        fmid(1) =  im + nx1*(  1-1) + nx1*ny1*( km-1) !  y = -1
        fmid(3) =  im + nx1*(ny1-1) + nx1*ny1*( km-1) !  y =  1
        fmid(5) =  im + nx1*( jm-1) + nx1*ny1*(  1-1) !  z = -1
        fmid(6) =  im + nx1*( jm-1) + nx1*ny1*(nz1-1) !  z =  1
c########################################
c     DRL initialisation
c########################################
cc YW: The idea is to find the wall points by using the range and the wall boundary condition
cc YW: FOR DRL, we only find the face mid points
        numctrl = 0 
        do ie=1,NEL
        do iface=1,nfaces
                xr=xm1(fmid(iface),1,1,ie) ! X coord for face mid point
                yr=ym1(fmid(iface),1,1,ie) ! Y coord for face mid point
                zr=zm1(fmid(iface),1,1,ie) ! Y coord for face mid point
                bcb=CBC(iface,ie,1)
                
                ! call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                ! print *, "FACEMID",fmid(iface)
                ! print *,"FACEIND",KX1,KX2,KY1,KY2,KZ1,KZ2
                call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
                do iz=KZ1,KZ2
                do iy=KY1,KY2
                do ix=KX1,KX2
                xf=xm1(ix,iy,iz,ie)
                yf=ym1(ix,iy,iz,ie)
                zf=zm1(ix,iy,iz,ie)        
                if (bcb.eq.'W  ') then
                numctrl = numctrl +1 ! if meet condition
                lgi=lglel(ie)
                crdwall(1,numctrl)=xf
                crdwall(2,numctrl)=yf
                crdwall(3,numctrl)=zf
cc YW: I write the global grid information here for find index in USERBC
                grdwall(1,numctrl)=lgi         ! Global indicies
                grdwall(2,numctrl)=iface       ! Number of face
                grdwall(3,numctrl)=ix         ! Number of x ploy
                grdwall(4,numctrl)=iy ! Number of y poly 
                grdwall(5,numctrl)=iz ! Number of z poly 
cc YW: Write the RANK info, for further interpolation
                proc(numctrl)=NID
cc YW: Change the BC to Dirichlet LATER                 
                CBC(iface,ie,1) = "v  "
                endif ! If bcb.eq."W  "
                enddo ! do ix = KX1, KX2 
                enddo ! do iy = KY1, KY2 
                enddo ! do iz = KZ1, KZ2
        enddo ! iface = 1, nfaces
        enddo ! ie = 1, NEL

        if (NID.eq.0) print *, "YW: FIND WALL PTS"

c$$$ TEST: Write down all the walll points that we have found
#ifdef YWDEBUG
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdwall(ily,ilx), ily=1,NDIM), 
     $         (grdwall(ily,ilx), ily=1,5) 
        enddo
        close(10001)
        endif
        ! call outpost(xwf,ywf,zwf,zwf,zwf,'wpt')
#endif
c$$$ TEST END
        if (numctrl.gt.totctrl) then
                print *, "NUMCTRL=",numctrl
                print *, "YW: ERROR AT NID=",NID,"BEYOND TOTCTRL"
                call exit
        endif
        return 
        end subroutine register_wall_pts 
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine sort_wall_pts
cc YW; A subroutine for sort out the wall points through descending order
cc Boubble sort Algorithm
cc We do this before we interpolate the sensing points
c=============================================
c       Define variable
c=============================================

        implicit none 
        include 'SIZE'
        include "NEKUSE"
        include "PARALLEL"
        include "DRL" ! numctrl, crdwall 

        ! Parameter
        real jx, jy ,jz ! For the second loop
        real jx1, jy1 ,jz1 ! For the second loop
        real vdm     ! Dummy value
        integer sid  ! A dummy variable for inidicies
        
        integer il,jl,kl ! Loop 
        logical swaped           ! if swaped happend
        integer totpts,totfeat,tot_ctrl ! for sake of copy
        integer iglsum 
        !For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================

        if (NUMCTRL.gt.0) then 
        
! step 1: Bubble sort 
!-------------------------------------
        ! Bubble sort         
        do il = 1, NUMCTRL-1
        do jl = 1, NUMCTRL-1
                ! compare the x-coordinate
                jx = crdwall(1,jl)
                jx1 = crdwall(1,jl+1)
                swaped =.FALSE.
                ! We sort by ascent order, i.e. large value at last
                ! Rearrange the array
                if (jx.gt.jx1) then 
                        ! Exchange coordinates info
                        do kl=1,LDIM
                        vdm=crdwall(kl,jl)
                        crdwall(kl,jl)=crdwall(kl,jl+1)
                        crdwall(kl,jl+1)=vdm
                        enddo
                        
                        ! Exchange ieg, iface, ix,iy,iz
                        do kl=1,nfeat
                        sid=grdwall(kl,jl)
                        grdwall(kl,jl)=grdwall(kl,jl+1)
                        grdwall(kl,jl+1)=sid
                        enddo
                        
                        ! Exchange NID info
                        sid = proc(jl)
                        proc(jl)=proc(jl+1)
                        proc(jl+1)=sid

                        ! Switch the flag
                        swaped = .TRUE.
                else
                        swaped = .FALSE.                        
                endif
                if (.NOT.swaped) continue
        enddo ! jl = 1, NUMCTRL-1 
        enddo ! il = 1, NUMCTRL-1
        ! print *,"INDICIES:", indices
                
c$$$ TEST: Write down all the walll points that we have found
#ifdef YWDEBUG
        write(str,"(i4.4)") NID
        open(10001,file="sortedWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "iel  ", "iface  ",
     $                  "ix  ", "iy  ", "iz  "
        do ilx = 1,numctrl
                write(10001,*) proc(ilx),
     $         (crdwall(ily,ilx), ily=1,NDIM), 
     $         (grdwall(ily,ilx), ily=1,5) 
        enddo
        close(10001)
#endif 
c$$$ TEST END
        endif ! if (numctrl.gt.0) 
        
        tot_ctrl = iglsum(NUMCTRL,1)
        
        if (NID.eq.0) print *, "[DRL] STATE SORTING NUMCTRL=",tot_ctrl

        do il=1,TOTCTRL
        call icopy(iptctl(il),grdwall(1,il),1)
        enddo 

        
        return 
        end subroutine sort_wall_pts
c--------------------------------------------------------------------



c--------------------------------------------------------------------
        subroutine interp_sensing_plane
cc YW:  A subroutine for interploation of the sensing points based on specificed y+ 
cc      We use the current wall point (xw,yw,z) to get (xc,yc,z)
cc      xc = xw + dx; yc = yw+dy
c=============================================
c       Define variable
c=============================================

        implicit none 
        include "SIZE"
        include "INPUT"
        include "DRL"
        ! include "NEKUSE"
        include "GEOM" !  The angle unx, uny, unz 
        include 'TOPOL'
        include 'PARALLEL'
        real xw,yw,zw
        real xct,yct,zct
        real snx,sny,snz
        real Ret,nu, h 
        parameter(Ret=180.0)
        parameter(nu=2.0e-5)
        parameter(h=1.0)
        real utaux,ynorm,dyy,dyx
        integer iel, f                     ! For find the face normal 
        integer ie, iface, ix, iy, iz ! indices 
        integer il, jl, kl            ! Iteration
        ! For interploation
        integer nfail 
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c      Function
c=============================================
        ! Calculate viscous velocity u_tau
        !YW: I pre-determine this value because the resolution is very low
        ! The actual value should be -0.917xxxx
        utaux  = Ret * nuctrl / h
        ynorm = ypctrl*nuctrl/utaux ! compute the norm between ctrl and sensing point
        ! ynorm = -0.923879533
        if (NID.eq.0) print *, "[DRL] SENSING PLANE DISTANCE:",ynorm
        
        if (numctrl.gt.0) then  ! Only works when wall point exists in this RANK
        do il=1,numctrl
                ! Wall points coordinates
                xw = crdwall(1,il)
                yw = crdwall(2,il)
                zw = crdwall(3,il)
                ! Only Y-dir changes 
                dyx = 0
                dyy = ynorm
                xct = xw + dyx
                yct = yw + dyy
                zct = zw
                ! Store the position into array
                crdctl(1,il) = xct
                crdctl(2,il) = yct
                crdctl(3,il) = zct
        enddo 
        endif ! IF NUMCTRL.gt.0 

#ifdef YWDEBUG
c$$$ TEST: Write Sensing Points
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRb.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,numctrl
                write(10001,*) iptctl(ilx),
     $         (crdctl(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif 
c$$$ TEST END

        return 
        end subroutine interp_sensing_plane 
c--------------------------------------------------------------------


c--------------------------------------------------------------------
       subroutine register_ctrl_pts 
cc: Defining the sensing plane points and registered on the mesh
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "DRL"
        include 'PARALLEL'
        include 'GEOM'
        include 'NEKUSE'
        integer nfail
        real tolin ! The tolerence for interpolation 
        integer nmsh, nelm, ih
        integer il,jl,kl ! Iteration 
        integer iglsum
        integer ih_intp(2,20)
        common /intp_h/ ih_intp
        
        ! global memory access
        integer nidd,npp,nekcomm,nekgroup,nekreal
        common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal
        integer ntot,npt_max, nxf, nyf, nzf
        real ltim
        real tol, bb_t            ! interpolation tolerance and relative size to expand bounding boxes by
        
        ! The problem is to make this bb_t larger for more inflation
        parameter (tol = 5.0E-13, bb_t = 0.1)
        
        integer totpts,totfail
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

c SetUp for the interploation
c----------------------------
        
      ! initialise findpts
        ntot=lx1*ly1*lz1*lelt 
        npt_max=256
        nxf=2*lx1 ! fine mesh for bb-test
        nyf=2*ly1
        nzf=2*lz1
        

        ! call interp_free(inth_hpts1)
        call fgslib_findpts_setup(inth_hpts1,nekcomm,npp,ldim,
     $     xm1,ym1,zm1,lx1,ly1,lz1,lelt,nxf,nyf,nzf,bb_t,ntot,ntot,
     $     npt_max,tol)
     
        ! inth_hpts2 = inth_hpts1
        
c Interpolation, the same procedure in time-series 
c-----------------------
        ! do il=1,totctrl
        !         iwk(il,3)=NID
        ! enddo 

        call fgslib_findpts(inth_hpts1,
     &     iwk(1,1),1,                      ! $ rcode 1
     &     iwk(1,3),1,                      ! $ proc,1
     &     iwk(1,2),1,                      ! $ elid, 1
     &     rwk(1,2),LDIM,                   ! $ rst, ndim
     &     rwk(1,1),1,                      ! $ dist,1
     &     crdctl(1,1),ldim,                   ! $ x
     &     crdctl(2,1),ldim,                   ! $ y
     &     crdctl(3,1),ldim,totctrl)           ! $ z

c Examine the deviations after interoplation
c----------------------
        nfail = 0 
        ! do il=1,totctrl
        do il=1,numctrl
        ! check return code
        if(iwk(il,1).eq.1) then
          if(rwk(il,1).gt.10*tolin) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     crdctl(1,il),crdctl(2,il),crdctl(3,il),rwk(il,1)
          endif
        elseif(iwk(il,1).eq.2) then
          nfail = nfail + 1
          if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        crdctl(1,il),crdctl(2,il),crdctl(3,il)
        endif
        enddo
        
        totfail=iglsum(nfail,1)
        if (NID.eq.0) print *,"[DRL] Failed Interp:",totfail

        ! Interesting; Even though all interpolation data seems to be fine I have to reinitialise it here
      ! It looks like gslib has some internal data that has to be wahsed up; Must be new stuff not present in older gslib versions



c Initialise the Velocity tensor 
cc Idea is to mask the tensor with a constant but pretty large value 
cc So it can become a condition when imposing the velocity 
c-------------------------------------
        ! totpts =LX1*LY1*LZ1*LELT*6
        ! do il=1,totpts
        !         Vnfluct(il,1,1,1,1) = dumi
        ! enddo 
c-----------------------
#ifdef YWDEBUG
c$$$ TEST: Write down all the sensing points that we have found
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="findCTRL.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,numctrl
                write(10001,*) iptctl(ilx),
     $         (crdctl(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif
c$$$ TEST END
c-----------------------
        return 
        end subroutine register_ctrl_pts
c--------------------------------------------------------------------

        subroutine interp_wall_pts 
cc: Defining the sensing plane points and registered on the mesh
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include "DRL"
        include 'PARALLEL'
        include 'GEOM'
        include 'NEKUSE'
        integer nfail
        real tolin ! The tolerence for interpolation 
        integer nmsh, nelm, ih
        integer il,jl,kl ! Iteration 
        integer iglsum
        integer ih_intp(2,20)
        common /intp_h/ ih_intp
        
        ! global memory access
        integer nidd,npp,nekcomm,nekgroup,nekreal
        common /nekmpi/ nidd,npp,nekcomm,nekgroup,nekreal
        integer ntot,npt_max, nxf, nyf, nzf
        real ltim
        real tol, bb_t            ! interpolation tolerance and relative size to expand bounding boxes by
        
        ! The problem is to make this bb_t larger for more inflation
        parameter (tol = 5.0E-13, bb_t = 0.1)
        
        integer totpts,totfail
        ! For test 
        integer ilx, ily
        character*4 str, str1
c=============================================
c       Function
c=============================================

c------- Interplolation of Wall points---------------------------

c SetUp for the interploation
c----------------------------
      ! initialise findpts
        ntot=lx1*ly1*lz1*lelt 
        npt_max=256
        nxf=2*lx1 ! fine mesh for bb-test
        nyf=2*ly1
        nzf=2*lz1
        


        if (NID.eq.0) print *,"[NEK] FGSLIB FINDPTS WALL"

        ! call fgslib_findpts_free(inth_hpts2)
        ! inth_hpts2=ih_intp(2,20)
!         call fgslib_findpts_setup(inth_hpts2,nekcomm,npp,ldim,
!      $     xm1,ym1,zm1,lx1,ly1,lz1,lelt,nxf,nyf,nzf,bb_t,ntot,ntot,
!      $     npt_max,tol)

        call fgslib_findpts(inth_hpts1,
     &     iwk_wall(1,1),1,                      ! $ rcode 1
     &     iwk_wall(1,3),1,                      ! $ proc,1
     &     iwk_wall(1,2),1,                      ! $ elid, 1
     &     rwk_wall(1,2),LDIM,                   ! $ rst, ndim
     &     rwk_wall(1,1),1,                      ! $ dist,1
     &     crdwall(1,1),ldim,                   ! $ x
     &     crdwall(2,1),ldim,                   ! $ y
     &     crdwall(3,1),ldim,totctrl)           ! $ z
        
       if (NID.eq.0) print *,"[NEK] END FGSLIB FINDPTS WALL" 

c Examine the deviations after interoplation
c----------------------
        nfail = 0 
        ! do il=1,totctrl
        do il=1,numctrl
        ! check return code
        if(iwk_wall(il,1).eq.1) then
          if(rwk_wall(il,1).gt.10*tolin) then
            nfail = nfail + 1
            if (nfail.le.5) write(6,'(a,1p4e15.7)')
     &     ' WARNING: point on boundary or outside the mesh xy[z]d^2: ',
     &     crdwall(1,il),crdwall(2,il),crdwall(3,il),rwk_wall(il,1)
          endif
        elseif(iwk_wall(il,1).eq.2) then
          nfail = nfail + 1
          if (nfail.le.5) write(6,'(a,1p3e15.7)')
     &        ' WARNING: point not within mesh xy[z]: !',
     &        crdwall(1,il),crdwall(2,il),crdwall(3,il)
        endif
        enddo
        
        ! totfail=iglsum(nfail,1)
        ! if (NID.eq.0) print *,"[DRL] Failed Interp:",totfail

c-----------------------
#ifdef YWDEBUG
c$$$ TEST: Write down all the sensing points that we have found
        if (numctrl.gt.0) then 
        write(str,"(i4.4)") NID
        open(10001,file="interpWall.txt"//str)
        write(10001,*) "NID  ", "X  ", "Y  ", "Z  "
        do ilx = 1,numctrl
                write(10001,*) NID,
     $         (crdwall(ily,ilx), ily=1,NDIM) 
        enddo
        close(10001)
        endif
#endif
c$$$ TEST END
c-----------------------
        return 
        end subroutine interp_wall_pts
c--------------------------------------------------------------------
