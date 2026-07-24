c-------------------------------------
! All subroutines for the rewards func
! TODO: Need to be eventually adopted to the wing  
c-------------------------------------

! SOME PREDEFINED PARAM
#define INTP_NMAX 200 /* number of sample points */
#define XCINT 0.0     /* x coordinate of 1D line*/
#define ZCINT 0.0     /* z coordinate of 1D line */

c mesh dimensions
#define BETAM 1.0     /* wall normal stretching parameter */
#define PI (4.*atan(1.))

c------------------------------------------------------------------
        subroutine drl_reward(i_evolv)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include "INPUT"
        include 'PARALLEL'
        integer i_evolv
        integer reward_mode !#[MOD] runtime reward-mode selector
c=============================================
c       Function
c=============================================
        
        ! if (ISTEP.ne.0) then
        
        if (NID.eq.0) then
                print *,"--------------------------------"
                print *, "[REWARD] INQURY"
                print *,"--------------------------------"
        endif

        ! YW: OCT15 I got a issue regarding the MEMORY
        ! I comment this and will test it on cluster in the future.
        !-----------------
!#[MOD] was: #ifdef NETGAIN / <netgain> / #else / compute_dudy / #endif
        reward_mode = nint(UPARAM(9))
        if (reward_mode.eq.1) then
        call compute_netGain(i_evolv)
        else
        call compute_dudy(i_evolv)
        endif
        !-----------------
        call drl_reward_out(i_evolv)
        
        if (NIO.eq.0) then  
            print *, "----------------------"
            print *, "[REWARD] GET!"
            print *, "----------------------"
        endif 
        
        ! endif 
        end subroutine
c------------------------------------------------------------------

c------------------------------------------------------------------
        subroutine compute_dudy(i_evolv)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include 'SIZE'
        include 'SOLN'     ! vx,vy,vz
        include 'PARALLEL'
        ! include 'GEOM'     ! rxm1
        include 'NEKUSE'
        include 'TSTEP'
        include 'DRL'
        ! For calculating Derivative
        real duidxj(LX1,LY1,LZ1,lelt,3)
        ! real devU2(LX1*LY1*LZ1*lelt,1)
        real devU1(LX1,LY1,LZ1,lelt)

        ! Doing average
        real velV(LX1,LY1,LZ1,LELT),avgV(LX1,LY1,LZ1,LELT)
        real avgVZ(LX1,LY1,xnel,ynel)   
        real avgVX(LX1,LY1,ynel,znel)
        
        ! Iteration 
        integer i_evolv
        real    rwd_i, rwd_c ! input and current reward in the buffer
        ! real rewrd(TOTCTRL)
        integer im, jm, km, fmid(6) ! Face mid points on each direction 
        integer ie,iface,ix,iy,iz,lgi ! Iteration
        integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2 ! Face related
        integer ntot,nxyz
        integer igs_x,igs_z
        save igs_x,igs_z
        integer il,jl,kl,ll,ilx,ily
        character*4 str, str1
c=============================================
c       Function
c=============================================
c-----------------------------------------------
c: Step 1: copy the current array into a new array for computing the Mean
c-----------------------------------------------
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] XNEL,YNEL:",XNEL,YNEL
#endif

        nxyz=LX1*LY1*LZ1
        ntot=LX1*LY1*LZ1*LELT

        if (igs_z.eq.0.and.igs_x.eq.0) then 
        ! call interp_wall_pts
        call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
        call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
        if (NID.eq.0) print *, "[REWARD] AVG HANDLE INIT!",igs_z,igs_x
        endif 

c-----------------------------------------------
c: Step 2: Compute the Derivatives 
c-----------------------------------------------
        ! Copy the U into devU1
        call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)
        call gradm1(duidxj(1,1,1,1,1),
     $              duidxj(1,1,1,1,2),
     $              duidxj(1,1,1,1,3),
     $              devU1(1,1,1,1))

c-----------------------------------------------
c: Step 3: Compute the spatial Mean 
c-----------------------------------------------

        ! We know dUdy is at 2-th indicies!
        call copy(velV(1,1,1,1),duidxj(1,1,1,1,2),ntot)

!------- NOTE  THIS ONLY WORKED FOR NEW version!-----------
        if (rwd_zavg) then 
        ! call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
        call planar_avg(avgV,velV,igs_z)
        call copy(velV,avgV,ntot)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] Z-AVG!"
#endif
        endif ! if (rwd_zavg)

        ! Do streamwise average if it allowed/defined
        if (rwd_xavg) then 
        ! call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
        call planar_avg(avgV,velV,igs_x)
        call copy(velV,avgV,ntot)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] X-AVG!"
#endif
        endif 


c-----------------------------------------------
c: Step 4: Get the value at the Agent 
c-----------------------------------------------
        ! Initialize the array: 
        if (i_evolv.eq.1) call rzero(rwd_agt,TOTCTRL)

        ! Moving average
        do il=1,NUMCTRL
        ie=info_agt(1,il)
        ie=gllel(ie)
        iface=info_agt(2,il)
        ix=info_agt(3,il)
        iy=info_agt(4,il)
        iz=info_agt(5,il)
        rwd_i=velV(ix,iy,iz,ie)
        rwd_c=rwd_agt(il)
        
        ! Moving Average
        if (i_evolv.eq.1) then
        rwd_agt(il)=rwd_i
        else
        rwd_agt(il)=(rwd_c*(i_evolv-1)+rwd_i)/i_evolv
        endif

        enddo ! do il=1,NUMCTRL

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy ASSIGNED!"
#endif

#ifdef YWDEBUG
        if (ISTEP.le.3) then 
        if (NUMCTRL.ne.0) then 

        write(str,"(i4.4)") NID
        open(51001,file="dUdy_Wall.txt"//str)
        write(51001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  ",
     $                  "dUdy  "
        do ilx = 1,numctrl
                write(51001,*) proc(ilx),
     $         (pos_agt(ily,ilx), ily=1,NDIM), 
     $         (info_agt(ily,ilx), ily=1,5),
     $         rwd_agt(ilx) 
        enddo
        close(51001)
        endif ! if (NUMCTRL.ne.0)
        endif 
#endif

        end subroutine compute_dudy


c------------------------------------------------------------------
      subroutine compute_netGain(i_evolv)
c Net energy saving: sends (tau_w + |p'_w*v_w| + 0.5*|v^3_w|) to Python
c Python reward = 1 - sent/tau_w_ref = (Cf_ref - Cf_ctrl - win) / Cf_ref
c where win = |p'_w*v_w| + 0.5*|v^3_w|
c=============================================
c       Define variable
c=============================================
         implicit none
         include 'SIZE'
         include 'TOTAL'
         include 'DRL'
         real duidxj(LX1,LY1,LZ1,lelt,3)
         real devU1(LX1,LY1,LZ1,lelt), ! derivative of U
     $     devU2(LX1,LY1,LZ1,lelt) ! derviative of V

         real velV(LX1,LY1,LZ1,LELT),avgV(LX1,LY1,LZ1,LELT)
         real avgVZ(LX1,LY1,xnel,ynel), avgVX(LX1,LY1,ynel,znel)

         real denu,rho
         real dudy_i, tau_w, pwvw_i, v3_i, dvdx_i, uv_i ! Scalars 
         real tauw(LX1,LY1,LZ1,LELT)
         real pwvw(LX1,LY1,LZ1,LELT), v3(LX1,LY1,LZ1,LELT), 
     $         uv_fluct(LX1,LY1,LZ1,LELT)
         real buffer(LX1,LY1,LZ1,LELT), wrk_buff(LX1,LY1,LZ1,LELT)

         integer i_evolv, n_drl
         real    rwd_i, rwd_c
         integer im, jm, km, fmid(6)
         integer ie,iface,ix,iy,iz,lgi
         integer NEL,nfaces,KX1,KX2,KY1,KY2,KZ1,KZ2
         integer ntot,nxyz
         integer igs_x,igs_z
         save igs_x,igs_z
         integer il,jl,kl,ll,ilx,ily
         character*4 str, str1
c=============================================
c       Function
c=============================================
         rho  = param(1)
         denu = param(2)
         n_drl = UPARAM(1)

         nxyz=LX1*LY1*LZ1
         ntot=LX1*LY1*LZ1*LELT

         if (igs_z.eq.0.and.igs_x.eq.0) then
            call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3)
            call gtpp_gs_setup(igs_x,xnel,ynel,znel,1)
            if (NID.eq.0) print *, "[REWARD] NETGAIN HANDLE INIT!",
     $                             igs_z,igs_x
         endif

c------- Step 1: Compute dUdy and apply spatial average
         call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)
         call gradm1(duidxj(1,1,1,1,1), ! dUdx
     $               duidxj(1,1,1,1,2), ! dUdy 
     $               duidxj(1,1,1,1,3), ! dUdz
     $               devU1(1,1,1,1))
         call copy(velV(1,1,1,1),duidxj(1,1,1,1,2),ntot)

        ! Shortly detach from taking the average
        !  if (rwd_zavg) then
        !     call planar_avg(avgV,velV,igs_z)
        !     call copy(velV,avgV,ntot)
        !  endif
        !  if (rwd_xavg) then
        !     call planar_avg(avgV,velV,igs_x)
        !     call copy(velV,avgV,ntot)
        !  endif
        ! Get the averaged dUdy
        call copy(devU1(1,1,1,1),velV(1,1,1,1),ntot)

! -------------------- [YW] Compute dVdx and apply spatial average
         call copy(devU2(1,1,1,1),vy(1,1,1,1),ntot)
         call gradm1(duidxj(1,1,1,1,1), ! dUdx
     $               duidxj(1,1,1,1,2), ! dUdy 
     $               duidxj(1,1,1,1,3), ! dUdz
     $               devU2(1,1,1,1))
         call copy(velV(1,1,1,1),duidxj(1,1,1,1,1),ntot)

        ! Shortly get rid of the average [11:06]
        !  if (rwd_zavg) then
        !     call planar_avg(avgV,velV,igs_z)
        !     call copy(velV,avgV,ntot)
        !  endif
        !  if (rwd_xavg) then
        !     call planar_avg(avgV,velV,igs_x)
        !     call copy(velV,avgV,ntot)
        !  endif
        ! Get the averaged dUdy
        call copy(devU2(1,1,1,1),velV(1,1,1,1),ntot)
!-------------------- End Compute dVdx here ----------------------

! [YW] Modify the pressure fluctuation term 
c------- Step 2: Compute pressure-velocity term |pprime_w * v_w|
         ! [IMPORTANT] First map pressure to the GLL space
         call mappr(buffer,pr,pwvw,velV) ! The last two tensor is just for grabage
         ! Normalized the pressure by subtracting the wall integration pressure.
       !  call normal_pressure(buffer)
         ! Give a copy of it
         call copy(pwvw(1,1,1,1),buffer(1,1,1,1),ntot)
         if (rwd_zavg) then
            call planar_avg(avgV,buffer,igs_z)
            call copy(buffer,avgV,ntot)
         endif
         if (rwd_xavg) then
            call planar_avg(avgV,buffer,igs_x)
            call copy(buffer,avgV,ntot)
         endif
         ! wrk_buff <= pressure_mapped - mean 
         call sub3(wrk_buff,pwvw,buffer,ntot)
         ! buffer <= wrk_buff == pre fluctuation
         call copy(buffer(1,1,1,1),wrk_buff(1,1,1,1),ntot)
         ! wrk_buff <= v_w (zero-mean)
         call copy(wrk_buff(1,1,1,1),ACTIONS(1,1,1,1),ntot)
         ! Correlation: pwvw = p'_w * v_w
         call col3(pwvw,buffer,wrk_buff,ntot)
        
        !! YW: trail one is to take the correlation first and then do the averaging, 
        !! but it seems to be more noisy than doing the average first.
        ! pwvw = abs(pwvw)
        call copy(buffer(1,1,1,1),pwvw(1,1,1,1),ntot)
        if (rwd_zavg) then
            call planar_avg(avgV,buffer,igs_z)
            call copy(buffer,avgV,ntot)
        endif
        if (rwd_xavg) then
            call planar_avg(avgV,buffer,igs_x)
            call copy(buffer,avgV,ntot)
        endif
        ! Make copy
        call copy(pwvw(1,1,1,1),buffer(1,1,1,1),ntot)
        pwvw = abs(pwvw)
        
        

c------- Step 3: Compute kinetic energy term 0.5*|v^3_w|
         call copy(v3(1,1,1,1),ACTIONS(1,1,1,1),ntot)
         v3 = 0.5 * abs(v3**3)
         call copy(buffer(1,1,1,1),v3(1,1,1,1),ntot)
         if (rwd_zavg) then
            call planar_avg(avgV,buffer,igs_z)
            call copy(buffer,avgV,ntot)
         endif
         if (rwd_xavg) then
            call planar_avg(avgV,buffer,igs_x)
            call copy(buffer,avgV,ntot)
         endif
         call copy(v3(1,1,1,1),buffer(1,1,1,1),ntot)

c------- Step 4: Compute uv term u' * v_wall
         call copy(uv_fluct(1,1,1,1),VX(1,1,1,1),ntot) ! Copy from buffer
         call copy(buffer(1,1,1,1),uv_fluct(1,1,1,1),ntot) ! Copy for avg
         if (rwd_zavg) then
            call planar_avg(avgV,buffer,igs_z)
            call copy(buffer,avgV,ntot)
         endif
         if (rwd_xavg) then
            call planar_avg(avgV,buffer,igs_x)
            call copy(buffer,avgV,ntot)
         endif
         ! uv_fluct = u - u_avg
         call sub3(wrk_buff,uv_fluct,buffer,ntot)
         call copy(uv_fluct(1,1,1,1),wrk_buff(1,1,1,1),ntot)
         ! v_wall -> buffer
         call copy(buffer(1,1,1,1),ACTIONS(1,1,1,1),ntot)
         ! Take multiplication to buffer 
         call col3(wrk_buff,uv_fluct,buffer, ntot)
         ! Now put it back uv_fluct <- wrk_buff, done
         call copy(uv_fluct(1,1,1,1),wrk_buff(1,1,1,1),ntot)

c------- Step 5: Assemble reward at each agent location
         if (i_evolv.eq.1) call rzero(rwd_agt,TOTCTRL)

         do il=1,NUMCTRL
            ie=info_agt(1,il)
            ie=gllel(ie)
            iface=info_agt(2,il)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)

            dudy_i = devU1(ix,iy,iz,ie)
            tau_w  = rho * denu * dudy_i
            pwvw_i = pwvw(ix,iy,iz,ie)
            v3_i   = v3(ix,iy,iz,ie)
c ---------------Added for investigating the new terms --------
            dvdx_i = devU2(ix,iy,iz,ie)
            uv_i   = uv_fluct(ix,iy,iz,ie)  

            if (i_evolv.eq.1) then
               rwd_tau(il) = tau_w
               rwd_pw(il)  = pwvw_i
               rwd_v3(il)  = v3_i
c------------ Added for investigating the new terms --------
               rwd_dvdx(il)= dvdx_i
               rwd_uv(il)  = uv_i

            else
               rwd_tau(il) = (rwd_tau(il)*(i_evolv-1) + tau_w)
     $                     / i_evolv
               rwd_pw(il)  = (rwd_pw(il)*(i_evolv-1) + pwvw_i)
     $                     / i_evolv
               rwd_v3(il)  = (rwd_v3(il)*(i_evolv-1) + v3_i)
     $                     / i_evolv

c------------ Added for investigating the new terms --------
               rwd_dvdx(il)  = (rwd_dvdx(il)*(i_evolv-1) + dvdx_i)
     $                     / i_evolv
               rwd_uv(il)  = (rwd_uv(il)*(i_evolv-1) + uv_i)
     $                     / i_evolv
            endif
         enddo

#ifdef GAINMONITOR
        if (i_evolv.eq.n_drl) then
                call write_reward_monitor(i_evolv)
        endif
#endif

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] NETGAIN ASSIGNED!"
        if (ISTEP.le.3) then 
        if (NUMCTRL.ne.0) then 

        write(str,"(i4.4)") NID
        open(51001,file="Reward.txt"//str)
        write(51001,*) "NID  ", "X  ", "Y  ", "Z  ",
     $                  "ie  ", "iface  ", "ix ",
     $                  "iy  ", "iz  ", "nid  ",
     $                  "tauw  ", "pwvw  ", "v3  "
        do ilx = 1,numctrl
                write(51001,*) proc(ilx),
     $         (pos_agt(ily,ilx), ily=1,NDIM), 
     $         (info_agt(ily,ilx), ily=1,5),
     $         rwd_tau(ilx), rwd_pw(ilx), rwd_v3(ilx)
        enddo
        close(51001)
        endif ! if (NUMCTRL.ne.0)
        endif 
#endif

      end subroutine compute_netGain
        
c------------------------------------------------------
      subroutine normal_pressure(pres)
c=============================================
c       Define variable
c=============================================
         implicit none
         include 'SIZE'
         include 'SOLN'
         include 'INPUT'
         include 'NEKUSE'
         include 'PARALLEL'
         include 'TSTEP'
         include 'TOPOL'
         include 'GEOM' !unx, uny, unz
         include 'DRL'

         real pres(LX1,LY1,LZ1,LELT)
         real vrtmp(lx1*lz1)       ! work array for face
         real vrtmp2(2)            ! work array
         real vlsum ! function

         integer ifll,itmp
         integer il,jl,kl,ll,ilx,ily
         integer ictrl
         character*4 str, str1
c=============================================
c       Function
c=============================================
      ! normalise pressure
      ! in this example I integrate pressure over top faces marked "W"
      ifll = 1     ! I'm interested in velocity bc
      ! relying on mesh structure given by genbox set face number
      jl = 3
      call rzero(vrtmp2,2)  ! zero work array
      itmp = LX1*LZ1
      do ictrl=1,NUMCTRL   ! loop across the agents, here represents the entire wall but not applied for the wing
         il = info_agt(1,ictrl)
         il = gllel(il)
         jl = info_agt(2,ictrl)
         vrtmp2(1) = vrtmp2(1) + vlsum(area(1,1,jl,il),itmp)
         call ftovec(vrtmp,pres,il,jl,lx1,ly1,lz1)
         call col2(vrtmp,area(1,1,jl,il),itmp)
         vrtmp2(2) = vrtmp2(2) + vlsum(vrtmp,itmp)
      enddo
      ! global communication
      call gop(vrtmp2,vrtmp,'+  ',2)
      ! missing error check vrtmp2(1) == 0
      vrtmp2(2) = -vrtmp2(2)/vrtmp2(1)
      ! remove mean pressure
      itmp = LX1*LY1*LZ1*NELV

#ifdef YWDEBUG
      if (NID.eq.0) print *, "[NORMAL PRESSURE]", vrtmp2(2)
#endif
      call cadd(pres,vrtmp2(2),itmp)

      end subroutine normal_pressure

c------------------------------------------------------
      subroutine write_reward_monitor(i_evolv)
c     Append a one-line time-series record of the three NETGAIN reward
c     components to reward_monitor.dat (rank-0 only).
c     All agents share the same value after x/z averaging, so a global
c     mean across agents recovers the uniform scalar cleanly.
c=============================================
      implicit none
      include 'SIZE'
      include "NEKUSE"
      include "SOLN"
      include "TSTEP"
      include 'PARALLEL'
      include 'DRL'

      integer i_evolv, iglsum
      integer il, nc
      real wrk(5), tmp3(5)
      logical fexist
      integer, parameter :: iunit = 52002
      integer, parameter :: MAX_MON_LINES = 10000
      integer, save      :: ifile = 0
c     [MOD] nrec: records already in the current file, carried in memory so
c     [MOD] only the first record of a run has to touch the disk.
c     [MOD] hdrdon: the header of the current file has been dealt with.
      integer, save      :: nrec = -1
      logical, save      :: hdrdon = .false.
      integer :: nlines, ntot, ios
      character(len=200) :: cbuf
      character(len=40)  :: fname
c=============================================
      ! Sum local agent values; divide by global count for the mean.
      ! Since rwd_xavg=rwd_zavg=.TRUE., all agents hold the same value,
      ! so mean == that value regardless of how agents are distributed.
      nc     = iglsum(NUMCTRL, 1)
      wrk(1) = 0.0
      wrk(2) = 0.0
      wrk(3) = 0.0
      wrk(4) = 0.0
      wrk(5) = 0.0
      if (NUMCTRL.gt.0) then
         do il = 1, NUMCTRL
            wrk(1) = wrk(1) + rwd_tau(il)
            wrk(2) = wrk(2) + rwd_pw(il)
            wrk(3) = wrk(3) + rwd_v3(il)
            wrk(4) = wrk(4) + rwd_dvdx(il)
            wrk(5) = wrk(5) + rwd_uv(il)
         enddo
      endif
      call gop(wrk, tmp3, '+  ', 3)
      if (nc.gt.0) then
         wrk(1) = wrk(1) / nc
         wrk(2) = wrk(2) / nc
         wrk(3) = wrk(3) / nc
         wrk(4) = wrk(4) / nc
         wrk(5) = wrk(5) / nc
      endif

      if (NID.eq.0) then
c        [MOD] Locate the file to append to. The old code probed only
c        [MOD] reward_monitor<ifile>.dat, and on a full file bumped ifile once
c        [MOD] and opened the result with status='new' -- a fatal runtime error
c        [MOD] whenever that file already existed, which is exactly what a
c        [MOD] halted evaluation leaves behind in eval/env_XXX. A partially
c        [MOD] filled leftover was worse: it was appended to, splicing two
c        [MOD] independent trajectories into one series with no error at all.
         if (nrec.lt.0) then
            do while (ifile.lt.99999 .and. nrec.lt.0)
               write(fname,'(A,I5.5,A)')
     $            'reward_monitor', ifile, '.dat'
               inquire(file=trim(fname), exist=fexist)
               if (.not.fexist) then
                  nrec   = 0
                  hdrdon = .false.
               else
                  nlines = 0
                  ntot   = 0
                  open(iunit, file=trim(fname), status='old',
     $                 iostat=ios)
                  if (ios.ne.0) then
                     write(6,*) '[MONITOR] SKIP UNREADABLE ',
     $                          trim(fname)
                     ifile = ifile + 1
                  else
c                    [MOD] Count every non-comment line, instead of assuming
c                    [MOD] exactly one header line sits at the top.
                     do
                        read(iunit,'(A)',iostat=ios) cbuf
                        if (ios.ne.0) exit
                        ntot = ntot + 1
                        if (cbuf(1:1).ne.'#') nlines = nlines + 1
                     end do
                     close(iunit)
                     if (nlines.lt.MAX_MON_LINES) then
                        nrec   = nlines
                        hdrdon = ntot.gt.0
                     else
                        ifile = ifile + 1
                     end if
                  end if
               end if
            end do
            if (nrec.lt.0) then
               write(6,*) '[MONITOR] NO WRITABLE FILE, RECORD DROPPED'
               return
            end if
         end if
c        [MOD] Roll over once the current file is full.
         if (nrec.ge.MAX_MON_LINES) then
            ifile  = ifile + 1
            nrec   = 0
            hdrdon = .false.
         end if
         write(fname,'(A,I5.5,A)') 'reward_monitor', ifile, '.dat'
c        [MOD] Always append, never status='new'. A monitor record is a
c        [MOD] diagnostic side-channel, so an I/O failure warns and drops the
c        [MOD] record rather than aborting: killing rank 0 here would leave the
c        [MOD] solver ranks in the next collective, i.e. a whole-job deadlock.
c        [MOD] Returning is safe, the collectives above are already done.
         open(iunit, file=trim(fname), position='append', iostat=ios)
         if (ios.ne.0) then
            write(6,*) '[MONITOR] CANNOT OPEN ', trim(fname),
     $                 ' iostat=', ios, ' RECORD DROPPED'
            nrec = -1
            return
         end if
         if (.not.hdrdon) then
            write(iunit,'(A)')
     $         '# time          i_evolv'//
     $         '  rwd_tau         rwd_pw          rwd_v3'//
     $            '  rwd_dvdx         rwd_uv'
            hdrdon = .true.
         end if
         write(iunit,'(E16.8,1X,I6,5(1X,E16.8))')
     $      time, i_evolv, wrk(1), wrk(2), wrk(3),
     $                      wrk(4), wrk(5)
         close(iunit)
         nrec = nrec + 1
        print *, "[MONITOR] RECORD",time,i_evolv,wrk(1),wrk(2),wrk(3),
     $                               wrk(4), wrk(5)
      endif

      end subroutine write_reward_monitor