c-------------------------------------
! All subroutines for the rewards func
! TODO: Need to be eventually adopted to the wing  
c-------------------------------------
#define PI (4.*atan(1.))

c------------------------------------------------------------------
        subroutine drl_reward(i_evolv)
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include 'PARALLEL'
        integer i_evolv
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
#ifdef NETGAIN
        call compute_netGain_posOnly(i_evolv)
#else
        call compute_dudy(i_evolv)
#endif
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
        real dUdx(LX1,LY1,LZ1,LELT),
     $       dUdy(LX1,LY1,LZ1,LELT),
     $       avgV(LX1,LY1,LZ1,LELT)   

        real avgVZ(LX1,LY1,xnel,ynel),avgVX(LX1,LY1,ynel,znel)
        
        ! Incorporate the temporal evolution
        integer i_evolv
        real rwd_i, rwd_c ! instance of reward
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
c-----------------------------------------------
c: Step 2: Compute the Derivatives 
c-----------------------------------------------
        ! Copy the U into devU1
        call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] Copy Buffer!"
#endif
        call gradm1(duidxj(1,1,1,1,1),
     $              duidxj(1,1,1,1,2),
     $              duidxj(1,1,1,1,3),
     $              devU1(1,1,1,1))

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy Calculated!"
#endif
c-----------------------------------------------
c: Step 3: Compute the spatial Mean 
c-----------------------------------------------

        ! We know dUdy is at 2-th indicies!
        call copy(dUdx(1,1,1,1),duidxj(1,1,1,1,1),ntot)
        call copy(dUdy(1,1,1,1),duidxj(1,1,1,1,2),ntot)


#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy COPIED!"
#endif

!------- NOTE  THIS ONLY WORKED FOR OLD version!-----------
        if (rwd_zavg) then 
        call z_averaging(dUdx,avgVZ)
        call z_avg_reshape(dUdx,avgVZ)

        call z_averaging(dUdy,avgVZ)
        call z_avg_reshape(dUdy,avgVZ)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dU2dxi Z-AVG!"
#endif
        endif ! if (rwd_zavg)

        ! Do streamwise average if it allowed/defined
        if (rwd_xavg) then 
        call x_averaging(dUdx,avgVX)
        call x_avg_reshape(dUdx,avgVX)
        call x_averaging(dUdy,avgVX)
        call x_avg_reshape(dUdy,avgVX)

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] X-AVG!"
#endif
        endif 

c-----------------------------------------------
c: Step 4: Get the value at the Agent 
c-----------------------------------------------

        ! Rotate the dUdy according to the derivation, 
        ! remember body_sin == cosTheta 
        dUdy = -body_cos * body_sin * dUdx +
     &         -body_cos * body_cos * dUdx +
     &         +body_sin * body_sin * dUdy +
     &         +body_cos * body_sin * dUdy 

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy ROTATION DONE!"
#endif

        do il=1,NUMCTRL
        ie=info_agt(1,il)
        ie=gllel(ie)
        iface=info_agt(2,il)
        ix=info_agt(3,il)
        iy=info_agt(4,il)
        iz=info_agt(5,il)
        rwd_i = ABS(dUdy(ix,iy,iz,ie))
        rwd_c = rwd_agt(il)
        
        ! Moving Average
        if (i_evolv.eq.1) then 
        rwd_agt(il)=rwd_i
        else 
        rwd_agt(il)=(rwd_c*i_evolv+rwd_i)/(i_evolv+1) 
        endif 
        
        enddo ! do il=1,NUMCTRL

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy ASSIGNED!"
#endif

#ifdef YWDEBUG
        if (ISTEP.le.6) then 
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
      subroutine compute_netGain_posOnly(i_evolv)
c Net energy saving: sends (tau_w + |p'_w*v_w| + 0.5*|v^3_w|) to Python
c Python reward = 1 - sent/tau_w_ref = (Cf_ref - Cf_ctrl - win) / Cf_ref
c where win = p'_w*v_w (positive only) + 0.5*|v^3_w|
c=============================================
c       Define variable
c=============================================
         implicit none
         include 'SIZE'
         include "INPUT"
         include 'PARALLEL'
         include 'SOLN'     ! rxm1
         include 'NEKUSE'
         include 'TSTEP'
         include 'USERPAR'
         include 'DRL'
         real duidxj(LX1,LY1,LZ1,lelt,3)
         real devU1(LX1,LY1,LZ1,lelt)
        
         real dUdx(LX1,LY1,LZ1,LELT),
     $       dUdy(LX1,LY1,LZ1,LELT)

         real velV(LX1,LY1,LZ1,LELT),avgV(LX1,LY1,LZ1,LELT)
         real avgVZ(LX1,LY1,xnel,ynel)
         real avgVX(LX1,LY1,ynel,znel)

         real denu,rho
         real dudy_i, tau_w, pwvw_i, v3_i
         real vnorm(LX1,LY1,LZ1,LELT) ! The wall-normal component actuation
         real tauw(LX1,LY1,LZ1,LELT)
         real pwvw(LX1,LY1,LZ1,LELT), v3(LX1,LY1,LZ1,LELT)
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
         rho  = real(param(1)) ! density,  Check .rea file
         denu = abs(real(param(2))) ! dynamic viscosity,  Check .rea file
         n_drl = int(PARAM(89))

         nxyz=LX1*LY1*LZ1
         ntot=LX1*LY1*LZ1*LELT

c----- Step1: Wall-shear stress --------------------
        ! Copy the U into devU1
        call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] Copy Buffer!"
#endif
        call gradm1(duidxj(1,1,1,1,1),
     $              duidxj(1,1,1,1,2),
     $              duidxj(1,1,1,1,3),
     $              devU1(1,1,1,1))

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy Calculated!"
#endif
        ! We know dUdy is at 2-th indicies!
        call copy(dUdx(1,1,1,1),duidxj(1,1,1,1,1),ntot)
        call copy(dUdy(1,1,1,1),duidxj(1,1,1,1,2),ntot)

#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy COPIED!"
#endif

!------- NOTE  THIS ONLY WORKED FOR OLD version!-----------
        if (rwd_zavg) then 
        call z_averaging(dUdx,avgVZ)
        call z_avg_reshape(dUdx,avgVZ)

        call z_averaging(dUdy,avgVZ)
        call z_avg_reshape(dUdy,avgVZ)
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dU2dxi Z-AVG!"
#endif
        endif ! if (rwd_zavg)
        ! Do streamwise average if it allowed/defined
        if (rwd_xavg) then 
        call x_averaging(dUdx,avgVX)
        call x_avg_reshape(dUdx,avgVX)
        call x_averaging(dUdy,avgVX)
        call x_avg_reshape(dUdy,avgVX)

        ! Rotate the dUdy according to the derivation, 
        ! remember body_sin == cosTheta 
        dUdy = -body_cos * body_sin * dUdx +
     &         -body_cos * body_cos * dUdx +
     &         +body_sin * body_sin * dUdy +
     &         +body_cos * body_sin * dUdy 
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] dUdy ROTATION DONE!"
#endif
#ifdef YWDEBUG
        if (NID.eq.0) print *, "[REWARD] X-AVG!"
#endif
        endif 
c--------------------------------------------------



! [YW] Modify the pressure fluctuation term 
c------- Step 2: Compute pressure-velocity term |pprime_w * v_w|

c--- Preparation: Pressure and normal-velocity mapping

         ! [IMPORTANT] First map pressure to the GLL space
         call mappr(buffer,pr,pwvw,velV) ! The last two tensor is just for grabage
         ! Projection of wall actuation
         call copy(wrk_buff(1,1,1,1),ACTIONS(1,1,1,1),ntot) ! get zero-mean wall actuation
         call col3(vnorm,body_sin,wrk_buff,ntot) ! Project the wall-normal component -> vnorm

c--- Below is the JFM formulation----------
         call copy(wrk_buff(1,1,1,1),vnorm(1,1,1,1),ntot) ! vnorm -> wrk_buff

         call col3(pwvw,buffer,wrk_buff,ntot) ! pw * vw  -> pwvw

         ! Filter the negative values
         do 100 il=1,ntot
            if (pwvw(il,1,1,1) < 0.0) pwvw(il,1,1,1) = 0.0
100      continue
        
        !! YW: trail one is to take the correlation first and then do the averaging, 
        !! but it seems to be more noisy than doing the average first.
        ! pwvw = abs(pwvw)
        call copy(buffer(1,1,1,1),pwvw(1,1,1,1),ntot)
        if (rwd_zavg) then
            call z_averaging(buffer,avgVZ)
            call z_avg_reshape(buffer,avgVZ)
        endif
        if (rwd_xavg) then
            call x_averaging(buffer,avgVX)
            call x_avg_reshape(buffer,avgVX)
        endif
        ! Make copy
        call copy(pwvw(1,1,1,1),buffer(1,1,1,1),ntot)
        pwvw = abs(pwvw)
        
        

c------- Step 3: Compute kinetic energy term 0.5*|v^3_w|
         call copy(v3(1,1,1,1),vnorm(1,1,1,1),ntot) ! v3 <- vnorm

         ! Filter the negative values
         do 101 il=1,ntot
            if (v3(il,1,1,1) < 0.0) v3(il,1,1,1) = 0.0
101      continue

         ! Compute the kinetic energy term
         v3 = 0.5 * abs(v3**3)
         
         ! Take average
         call copy(buffer(1,1,1,1),v3(1,1,1,1),ntot)
         if (rwd_zavg) then
            call z_averaging(buffer,avgVZ)
            call z_avg_reshape(buffer,avgVZ)
         endif
         if (rwd_xavg) then
            call x_averaging(buffer,avgVX)
            call x_avg_reshape(buffer,avgVX)
         endif
         call copy(v3(1,1,1,1),buffer(1,1,1,1),ntot)

c------- Step 4: Assemble reward at each agent location
         if (i_evolv.eq.1) call rzero(rwd_agt,TOTCTRL)

         do il=1,NUMCTRL
            ie=info_agt(1,il)
            ie=gllel(ie)
            iface=info_agt(2,il)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)

            dudy_i = abs(dUdy(ix,iy,iz,ie))
            ! tau_w = mu * du/dy ; param(2)(=denu) is the DYNAMIC viscosity mu
            ! (Nek: param(2)=mu, param(1)=rho, kin.visc nu=param(2)/param(1)).
            ! Do NOT multiply by rho here -- that double-counts density.
            tau_w  = denu * dudy_i
            pwvw_i = pwvw(ix,iy,iz,ie)
            v3_i   = v3(ix,iy,iz,ie)

            if (i_evolv.eq.1) then
               rwd_tau(il) = tau_w
               rwd_pw(il)  = pwvw_i
               rwd_v3(il)  = v3_i
            else
               rwd_tau(il) = (rwd_tau(il)*(i_evolv-1) + tau_w)
     $                     / i_evolv
               rwd_pw(il)  = (rwd_pw(il)*(i_evolv-1) + pwvw_i)
     $                     / i_evolv
               rwd_v3(il)  = (rwd_v3(il)*(i_evolv-1) + v3_i)
     $                     / i_evolv
            endif
         enddo

! #ifdef GAINMONITOR
!         if (i_evolv.eq.n_drl) then
!                 call write_reward_monitor(i_evolv)
!         endif
! #endif

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

      end subroutine compute_netGain_posOnly





c------------------------------------------------------------------
c Calculate the dUidxj from OLD Statistics 
      subroutine comp_derivat(duidxj,u,v,w,ur,us,ut,vr,vs,vt,wr,ws,wt)
      include 'SIZE'
      include 'TOTAL'

      integer e

      real duidxj(lx1*ly1*lz1,lelt,3*ldim)    ! 9 terms
      real u  (lx1*ly1*lz1,lelt)
      real v  (lx1*ly1*lz1,lelt)
      real w  (lx1*ly1*lz1,lelt)
      real ur (1) , us (1) , ut (1)
      real vr (1) , vs (1) , vt (1)
      real wr (1) , ws (1) , wt (1)
c
c      common /dudxyj/ jacmi(lx1*ly1*lz1,lelt)
c      real jacmi
c
      n    = nx1-1                          ! Polynomial degree
      nxyz = nx1*ny1*nz1

      do e=1,nelv
         call local_grad3(ur,us,ut,u,N,e,dxm1,dxtm1)
         call local_grad3(vr,vs,vt,v,N,e,dxm1,dxtm1)
         call local_grad3(wr,ws,wt,w,N,e,dxm1,dxtm1)

!     Derivative tensor computed by using the inverse of 
!     the Jacobian array jacmi
      do k=1,nxyz
        ! dudx
         duidxj(k,e,1) = jacmi(k,e)*(ur(k)*rxm1(k,1,1,e)+
     $        us(k)*sxm1(k,1,1,e)+
     $        ut(k)*txm1(k,1,1,e))
        ! dvdy
         duidxj(k,e,2) = jacmi(k,e)*(vr(k)*rym1(k,1,1,e)+
     $        vs(k)*sym1(k,1,1,e)+
     $        vt(k)*tym1(k,1,1,e))
        ! dwdz
         duidxj(k,e,3) = jacmi(k,e)*(wr(k)*rzm1(k,1,1,e)+
     $        ws(k)*szm1(k,1,1,e)+
     $        wt(k)*tzm1(k,1,1,e))
        !dudy
         duidxj(k,e,4) = jacmi(k,e)*(ur(k)*rym1(k,1,1,e)+
     $        us(k)*sym1(k,1,1,e)+
     $        ut(k)*tym1(k,1,1,e))
        !dvdz
         duidxj(k,e,5) = jacmi(k,e)*(vr(k)*rzm1(k,1,1,e)+
     $        vs(k)*szm1(k,1,1,e)+
     $        vt(k)*tzm1(k,1,1,e))
        ! dwdx
         duidxj(k,e,6) = jacmi(k,e)*(wr(k)*rxm1(k,1,1,e)+
     $        ws(k)*sxm1(k,1,1,e)+
     $        wt(k)*txm1(k,1,1,e))
        ! dudz
         duidxj(k,e,7) = jacmi(k,e)*(ur(k)*rzm1(k,1,1,e)+
     $        us(k)*szm1(k,1,1,e)+
     $        ut(k)*tzm1(k,1,1,e))
        ! dvdx
         duidxj(k,e,8) = jacmi(k,e)*(vr(k)*rxm1(k,1,1,e)+
     $        vs(k)*sxm1(k,1,1,e)+
     $        vt(k)*txm1(k,1,1,e))
        ! dwdy
         duidxj(k,e,9) = jacmi(k,e)*(wr(k)*rym1(k,1,1,e)+
     $        ws(k)*sym1(k,1,1,e)+
     $        wt(k)*tym1(k,1,1,e))
      enddo
      enddo

      return
      end

c------------------------------------------------------------------




c------------------------------------------------------------------
        subroutine switch_BC_2Wall
c Change Wall-BC According to the indicies obtained before 
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
        
        integer ie,iface,ix,iy,iz ! Iteration
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
        if (NUMCTRL.ne.0) then 
            do idx=1,NUMCTRL
            ie=info_agt(1,idx)         ! Local indicies
            iface=info_agt(2,idx)      ! Number of face
            CBC(iface,ie,1)='W  '
            enddo
        endif
        if (NID.eq.0) print*,"[DRL] Switch B.C to WALL FOR REWARD"
        
        end subroutine switch_BC_2Wall
c------------------------------------------------------------------

c------------------------------------------------------------------
        subroutine switch_BC_2Drichlet
c Change Wall-BC According to the indicies obtained before 
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
        
        integer ie,iface,ix,iy,iz ! Iteration
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
        if (NUMCTRL.ne.0) then 
            do idx=1,NUMCTRL
            ie=info_agt(1,idx)         ! Local indicies
            iface=info_agt(2,idx)      ! Number of face
            CBC(iface,ie,1)='v  '
            enddo
        endif
        if (NID.eq.0) print*,"[DRL] Switch B.C to Dirichlet FOR CTRL"
        
        end subroutine switch_BC_2Drichlet
c------------------------------------------------------------------

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
      real wrk(3), tmp3(3)
      logical fexist
      integer, parameter :: iunit = 52002
      integer, parameter :: MAX_MON_LINES = 10000
      integer, save      :: ifile = 0
      integer :: nlines, ios
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
      if (NUMCTRL.gt.0) then
         do il = 1, NUMCTRL
            wrk(1) = wrk(1) + rwd_tau(il)
            wrk(2) = wrk(2) + rwd_pw(il)
            wrk(3) = wrk(3) + rwd_v3(il)
         enddo
      endif
      call gop(wrk, tmp3, '+  ', 3)
      if (nc.gt.0) then
         wrk(1) = wrk(1) / nc
         wrk(2) = wrk(2) / nc
         wrk(3) = wrk(3) / nc
      endif

      if (NID.eq.0) then
c        Build current filename
        write(fname,'(A,I5.5,A)') 'reward_monitor', ifile, '.dat'

         inquire(file=trim(fname), exist=fexist)
         if (fexist) then
c           Count existing data lines (skip header) before appending
            nlines = 0
            open(iunit, file=trim(fname), status='old')
            read(iunit,'(A)',iostat=ios) cbuf
            do
               read(iunit,'(A)',iostat=ios) cbuf
               if (ios.ne.0) exit
               nlines = nlines + 1
            end do
            close(iunit)
c           File full: roll over to a new numbered file
            if (nlines.ge.MAX_MON_LINES) then
               ifile = ifile + 1
               write(fname,'(A,I5.5,A)')
     $            'reward_monitor', ifile, '.dat'
               open(iunit, file=trim(fname), status='new')
               write(iunit,'(A)')
     $            '# time          i_evolv'//
     $            '  rwd_tau         rwd_pw          rwd_v3'
            else
               open(iunit, file=trim(fname), position='append')
            end if
         else
            open(iunit, file=trim(fname), status='new')
            write(iunit,'(A)')
     $         '# time          i_evolv'//
     $         '  rwd_tau         rwd_pw          rwd_v3'
         endif
         write(iunit,'(E16.8,1X,I6,3(1X,E16.8))')
     $      time, i_evolv, wrk(1), wrk(2), wrk(3)
         close(iunit)
        print *, "[MONITOR] RECORD",time,i_evolv,wrk(1),wrk(2),wrk(3)
      endif

      end subroutine write_reward_monitor

