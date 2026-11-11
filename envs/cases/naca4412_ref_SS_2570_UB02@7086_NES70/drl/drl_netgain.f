c------------------------------------------------------------------
      subroutine compute_netGain_posOnly(i_evolv)
c Net-energy reward components for the local embedded-policy recorder.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'NEKUSE'
      include 'TSTEP'
      include 'USERPAR'
      include 'DRL'
      include 'DRL_RWD_NETGAIN'

      real avgVZ(LX1,LY1,xnel,ynel),avgVX(LX1,LY1,ynel,znel)
      real denu, dudy_i, tau_w, pwvw_i, v3_i
      integer i_evolv, n_drl
      integer ie,iface,ix,iy,iz
      integer ntot, il, ilx, ily
      character*4 str

      denu  = abs(real(param(2)))
      n_drl = int(PARAM(89))
      ntot  = LX1*LY1*LZ1*LELT

c----- wall-shear stress
      call copy(devU1(1,1,1,1),vx(1,1,1,1),ntot)
      call gradm1(duidxj(1,1,1,1,1),
     $            duidxj(1,1,1,1,2),
     $            duidxj(1,1,1,1,3),devU1(1,1,1,1))
      call copy(dUdx(1,1,1,1),duidxj(1,1,1,1,1),ntot)
      call copy(dUdy(1,1,1,1),duidxj(1,1,1,1,2),ntot)

      if (rwd_zavg) then
         call z_averaging(dUdx,avgVZ)
         call z_avg_reshape(dUdx,avgVZ)
         call z_averaging(dUdy,avgVZ)
         call z_avg_reshape(dUdy,avgVZ)
      endif
      if (rwd_xavg) then
         call x_averaging(dUdx,avgVX)
         call x_avg_reshape(dUdx,avgVX)
         call x_averaging(dUdy,avgVX)
         call x_avg_reshape(dUdy,avgVX)
         dUdy = -body_cos * body_sin * dUdx +
     $          -body_cos * body_cos * dUdx +
     $          +body_sin * body_sin * dUdy +
     $          +body_cos * body_sin * dUdy
      endif

c----- positive pressure-work and cubic velocity-work contributions
      call mappr(buffer,pr,pwvw,velV)
      call copy(wrk_buff(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      call col3(vnorm,body_sin,wrk_buff,ntot)
      call copy(wrk_buff(1,1,1,1),vnorm(1,1,1,1),ntot)
      call col3(pwvw,buffer,wrk_buff,ntot)
      do il=1,ntot
         if (pwvw(il,1,1,1).lt.0.0) pwvw(il,1,1,1) = 0.0
      enddo
      call copy(buffer(1,1,1,1),pwvw(1,1,1,1),ntot)
      if (rwd_zavg) then
         call z_averaging(buffer,avgVZ)
         call z_avg_reshape(buffer,avgVZ)
      endif
      if (rwd_xavg) then
         call x_averaging(buffer,avgVX)
         call x_avg_reshape(buffer,avgVX)
      endif
      call copy(pwvw(1,1,1,1),buffer(1,1,1,1),ntot)
      pwvw = abs(pwvw)

      call copy(v3(1,1,1,1),vnorm(1,1,1,1),ntot)
      do il=1,ntot
         if (v3(il,1,1,1).lt.0.0) v3(il,1,1,1) = 0.0
      enddo
      v3 = 0.5 * abs(v3**3)
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

c----- moving average over one policy-control interval
      do il=1,NUMCTRL
         ie    = gllel(info_agt(1,il))
         iface = info_agt(2,il)
         ix    = info_agt(3,il)
         iy    = info_agt(4,il)
         iz    = info_agt(5,il)
         dudy_i = abs(dUdy(ix,iy,iz,ie))
         tau_w  = denu*dudy_i
         pwvw_i  = pwvw(ix,iy,iz,ie)
         v3_i    = v3(ix,iy,iz,ie)
         if (i_evolv.eq.1) then
            rwd_tau(il) = tau_w
            rwd_pw(il)  = pwvw_i
            rwd_v3(il)  = v3_i
         else
            rwd_tau(il) = (rwd_tau(il)*(i_evolv-1)+tau_w)/i_evolv
            rwd_pw(il)  = (rwd_pw(il)*(i_evolv-1)+pwvw_i)/i_evolv
            rwd_v3(il)  = (rwd_v3(il)*(i_evolv-1)+v3_i)/i_evolv
         endif
      enddo

#ifdef GAINMONITOR
      if (i_evolv.eq.n_drl) call write_reward_monitor(i_evolv)
#endif

#ifdef YWDEBUG
      if (ISTEP.le.3 .and. NUMCTRL.ne.0) then
         write(str,"(i4.4)") NID
         open(51001,file="Reward.txt"//str)
         do ilx=1,NUMCTRL
            write(51001,*) proc(ilx),
     $       (pos_agt(ily,ilx),ily=1,NDIM),
     $       (info_agt(ily,ilx),ily=1,5),
     $       rwd_tau(ilx),rwd_pw(ilx),rwd_v3(ilx)
         enddo
         close(51001)
      endif
#endif

      return
      end
c------------------------------------------------------------------


c------------------------------------------------------------------
      subroutine write_reward_monitor(i_evolv)
c Append a rank-zero cross-check of the completed net-gain interval.
c------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'NEKUSE'
      include 'SOLN'
      include 'TSTEP'
      include 'PARALLEL'
      include 'DRL'

      integer i_evolv, iglsum, il, nc
      real wrk(3), tmp3(3)
      logical fexist
      integer, parameter :: iunit = 52002
      integer, parameter :: MAX_MON_LINES = 10000
      integer, save :: ifile = 0
      integer, save :: nrec = -1
      logical, save :: hdrdon = .false.
      integer nlines, ntot, ios
      character*200 cbuf
      character*40 fname

      nc     = iglsum(NUMCTRL,1)
      wrk(1) = 0.0
      wrk(2) = 0.0
      wrk(3) = 0.0
      do il=1,NUMCTRL
         wrk(1) = wrk(1) + rwd_tau(il)
         wrk(2) = wrk(2) + rwd_pw(il)
         wrk(3) = wrk(3) + rwd_v3(il)
      enddo
      call gop(wrk,tmp3,'+  ',3)
      if (nc.gt.0) then
         wrk(1) = wrk(1)/nc
         wrk(2) = wrk(2)/nc
         wrk(3) = wrk(3)/nc
      endif

      if (NID.ne.0) return
      if (nrec.lt.0) then
         do while (ifile.lt.99999 .and. nrec.lt.0)
            write(fname,'(A,I5.5,A)') 'reward_monitor',ifile,'.dat'
            inquire(file=trim(fname),exist=fexist)
            if (.not.fexist) then
               nrec   = 0
               hdrdon = .false.
            else
               nlines = 0
               ntot   = 0
               open(iunit,file=trim(fname),status='old',iostat=ios)
               if (ios.ne.0) then
                  ifile = ifile + 1
               else
                  do
                     read(iunit,'(A)',iostat=ios) cbuf
                     if (ios.ne.0) exit
                     ntot = ntot + 1
                     if (cbuf(1:1).ne.'#') nlines = nlines + 1
                  enddo
                  close(iunit)
                  if (nlines.lt.MAX_MON_LINES) then
                     nrec   = nlines
                     hdrdon = ntot.gt.0
                  else
                     ifile = ifile + 1
                  endif
               endif
            endif
         enddo
         if (nrec.lt.0) return
      endif
      if (nrec.ge.MAX_MON_LINES) then
         ifile  = ifile + 1
         nrec   = 0
         hdrdon = .false.
      endif
      write(fname,'(A,I5.5,A)') 'reward_monitor',ifile,'.dat'
      open(iunit,file=trim(fname),position='append',iostat=ios)
      if (ios.ne.0) then
         nrec = -1
         return
      endif
      if (.not.hdrdon) then
         write(iunit,'(A)') '# time          i_evolv'//
     $      '  rwd_tau         rwd_pw          rwd_v3'
         hdrdon = .true.
      endif
      write(iunit,'(E16.8,1X,I6,3(1X,E16.8))')
     $ time,i_evolv,wrk(1),wrk(2),wrk(3)
      close(iunit)
      nrec = nrec + 1

      return
      end
c------------------------------------------------------------------
