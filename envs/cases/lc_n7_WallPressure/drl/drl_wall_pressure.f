c-----------------------------------------------------------------------
      subroutine compute_wall_pressure(i_evolv)
c [WALL_PRESSURE] tau=mu*dUdy; pressure RMS replaces BOTH power costs.
c Average squared fluctuations in space and time BEFORE taking sqrt.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'TSTEP'
      include 'DRL'
      integer i_evolv,ntot,igs_z,igs_x,il,ie,ix,iy,iz
      real dx(lx1,ly1,lz1,lelt),dy(lx1,ly1,lz1,lelt)
      real dz(lx1,ly1,lz1,lelt),work(lx1,ly1,lz1,lelt)
      real p2(lx1,ly1,lz1,lelt),scratch(lx1,ly1,lz1,lelt)
      real mean_p2(totctrl),elapsed,weight,stepdt,tau
      save dx,dy,dz,work,p2,scratch,mean_p2,igs_z,igs_x,elapsed
      data igs_z,igs_x /0,0/
      data elapsed /0.0/
      ntot=lx1*ly1*lz1*nelv
      stepdt=abs(dt)
      if (stepdt.le.0.0) then
         if (nid.eq.0) write(6,*) '[WALL_PRESSURE] invalid timestep'
         call exitt
         return
      endif
      if (i_evolv.eq.1) then
         elapsed=0.0
         call rzero(mean_p2,totctrl)
         call rzero(rwd_agt,totctrl)
         call rzero(rwd_tau,totctrl)
         call rzero(rwd_pw,totctrl)
         call rzero(rwd_v3,totctrl)
      endif
      elapsed=elapsed+stepdt
      weight=stepdt/elapsed
      if (igs_z.eq.0.and.igs_x.eq.0) then
         call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3)
         call gtpp_gs_setup(igs_x,xnel,ynel,znel,1)
      endif
      call gradm1(dx,dy,dz,vx)
      call wall_pressure_squared(p2,scratch,work)
      if (rwd_zavg) then
         call planar_avg(work,dy,igs_z)
         call copy(dy,work,ntot)
         call planar_avg(work,p2,igs_z)
         call copy(p2,work,ntot)
      endif
      if (rwd_xavg) then
         call planar_avg(work,dy,igs_x)
         call copy(dy,work,ntot)
         call planar_avg(work,p2,igs_x)
         call copy(p2,work,ntot)
      endif
      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         tau=abs(param(2))*dy(ix,iy,iz,ie)
         rwd_tau(il)=rwd_tau(il)+weight*(tau-rwd_tau(il))
         mean_p2(il)=mean_p2(il)
     $             +weight*(p2(ix,iy,iz,ie)-mean_p2(il))
c [WALL_PRESSURE] Expose RMS in the old pw slot; v3 stays zero.
         rwd_pw(il)=sqrt(max(0.0,mean_p2(il)))
      enddo
#ifdef GAINMONITOR
      if (i_evolv.eq.nint(uparam(1)))
     $     call write_reward_monitor(i_evolv)
#endif
      return
      end

c-----------------------------------------------------------------------
      subroutine wall_pressure_squared(p2,scratch,work)
c [WALL_PRESSURE] Use mechanical pressure, independent of jet velocity.
c Remove the instantaneous wall-area mean to cancel pressure gauge shifts.
c Fill entire registered faces, including GLL nodes omitted by gll_unique.
c No actuation mask: baseline pressure fluctuations exist even at rest.
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'SOLN'
      include 'DRL'
      real p2(lx1,ly1,lz1,lelt),scratch(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),pwall(lx1,ly1,lz1,lelt)
      real pbar
      integer il,ie,ix,iy,iz,iface,kx1,kx2,ky1,ky2,kz1,kz2
      logical seen(2*ldim,lelt)
      save pwall
      call mappr(pwall,pr,scratch,work)
      call wall_pressure_mean(pwall,pbar)
      call rzero(p2,lx1*ly1*lz1*lelt)
      seen=.false.
      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         iface=info_agt(2,il)
         if (.not.seen(iface,ie)) then
            call facind(kx1,kx2,ky1,ky2,kz1,kz2,
     $                  lx1,ly1,lz1,iface)
            do iz=kz1,kz2
            do iy=ky1,ky2
            do ix=kx1,kx2
               p2(ix,iy,iz,ie)=(pwall(ix,iy,iz,ie)-pbar)**2
            enddo
            enddo
            enddo
            seen(iface,ie)=.true.
         endif
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine wall_pressure_mean(pwall,pbar)
c Instantaneous area mean over registered lower-wall control faces.
c Deduplicate faces locally; all ranks join the global reduction.
      implicit none
      include 'SIZE'
      include 'PARALLEL'
      include 'DRL'
      real pwall(lx1,ly1,lz1,lelt),pbar
      real sums(2),work(2),pint,awall
      logical seen(2*ldim,lelt)
      integer il,ie,iface
      seen=.false.
      sums(1)=0.0
      sums(2)=0.0
      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         iface=info_agt(2,il)
         if (.not.seen(iface,ie)) then
            call surface_int(pint,awall,pwall,ie,iface)
            sums(1)=sums(1)+pint
            sums(2)=sums(2)+awall
            seen(iface,ie)=.true.
         endif
      enddo
      call gop(sums,work,'+  ',2)
      if (sums(2).le.0.0) then
         if (nid.eq.0) write(6,*) '[WALL_PRESSURE] no control-wall area'
         call exitt
         return
      endif
      pbar=sums(1)/sums(2)
      return
      end
