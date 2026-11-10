c-----------------------------------------------------------------------
      subroutine compute_netGain_Bewley(i_evolv)
c Bewley actuator costs: take local absolute values BEFORE averaging.
c Pressure is mechanical pressure; param(2) is dynamic viscosity in Nek.
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
      real cp(lx1,ly1,lz1,lelt),ck(lx1,ly1,lz1,lelt)
      real elapsed,weight,stepdt,tau
      save dx,dy,dz,work,cp,ck,igs_z,igs_x,elapsed
      data igs_z,igs_x /0,0/
      data elapsed /0.0/
      ntot=lx1*ly1*lz1*nelv
      stepdt=abs(dt)
      if (stepdt.le.0.0) then
         if (nid.eq.0) write(6,*) '[BEWLEY] nonpositive timestep'
         call exitt
         return
      endif
      if (i_evolv.eq.1) elapsed=0.0
      elapsed=elapsed+stepdt
      weight=stepdt/elapsed

      if (igs_z.eq.0.and.igs_x.eq.0) then
         call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3)
         call gtpp_gs_setup(igs_x,xnel,ynel,znel,1)
      endif
      call gradm1(dx,dy,dz,vx)
      call bewley_cost_fields(cp,ck)
      
      if (rwd_zavg) then
         call planar_avg(work,dy,igs_z)
         call copy(dy,work,ntot)
         call planar_avg(work,cp,igs_z)
         call copy(cp,work,ntot)
         call planar_avg(work,ck,igs_z)
         call copy(ck,work,ntot)
      endif
      if (rwd_xavg) then
         call planar_avg(work,dy,igs_x)
         call copy(dy,work,ntot)
         call planar_avg(work,cp,igs_x)
         call copy(cp,work,ntot)
         call planar_avg(work,ck,igs_x)
         call copy(ck,work,ntot)
      endif

      if (i_evolv.eq.1) call rzero(rwd_agt,totctrl)
      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         tau=abs(param(2))*dy(ix,iy,iz,ie)
         if (i_evolv.eq.1) then
            rwd_tau(il)=tau
            rwd_pw(il)=cp(ix,iy,iz,ie)
            rwd_v3(il)=ck(ix,iy,iz,ie)
         else
            rwd_tau(il)=rwd_tau(il)+weight*(tau-rwd_tau(il))
            rwd_pw(il)=rwd_pw(il)
     $                 +weight*(cp(ix,iy,iz,ie)-rwd_pw(il))
            rwd_v3(il)=rwd_v3(il)
     $                 +weight*(ck(ix,iy,iz,ie)-rwd_v3(il))
         endif
      enddo
#ifdef GAINMONITOR
      if (i_evolv.eq.nint(uparam(1)))
     $     call write_reward_monitor(i_evolv)
#endif
      return
      end

c-----------------------------------------------------------------------
      subroutine bewley_cost_fields(costp,costk)
c Native-order instantaneous fields shared by DRL and statistics.
c Match ACTUATE_JET: phi = ACTIONS * msk_act. No wing suction offset.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'DRL'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real pwall(lx1,ly1,lz1,lelt),pbar,phi
      integer ie,ix,iy,iz
      save pwall
      call mappr(pwall,pr,costp,costk)
      call bewley_wall_pressure(pwall,pbar)
      do ie=1,nelv
      do iz=1,lz1
      do iy=1,ly1
      do ix=1,lx1
         phi=actions(ix,iy,iz,ie)*msk_act(ix,iy,iz,ie)
         costp(ix,iy,iz,ie)=abs((pwall(ix,iy,iz,ie)-pbar)*phi)
         costk(ix,iy,iz,ie)=0.5*param(1)*abs(phi)**3
      enddo
      enddo
      enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine bewley_wall_pressure(pwall,pbar)
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
         if (nid.eq.0) write(6,*) '[BEWLEY] no control-wall area'
         call exitt
         return
      endif
      pbar=sums(1)/sums(2)
      return
      end
