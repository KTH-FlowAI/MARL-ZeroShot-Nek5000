c-----------------------------------------------------------------------
      subroutine compute_netGain_Bewley(i_evolv)
c Bewley, Moin & Temam (2001), JFM 447, p.218:
c local actuator cost = |(p-pbar)*phi| + 0.5*rho*|phi|**3.
c Nek pressure is mechanical pressure; phi includes uniform suction
c and ACTIONS, before projection onto Cartesian components in userbc.
c pbar is the instantaneous spanwise mean at each cross-section GLL point.
c Absolute values precede spatial averaging; cycle means are dt-weighted.
c The parent's wall-shear proxy and reward/recorder layout are retained.
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'TSTEP'
      include 'DRL'

      integer i_evolv,ntot,il,ie,ix,iy,iz
      real bw_dx(lx1,ly1,lz1,lelt),bw_dy(lx1,ly1,lz1,lelt)
      real bw_dz(lx1,ly1,lz1,lelt)
      real bw_pw(lx1,ly1,lz1,lelt),bw_v3(lx1,ly1,lz1,lelt)
      real bw_tmp(lx1,ly1,lz1,lelt)
      real bw_avgz(lx1,ly1,xnel,ynel)
      real bw_avgx(lx1,ly1,ynel,znel)
      real bw_mu,bw_tau,bw_dt,bw_elapsed,bw_frac
c Static scratch avoids putting the production-size fields on the stack.
      save bw_dx,bw_dy,bw_dz,bw_pw,bw_v3,bw_tmp
      save bw_avgz,bw_avgx,bw_elapsed
      data bw_elapsed /0.0/

      ntot=lx1*ly1*lz1*nelv
      bw_mu=abs(param(2))
      bw_dt=abs(dt)
      if (bw_dt.le.0.0) then
         if (nid.eq.0) write(6,*) '[BEWLEY] nonpositive timestep'
         call exitt0
         return
      endif
      if (i_evolv.eq.1) bw_elapsed=0.0
      bw_elapsed=bw_elapsed+bw_dt
      bw_frac=bw_dt/bw_elapsed

c Preserve the parent's shear computation for its existing baseline.
      call gradm1(bw_dx,bw_dy,bw_dz,vx)
      if (rwd_zavg) then
         call z_averaging(bw_dx,bw_avgz)
         call z_avg_reshape(bw_dx,bw_avgz)
         call z_averaging(bw_dy,bw_avgz)
         call z_avg_reshape(bw_dy,bw_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(bw_dx,bw_avgx)
         call x_avg_reshape(bw_dx,bw_avgx)
         call x_averaging(bw_dy,bw_avgx)
         call x_avg_reshape(bw_dy,bw_avgx)
         bw_dy=-body_cos*body_sin*bw_dx
     $         -body_cos*body_cos*bw_dx
     $         +body_sin*body_sin*bw_dy
     $         +body_cos*body_sin*bw_dy
      endif

      call bewley_cost_fields(bw_pw,bw_v3)

c Average NONNEGATIVE local costs, retaining the chordwise profile.
      call copy(bw_tmp,bw_pw,ntot)
      if (rwd_zavg) then
         call z_averaging(bw_tmp,bw_avgz)
         call z_avg_reshape(bw_tmp,bw_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(bw_tmp,bw_avgx)
         call x_avg_reshape(bw_tmp,bw_avgx)
      endif
      call copy(bw_pw,bw_tmp,ntot)
      call copy(bw_tmp,bw_v3,ntot)
      if (rwd_zavg) then
         call z_averaging(bw_tmp,bw_avgz)
         call z_avg_reshape(bw_tmp,bw_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(bw_tmp,bw_avgx)
         call x_avg_reshape(bw_tmp,bw_avgx)
      endif
      call copy(bw_v3,bw_tmp,ntot)

      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         bw_tau=bw_mu*abs(bw_dy(ix,iy,iz,ie))
         if (i_evolv.eq.1) then
            rwd_tau(il)=bw_tau
            rwd_pw(il)=bw_pw(ix,iy,iz,ie)
            rwd_v3(il)=bw_v3(ix,iy,iz,ie)
         else
            rwd_tau(il)=rwd_tau(il)
     $                 +bw_frac*(bw_tau-rwd_tau(il))
            rwd_pw(il)=rwd_pw(il)
     $                +bw_frac*(bw_pw(ix,iy,iz,ie)-rwd_pw(il))
            rwd_v3(il)=rwd_v3(il)
     $                +bw_frac*(bw_v3(ix,iy,iz,ie)-rwd_v3(il))
         endif
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine bewley_cost_fields(costp,costk)
c Instantaneous native-order fields, shared by reward and statistics.
c Only actuated nodes receive the uniform-suction offset, matching userbc.
c All ranks must call this routine, including ranks with no control face.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'DRL'
      include 'BEWLEY'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real pwall(lx1,ly1,lz1,lelt),phi
      integer ie,ix,iy,iz
      save pwall
c mappr's last two arguments are scratch, fully overwritten below.
      call mappr(pwall,pr,costp,costk)
c Reuse costp for the native-order mean, then overwrite point by point.
      call bewley_spanwise_pressure(pwall,costp)
      do ie=1,nelv
      do iz=1,lz1
      do iy=1,ly1
      do ix=1,lx1
         phi=0.0
         if (msk_act(ix,iy,iz,ie).ne.0) then
            phi=actions(ix,iy,iz,ie)+bw_suction
         endif
         costp(ix,iy,iz,ie)=
     $        abs((pwall(ix,iy,iz,ie)-costp(ix,iy,iz,ie))*phi)
         costk(ix,iy,iz,ie)=0.5*param(1)*abs(phi)**3
      enddo
      enddo
      enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine bewley_spanwise_pressure(pwall,pmean)
c Separate instantaneous reference at every cross-section GLL location.
c Only the homogeneous span is averaged; chord locations stay separate.
c The existing z operators use GLL weights on the uniform-span wing mesh
c and reduce matching cross-section points across all MPI ranks.
c-----------------------------------------------------------------------
      implicit none
      include 'SIZE'
      real pwall(lx1,ly1,lz1,lelt),pmean(lx1,ly1,lz1,lelt)
      real pavgz(lx1,ly1,xnel,ynel)
      save pavgz
      call z_averaging(pwall,pavgz)
      call z_avg_reshape(pmean,pavgz)
      return
      end
