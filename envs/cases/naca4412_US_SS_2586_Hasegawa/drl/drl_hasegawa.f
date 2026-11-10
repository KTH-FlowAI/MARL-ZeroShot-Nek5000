c-----------------------------------------------------------------------
      subroutine compute_netGain_Hasegawa(i_evolv)
c Sonoda et al. (2023), JFM 960, A30, Eq. (4.1), adapted to the wing:
c local cost = max((p-pbar)*phi,0) + 0.5*rho*max(phi,0)**3.
c Nek pressure is mechanical pressure; phi includes uniform suction
c and ACTIONS, before projection onto Cartesian components in userbc.
c pbar is the instantaneous spanwise mean at each cross-section GLL point.
c Each term is clipped before averaging; cycle means are dt-weighted.
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
      real hg_dx(lx1,ly1,lz1,lelt),hg_dy(lx1,ly1,lz1,lelt)
      real hg_dz(lx1,ly1,lz1,lelt)
      real hg_pw(lx1,ly1,lz1,lelt),hg_v3(lx1,ly1,lz1,lelt)
      real hg_tmp(lx1,ly1,lz1,lelt)
      real hg_avgz(lx1,ly1,xnel,ynel)
      real hg_avgx(lx1,ly1,ynel,znel)
      real hg_mu,hg_tau,hg_dt,hg_elapsed,hg_frac
c Static scratch avoids putting the production-size fields on the stack.
      save hg_dx,hg_dy,hg_dz,hg_pw,hg_v3,hg_tmp
      save hg_avgz,hg_avgx,hg_elapsed
      data hg_elapsed /0.0/

      ntot=lx1*ly1*lz1*nelv
      hg_mu=abs(param(2))
      hg_dt=abs(dt)
      if (hg_dt.le.0.0) then
         if (nid.eq.0) write(6,*) '[HASEGAWA] nonpositive timestep'
         call exitt0
         return
      endif
      if (i_evolv.eq.1) hg_elapsed=0.0
      hg_elapsed=hg_elapsed+hg_dt
      hg_frac=hg_dt/hg_elapsed

c Preserve the parent's shear computation for its existing baseline.
      call gradm1(hg_dx,hg_dy,hg_dz,vx)
      if (rwd_zavg) then
         call z_averaging(hg_dx,hg_avgz)
         call z_avg_reshape(hg_dx,hg_avgz)
         call z_averaging(hg_dy,hg_avgz)
         call z_avg_reshape(hg_dy,hg_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(hg_dx,hg_avgx)
         call x_avg_reshape(hg_dx,hg_avgx)
         call x_averaging(hg_dy,hg_avgx)
         call x_avg_reshape(hg_dy,hg_avgx)
         hg_dy=-body_cos*body_sin*hg_dx
     $         -body_cos*body_cos*hg_dx
     $         +body_sin*body_sin*hg_dy
     $         +body_cos*body_sin*hg_dy
      endif

      call hasegawa_cost_fields(hg_pw,hg_v3)

c Average NONNEGATIVE local costs, retaining the chordwise profile.
      call copy(hg_tmp,hg_pw,ntot)
      if (rwd_zavg) then
         call z_averaging(hg_tmp,hg_avgz)
         call z_avg_reshape(hg_tmp,hg_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(hg_tmp,hg_avgx)
         call x_avg_reshape(hg_tmp,hg_avgx)
      endif
      call copy(hg_pw,hg_tmp,ntot)
      call copy(hg_tmp,hg_v3,ntot)
      if (rwd_zavg) then
         call z_averaging(hg_tmp,hg_avgz)
         call z_avg_reshape(hg_tmp,hg_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(hg_tmp,hg_avgx)
         call x_avg_reshape(hg_tmp,hg_avgx)
      endif
      call copy(hg_v3,hg_tmp,ntot)

      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         hg_tau=hg_mu*abs(hg_dy(ix,iy,iz,ie))
         if (i_evolv.eq.1) then
            rwd_tau(il)=hg_tau
            rwd_pw(il)=hg_pw(ix,iy,iz,ie)
            rwd_v3(il)=hg_v3(ix,iy,iz,ie)
         else
            rwd_tau(il)=rwd_tau(il)
     $                 +hg_frac*(hg_tau-rwd_tau(il))
            rwd_pw(il)=rwd_pw(il)
     $                +hg_frac*(hg_pw(ix,iy,iz,ie)-rwd_pw(il))
            rwd_v3(il)=rwd_v3(il)
     $                +hg_frac*(hg_v3(ix,iy,iz,ie)-rwd_v3(il))
         endif
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine hasegawa_cost_fields(costp,costk)
c Instantaneous native-order fields, shared by reward and statistics.
c Only actuated nodes receive the uniform-suction offset, matching userbc.
c All ranks must call this routine, including ranks with no control face.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'DRL'
      include 'HASEGAWA'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real pwall(lx1,ly1,lz1,lelt),phi
      integer ie,ix,iy,iz
      save pwall
c mappr's last two arguments are scratch, fully overwritten below.
      call mappr(pwall,pr,costp,costk)
c Reuse costp for the native-order mean, then overwrite point by point.
      call hasegawa_spanwise_pressure(pwall,costp)
      do ie=1,nelv
      do iz=1,lz1
      do iy=1,ly1
      do ix=1,lx1
         phi=0.0
         if (msk_act(ix,iy,iz,ie).ne.0) then
            phi=actions(ix,iy,iz,ie)+hg_suction
         endif
         costp(ix,iy,iz,ie)=
     $        max((pwall(ix,iy,iz,ie)-costp(ix,iy,iz,ie))*phi,0.0)
         costk(ix,iy,iz,ie)=0.5*param(1)*max(phi,0.0)**3
      enddo
      enddo
      enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine hasegawa_spanwise_pressure(pwall,pmean)
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
