c-----------------------------------------------------------------------
      subroutine compute_netGain_NES(i_evolv)
c Bewley, Moin & Temam (2001), JFM 447, p.218:
c Default recorder: Bewley costs; both definitions always saved in STS.
c Nek pressure is mechanical pressure; phi includes applied background
c actuation and ACTIONS, with the same override order as userbc.
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
      real ns_dx(lx1,ly1,lz1,lelt),ns_dy(lx1,ly1,lz1,lelt)
      real ns_dz(lx1,ly1,lz1,lelt)
      real ns_pw(lx1,ly1,lz1,lelt),ns_v3(lx1,ly1,lz1,lelt)
      real ns_tmp(lx1,ly1,lz1,lelt)
      real ns_avgz(lx1,ly1,xnel,ynel)
      real ns_avgx(lx1,ly1,ynel,znel)
      real ns_mu,ns_tau,ns_dt,ns_elapsed,ns_frac
c Static scratch avoids putting the production-size fields on the stack.
      save ns_dx,ns_dy,ns_dz,ns_pw,ns_v3,ns_tmp
      save ns_avgz,ns_avgx,ns_elapsed
      data ns_elapsed /0.0/

      ntot=lx1*ly1*lz1*nelv
      ns_mu=abs(param(2))
      ns_dt=abs(dt)
      if (ns_dt.le.0.0) then
         if (nid.eq.0) write(6,*) '[NES70] nonpositive timestep'
         call exitt0
         return
      endif
      if (i_evolv.eq.1) ns_elapsed=0.0
      ns_elapsed=ns_elapsed+ns_dt
      ns_frac=ns_dt/ns_elapsed

c Preserve the parent's shear computation for its existing baseline.
      call gradm1(ns_dx,ns_dy,ns_dz,vx)
      if (rwd_zavg) then
         call z_averaging(ns_dx,ns_avgz)
         call z_avg_reshape(ns_dx,ns_avgz)
         call z_averaging(ns_dy,ns_avgz)
         call z_avg_reshape(ns_dy,ns_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(ns_dx,ns_avgx)
         call x_avg_reshape(ns_dx,ns_avgx)
         call x_averaging(ns_dy,ns_avgx)
         call x_avg_reshape(ns_dy,ns_avgx)
         ns_dy=-body_cos*body_sin*ns_dx
     $         -body_cos*body_cos*ns_dx
     $         +body_sin*body_sin*ns_dy
     $         +body_cos*body_sin*ns_dy
      endif

      call nes_reward_fields(ns_pw,ns_v3)

c Average NONNEGATIVE local costs, retaining the chordwise profile.
      call copy(ns_tmp,ns_pw,ntot)
      if (rwd_zavg) then
         call z_averaging(ns_tmp,ns_avgz)
         call z_avg_reshape(ns_tmp,ns_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(ns_tmp,ns_avgx)
         call x_avg_reshape(ns_tmp,ns_avgx)
      endif
      call copy(ns_pw,ns_tmp,ntot)
      call copy(ns_tmp,ns_v3,ntot)
      if (rwd_zavg) then
         call z_averaging(ns_tmp,ns_avgz)
         call z_avg_reshape(ns_tmp,ns_avgz)
      endif
      if (rwd_xavg) then
         call x_averaging(ns_tmp,ns_avgx)
         call x_avg_reshape(ns_tmp,ns_avgx)
      endif
      call copy(ns_v3,ns_tmp,ntot)

      do il=1,numctrl
         ie=gllel(info_agt(1,il))
         ix=info_agt(3,il)
         iy=info_agt(4,il)
         iz=info_agt(5,il)
         ns_tau=ns_mu*abs(ns_dy(ix,iy,iz,ie))
         if (i_evolv.eq.1) then
            rwd_tau(il)=ns_tau
            rwd_pw(il)=ns_pw(ix,iy,iz,ie)
            rwd_v3(il)=ns_v3(ix,iy,iz,ie)
         else
            rwd_tau(il)=rwd_tau(il)
     $                 +ns_frac*(ns_tau-rwd_tau(il))
            rwd_pw(il)=rwd_pw(il)
     $                +ns_frac*(ns_pw(ix,iy,iz,ie)-rwd_pw(il))
            rwd_v3(il)=rwd_v3(il)
     $                +ns_frac*(ns_v3(ix,iy,iz,ie)-rwd_v3(il))
         endif
      enddo
      return
      end


c-----------------------------------------------------------------------
      subroutine nes_reward_fields(costp,costk)
c The existing three-slot recorder carries one definition. STS has both.
      implicit none
      include 'SIZE'
      include 'NES'
      real costp(lx1,ly1,lz1,lelt),costk(lx1,ly1,lz1,lelt)
      real hp(lx1,ly1,lz1,lelt),hk(lx1,ly1,lz1,lelt)
      save hp,hk
      call nes_cost_fields(costp,costk,hp,hk)
      if (nes_reward_model.eq.2) then
         call copy(costp,hp,lx1*ly1*lz1*nelv)
         call copy(costk,hk,lx1*ly1*lz1*nelv)
      endif
      return
      end

c-----------------------------------------------------------------------
      subroutine nes_cost_fields(bp,bk,hp,hk)
c Four instantaneous costs, formed BEFORE spatial or time averaging.
c Mechanical pressure; total prescribed normal velocity, positive blowing.
c Reuse bp for the local span mean, and hk for total wall velocity.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'SOLN'
      real bp(lx1,ly1,lz1,lelt),bk(lx1,ly1,lz1,lelt)
      real hp(lx1,ly1,lz1,lelt),hk(lx1,ly1,lz1,lelt)
      real pwall(lx1,ly1,lz1,lelt),phi,pwork,kwork
      integer ix,iy,iz,ie
      save pwall
      call mappr(pwall,pr,bp,bk)
      call nes_spanwise_pressure(pwall,bp)
      call nes_wall_velocity(hk)
      do ie=1,nelv
      do iz=1,lz1
      do iy=1,ly1
      do ix=1,lx1
         phi=hk(ix,iy,iz,ie)
         pwork=(pwall(ix,iy,iz,ie)-bp(ix,iy,iz,ie))*phi
         kwork=0.5*param(1)*phi**3
         bp(ix,iy,iz,ie)=abs(pwork)
         bk(ix,iy,iz,ie)=abs(kwork)
         hp(ix,iy,iz,ie)=max(pwork,0.0)
         hk(ix,iy,iz,ie)=max(kwork,0.0)
      enddo
      enddo
      enddo
      enddo
      return
      end

c-----------------------------------------------------------------------
      subroutine nes_spanwise_pressure(pwall,pmean)
c Independent instantaneous mean at each cross-section GLL location.
c Only the homogeneous span is averaged, using the uniform-span mesh.
      implicit none
      include 'SIZE'
      real pwall(lx1,ly1,lz1,lelt),pmean(lx1,ly1,lz1,lelt)
      real pavgz(lx1,ly1,xnel,ynel)
      save pavgz
      call z_averaging(pwall,pavgz)
      call z_avg_reshape(pmean,pavgz)
      return
      end

c-----------------------------------------------------------------------
      subroutine nes_wall_velocity(phi)
c Mirror userbc precedence: separate blowing patch first, DRL overrides.
c The US offset is added only to active DRL nodes on the suction side.
c Never assign background blowing to volume points or wall-only faces.
      implicit none
      include 'SIZE'
      include 'INPUT'
      include 'GEOM'
      include 'DRL'
      include 'NES'
      real phi(lx1,ly1,lz1,lelt),x,y
      integer ie,iface,ix,iy,iz,kx1,kx2,ky1,ky2,kz1,kz2
      call rzero(phi,lx1*ly1*lz1*lelt)
      if (nes_blow.ne.0.0) then
         do ie=1,nelv
         do iface=1,2*ldim
            if (cbc(iface,ie,1).eq.'v  ') then
               call facind(kx1,kx2,ky1,ky2,kz1,kz2,
     $                     lx1,ly1,lz1,iface)
               do iz=kz1,kz2
               do iy=ky1,ky2
               do ix=kx1,kx2
                  x=xm1(ix,iy,iz,ie)
                  y=ym1(ix,iy,iz,ie)
                  if (x.ge.nes_blow_xmin.and.x.le.nes_blow_xmax
     $                .and.y.ge.0.0.and.y.le.0.6) then
                     phi(ix,iy,iz,ie)=nes_blow
                  endif
               enddo
               enddo
               enddo
            endif
         enddo
         enddo
      endif
      do ie=1,nelv
      do iz=1,lz1
      do iy=1,ly1
      do ix=1,lx1
         if (msk_act(ix,iy,iz,ie).ne.0) then
            phi(ix,iy,iz,ie)=actions(ix,iy,iz,ie)
            if (ym1(ix,iy,iz,ie).gt.0.0) then
               phi(ix,iy,iz,ie)=phi(ix,iy,iz,ie)+nes_suction
            endif
         endif
      enddo
      enddo
      enddo
      enddo
      return
      end
