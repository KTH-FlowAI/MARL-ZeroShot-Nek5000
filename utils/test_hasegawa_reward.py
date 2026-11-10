"""Check the wing adaptation of Sonoda et al. (2023), Eq. (4.1).

Run: python utils/test_hasegawa_reward.py
Compiles the actual cost, spanwise-reference and statistics routines on
1, 2 and 3 MPI ranks. Hand-calculated fixtures distinguish independent
pointwise clipping from absolute values, clipping the sum, and clipping
after averaging. Shares the solver stubs/build harness with the Bewley test.
"""

from test_bewley_reward import DRIVER as BEWLEY_DRIVER, ROOT, run_wing_checks


CASE = ROOT / "envs/cases/naca4412_US_SS_2586_Hasegawa"

DRIVER = """
      program test_hasegawa
      implicit none
      include 'mpif.h'
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'TSTEP'
      include 'DRL'
      include 'HASEGAWA'
      include 'STATS'
      include 'FIXTURE'
      include 'WZ'
      include 'ZPER'
      integer ierr,eg,ie,ix,iz,lnvar,ex,ez
      real ref(lx1,ly1,lz1,lelt),expected
      real cp(lx1,ly1,lz1,lelt),ck(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt)
      real point_p(2,2,2),point_k(2,2),stats_p(2,2)
      real reward_p(2),reward_k(2)
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,nid,ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,np,ierr)
      nio=nid
      if3d=.true.
      numctrl=0
      nelv=0
      gllel=0
      lglel=0
      nelz=2
      wzm1(1)=1.0
      wzm1(2)=3.0
      pr=999.0
      vx=0.0
      actions=0.0
      msk_act=0
      body_sin=0.6
      body_cos=0.8
      param=0.0
      param(1)=1.0
      param(2)=0.1
c Two cross sections, two span slabs; rank 2 is empty in the 3-rank run.
c Pressure P(s)+[-3,-1,1,3] with weights [1,3,1,3] has mean P(s)+0.5.
c Velocities at ix=1: [1,-1,1,-1]; at ix=2: [-2,2,-2,2].
      do eg=1,4
         ex=mod(eg-1,2)+1
         ez=(eg-1)/2+1
         if (np.eq.1.or.nid.eq.ez-1) then
            nelv=nelv+1
            ie=nelv
            gllel(eg)=ie
            lglel(ie)=eg
            do iz=1,2
            do ix=1,2
               pr(ix,1,iz,ie)=40*(ex-1)+4*(ix-1)
     $                            +4*(ez-1)+2*(iz-1)-3
               actions(ix,1,iz,ie)=ix*(-1.0)**(ix+iz)-hg_suction
               msk_act(ix,1,iz,ie)=1
               numctrl=numctrl+1
               info_agt(1,numctrl)=eg
               info_agt(2,numctrl)=1
               info_agt(3,numctrl)=ix
               info_agt(4,numctrl)=1
               info_agt(5,numctrl)=iz
            enddo
            enddo
         endif
      enddo
      call hasegawa_spanwise_pressure(pr,ref)
      do ie=1,nelv
         ex=mod(lglel(ie)-1,2)+1
         ez=(lglel(ie)-1)/2+1
         do iz=1,2
         do ix=1,2
            expected=40*(ex-1)+4*(ix-1)+0.5
            call assert_close(ref(ix,1,iz,ie),expected,
     $                        'separate pressure reference at each s')
            expected=40*(ex-1)+4*(ix-1)+4*(ez-1)+2*(iz-1)-3
            call assert_close(pr(ix,1,iz,ie),expected,
     $                        'pressure input preserved')
         enddo
         enddo
      enddo

c Golden values for independently clipped local terms, then averages.
c At ix=1 the mean signed pressure work is -0.75, but its positive
c part averages to 0.625. Averaging before clipping would give zero.
      point_p(:,:,1)=reshape((/0.0,7.0,1.5,0.0/),(/2,2/))
      point_p(:,:,2)=reshape((/0.5,0.0,0.0,5.0/),(/2,2/))
      point_k=reshape((/0.5,0.0,0.0,4.0/),(/2,2/))
      stats_p(:,1)=(/1.125,1.75/)
      stats_p(:,2)=(/0.125,3.75/)
      reward_p=(/0.625,2.75/)
      reward_k=(/0.125,3.0/)
      call hasegawa_cost_fields(cp,ck)
      call check_native(1.0,1.0)

c Exercise production statistics reshuffle and span/time accumulation.
      stat_idir=3
      stat_nm1=2
      stat_nm2=2
      stat_nm3=2
      stat_lnum=nelv
      stat_lmap=-1
      do ie=1,nelv
         stat_lmap(ie)=ie
      enddo
      stat_bm1d(1,:,:,:)=0.25
      stat_bm1d(2,:,:,:)=0.75
      stat_ruavg=123.0
      lnvar=66
      call hasegawa_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      if (lnvar.ne.68) call exitt0
      call check_stats(1.0,1.0)
      actions=2.0*(actions+hg_suction)-hg_suction
      lnvar=66
      call hasegawa_stat_compute(cp,ck,work,lnvar,0.25,0.75)
      call check_stats(1.75,6.25)
      lnvar=66
      call hasegawa_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(2.0,8.0)
      actions=0.5*(actions+hg_suction)-hg_suction

      dt=1.0
      shear=2.0
      call compute_netGain_Hasegawa(1)
      call check_rewards(1.0,1.0,0.2)
c Offsets varying with chord location must cancel independently.
      do ie=1,nelv
         ex=mod(lglel(ie)-1,2)+1
         do iz=1,2
         do ix=1,2
            pr(ix,1,iz,ie)=pr(ix,1,iz,ie)+11*ex*ix
         enddo
         enddo
      enddo
      call compute_netGain_Hasegawa(1)
      call check_rewards(1.0,1.0,0.2)
      lnvar=66
      call hasegawa_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(1.0,1.0)
      pr=pr+100.0
      call compute_netGain_Hasegawa(1)
      call check_rewards(1.0,1.0,0.2)

c Reversing jets changes the positive-only costs, unlike absolute costs.
      actions=-(actions+hg_suction)-hg_suction
      point_p(:,:,1)=reshape((/3.5,0.0,0.0,3.0/),(/2,2/))
      point_p(:,:,2)=reshape((/0.0,1.0,2.5,0.0/),(/2,2/))
      point_k=reshape((/0.0,4.0,0.5,0.0/),(/2,2/))
      stats_p(:,1)=(/0.875,2.25/)
      stats_p(:,2)=(/1.875,0.25/)
      reward_p=(/1.375,1.25/)
      reward_k=(/0.375,1.0/)
      call hasegawa_cost_fields(cp,ck)
      call check_native(1.0,1.0)
      lnvar=66
      call hasegawa_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(1.0,1.0)
      call compute_netGain_Hasegawa(1)
      call check_rewards(1.0,1.0,0.2)
c dt-weighted cycle: dt=1 followed by dt=3 at doubled normal velocity.
      actions=2.0*(actions+hg_suction)-hg_suction
      shear=4.0
      dt=3.0
      call compute_netGain_Hasegawa(2)
      call check_rewards(1.75,6.25,0.35)
      dt=2.0
      call compute_netGain_Hasegawa(1)
      call check_rewards(2.0,8.0,0.4)
      param(1)=2.0
      call compute_netGain_Hasegawa(1)
      call check_rewards(2.0,16.0,0.4)

c Zero DRL action retains suction, hence zero kinetic cost. Pressure
c can still cost power when both the pressure fluctuation and phi < 0.
      actions=0.0
      point_p(:,:,1)=reshape((/0.007,0.007,0.003,0.003/),(/2,2/))
      point_p(:,:,2)=0.0
      point_k=0.0
      call hasegawa_cost_fields(cp,ck)
      call check_native(1.0,1.0)
c Positive DRL action below 0.002 still gives net suction.
      actions=0.001
      call hasegawa_cost_fields(cp,ck)
      call check_native(0.5,1.0)
c Total velocity exactly zero makes both terms zero.
      actions=-hg_suction
      call hasegawa_cost_fields(cp,ck)
      call check_native(0.0,0.0)
c DRL action above the suction offset gives net blowing (phi=0.001).
      actions=0.003
      point_p(:,:,1)=0.0
      point_p(:,:,2)=reshape((/0.0005,0.0005,0.0025,0.0025/),
     $                      (/2,2/))
      point_k=1.0e-9
      call hasegawa_cost_fields(cp,ck)
      call check_native(1.0,1.0)

c Chordwise-only pressure variation gives no fluctuation-pressure cost.
      do ie=1,nelv
         ex=mod(lglel(ie)-1,2)+1
         do iz=1,2
         do ix=1,2
            pr(ix,1,iz,ie)=100.0*ex+10.0*ix
         enddo
         enddo
      enddo
      actions=1.0-hg_suction
      point_p=0.0
      point_k=1.0
      call hasegawa_cost_fields(cp,ck)
      call check_native(1.0,1.0)
      if(nid.eq.0) write(*,*) 'PASS Hasegawa reward/stats, ranks:',np
      call MPI_FINALIZE(ierr)

      contains

      subroutine check_native(pfactor,kfactor)
      real pfactor,kfactor
      integer e,x,z,slab
      do e=1,nelv
      slab=(lglel(e)-1)/2+1
      do z=1,2
      do x=1,2
         call assert_close(cp(x,1,z,e),point_p(x,z,slab)*pfactor,
     $                     'local positive pressure work')
         call assert_close(ck(x,1,z,e),point_k(x,z)*kfactor,
     $                     'local positive kinetic term')
         call assert_close(cp(x,2,z,e),0.0,'unactuated pressure')
         call assert_close(ck(x,2,z,e),0.0,'unactuated kinetic')
      enddo
      enddo
      enddo
      end subroutine

      subroutine check_stats(pfactor,kfactor)
      real pfactor,kfactor
      integer a,x,e,local_e,slab
      do a=1,numctrl
         x=info_agt(3,a)
         e=info_agt(1,a)
         local_e=gllel(e)
         slab=(e-1)/2+1
         call assert_close(stat_ruavg(x,1,local_e,67),
     $        stats_p(x,slab)*pfactor,'statistics pressure cost')
         call assert_close(stat_ruavg(x,1,local_e,68),
     $        reward_k(x)*kfactor,'statistics kinetic cost')
         call assert_close(stat_ruavg(x,2,local_e,67),0.0,
     $                     'statistics unactuated pressure')
         call assert_close(stat_ruavg(x,2,local_e,68),0.0,
     $                     'statistics unactuated kinetic')
         call assert_close(stat_ruavg(x,1,local_e,66),123.0,
     $                     'original statistic preserved')
      enddo
      end subroutine

      subroutine check_rewards(pfactor,kfactor,tau)
      real pfactor,kfactor,tau
      integer a,x
      do a=1,numctrl
         x=info_agt(3,a)
         call assert_close(rwd_pw(a),reward_p(x)*pfactor,
     $                     'positive pressure work before averaging')
         call assert_close(rwd_v3(a),reward_k(x)*kfactor,
     $                     'positive kinetic term before averaging')
         call assert_close(rwd_tau(a),tau,'cycle shear mean')
      enddo
      end subroutine
      end
""" + BEWLEY_DRIVER[BEWLEY_DRIVER.index("      subroutine assert_close("):]


def main():
    run_wing_checks(CASE, DRIVER, "hasegawa")


if __name__ == "__main__":
    main()
