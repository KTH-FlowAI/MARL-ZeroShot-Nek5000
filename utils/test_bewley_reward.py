"""Exercise production Bewley reward/statistics on synthetic MPI wall data.

Run: python utils/test_bewley_reward.py
Requires mpif77 and mpirun. No mesh, policies, or simulation is needed.
The stubs supply solver fields/operators; the reward and wall-pressure
reduction, statistics append routine, and reshuffle/averaging operators are
compiled directly from the new case, including its real DRL/STATS declarations.
Two wall faces are partitioned across 1, 2, and 3 ranks.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "envs/cases/naca4412_US_SS_2586_Bewley"

INCLUDES = {
    "SIZE": """
      integer lx1,ly1,lz1,lelt,lelv,ldim,totctrl,xnel,ynel,znel
      parameter(lx1=2,ly1=2,lz1=2,lelt=2,ldim=3,totctrl=8)
      parameter(lelv=lelt)
      parameter(xnel=2,ynel=1,znel=1)
      integer nelv,nio
      common /test_dims/ nelv,nio
""",
    "INPUT": """
      real param(200)
      common /test_input/ param
      logical if3d
      common /test_flags/ if3d
""",
    "PARALLEL": """
      integer nid,np,gllel(2)
      common /test_parallel/ nid,np,gllel
""",
    "SOLN": """
      real vx(lx1,ly1,lz1,lelt),pr(lx1,ly1,lz1,lelt)
      common /test_soln/ vx,pr
""",
    "TSTEP": """
      real dt
      common /test_tstep/ dt
""",
    "FIXTURE": """
      real shear
      integer surface_calls
      common /test_fixture/ shear,surface_calls
""",
}

DRIVER = """
      program test_bewley
      implicit none
      include 'mpif.h'
      include 'SIZE'
      include 'INPUT'
      include 'PARALLEL'
      include 'SOLN'
      include 'TSTEP'
      include 'DRL'
      include 'BEWLEY'
      include 'STATS'
      include 'FIXTURE'
      integer ierr,eg,ie,ix,iz,il,lnvar
      real ref,expected,base_pw(2,2),base_v3(2)
      real cp(lx1,ly1,lz1,lelt),ck(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt)
      data base_pw /7.0,6.0,1.0,10.0/
      data base_v3 /0.5,4.0/
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,nid,ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,np,ierr)
      nio=nid
      if3d=.true.
      numctrl=0
      nelv=0
      gllel=0
      pr=999.0
      vx=0.0
      actions=0.0
      msk_act=0
      body_sin=0.6
      body_cos=0.8
      param=0.0
      param(1)=1.0
      param(2)=0.1
c Two faces: p=(1,3;5,7) and p+8, weights (1,1;3,3).
c Their combined surface-weighted pressure reference is exactly 9.
c In serial both faces are local; in MPI ranks 0/1 own one face each.
c Rank 2 has no control faces but must participate in the reduction.
      do eg=1,2
         if (np.eq.1.or.nid.eq.eg-1) then
            nelv=nelv+1
            ie=nelv
            gllel(eg)=ie
            do iz=1,2
            do ix=1,2
               pr(ix,1,iz,ie)=1+4*(ix-1)+2*(iz-1)+8*(eg-1)
               actions(ix,1,iz,ie)=ix*(-1.0)**(ix+iz)-bw_suction
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
      surface_calls=0
      call bewley_wall_pressure(pr,ref)
      call assert_close(ref,9.0,'area-weighted global reference')
      if (surface_calls.ne.nelv) call exitt0

c Exercise real statistics reshuffle + span/time accumulation, with the
c span dimension moved from the third index to the first index.
      stat_idir=3
      stat_nm1=2
      stat_nm2=2
      stat_nm3=2
      stat_lnum=nelv
      stat_lmap=-1
      do ie=1,nelv
         stat_lmap(ie)=ie
      enddo
      stat_bm1d=0.5
      stat_ruavg=123.0
      lnvar=66
      call bewley_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      if (lnvar.ne.68) call exitt0
      call check_stats(1.0,1.0)
      actions=2.0*(actions+bw_suction)-bw_suction
      lnvar=66
      call bewley_stat_compute(cp,ck,work,lnvar,0.25,0.75)
      call check_stats(1.75,6.25)
      lnvar=66
      call bewley_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(2.0,8.0)
      actions=0.5*(actions+bw_suction)-bw_suction

      dt=1.0
      shear=2.0
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
c A pressure gauge shift must change neither actuator cost.
      pr=pr+100.0
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
c Reversing ALL jets leaves both absolute costs unchanged.
      actions=-(actions+bw_suction)-bw_suction
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
c dt-weighting: samples at dt=1 and dt=3, speed doubled in sample 2.
      actions=2.0*(actions+bw_suction)-bw_suction
      shear=4.0
      dt=3.0
      call compute_netGain_Bewley(2)
      call check_rewards(1.75,6.25,0.35)
c A new cycle resets both the elapsed time and all three means.
      dt=2.0
      call compute_netGain_Bewley(1)
      call check_rewards(2.0,8.0,0.4)
c Mechanical-pressure convention: density enters the kinetic term.
      param(1)=2.0
      call compute_netGain_Bewley(1)
      call check_rewards(2.0,16.0,0.4)
      actions=-bw_suction
      call compute_netGain_Bewley(1)
      call check_rewards(0.0,0.0,0.4)
c Zero DRL action still costs power: the uniform suction remains active.
      actions=0.0
      call bewley_cost_fields(cp,ck)
      do ie=1,nelv
      do iz=1,2
      do ix=1,2
         call assert_close(ck(ix,1,iz,ie),abs(bw_suction)**3,
     $                     'uniform suction with zero DRL action')
         call assert_close(cp(ix,1,iz,ie),
     $          abs((pr(ix,1,iz,ie)-109.0)*bw_suction),
     $          'uniform suction pressure cost')
         call assert_close(cp(ix,2,iz,ie),0.0,'unactuated pressure')
         call assert_close(ck(ix,2,iz,ie),0.0,'unactuated kinetic')
      enddo
      enddo
      enddo
      if(nid.eq.0) write(*,*) 'PASS Bewley reward/stats, MPI ranks:',np
      call MPI_FINALIZE(ierr)

      contains

      subroutine check_stats(pfactor,kfactor)
      real pfactor,kfactor
      integer a,x,e,local_e
      do a=1,numctrl
         x=info_agt(3,a)
         e=info_agt(1,a)
         local_e=gllel(e)
         call assert_close(stat_ruavg(x,1,local_e,67),
     $        base_pw(x,e)*pfactor,'statistics pressure cost')
         call assert_close(stat_ruavg(x,1,local_e,68),
     $        base_v3(x)*kfactor,'statistics kinetic cost')
         call assert_close(stat_ruavg(x,2,local_e,67),0.0,
     $                     'statistics unactuated node')
         call assert_close(stat_ruavg(x,1,local_e,66),123.0,
     $                     'original statistic preserved')
      enddo
      end subroutine

      subroutine check_rewards(pfactor,kfactor,tau)
      real pfactor,kfactor,tau
      integer a,x,e
      do a=1,numctrl
         x=info_agt(3,a)
         e=info_agt(1,a)
         call assert_close(rwd_pw(a),base_pw(x,e)*pfactor,
     $                     'pointwise pressure-work magnitude')
         call assert_close(rwd_v3(a),base_v3(x)*kfactor,
     $                     'blowing and suction kinetic cost')
         call assert_close(rwd_tau(a),tau,'cycle shear mean')
      enddo
      end subroutine
      end

      subroutine assert_close(actual,expected,label)
      real actual,expected
      character*(*) label
      if (abs(actual-expected).gt.1.0e-11) then
         write(*,*) 'FAIL ',label,actual,expected
         call exitt0
      endif
      end

      subroutine gradm1(dx,dy,dz,v)
      include 'SIZE'
      include 'PARALLEL'
      include 'FIXTURE'
      real dx(*),dy(*),dz(*),v(*)
      integer i
      do i=1,lx1*ly1*lz1*nelv
         dx(i)=0.0
         dy(i)=shear
         dz(i)=0.0
      enddo
      end

      subroutine copy(a,b,n)
      real a(n),b(n)
      integer i,n
      do i=1,n
         a(i)=b(i)
      enddo
      end

      subroutine mappr(pout,pin,tmp1,tmp2)
      include 'SIZE'
      include 'PARALLEL'
      real pout(*),pin(*),tmp1(*),tmp2(*)
      call copy(pout,pin,lx1*ly1*lz1*nelv)
      end

      subroutine z_averaging(a,scratch)
      include 'SIZE'
      include 'PARALLEL'
      real a(lx1,ly1,lz1,lelt),scratch(*),val
      integer x,y,e
      do e=1,nelv
      do y=1,ly1
      do x=1,lx1
         val=0.5*(a(x,y,1,e)+a(x,y,2,e))
         a(x,y,1,e)=val
         a(x,y,2,e)=val
      enddo
      enddo
      enddo
      end

      subroutine z_avg_reshape(a,scratch)
      real a(*),scratch(*)
      end

      subroutine x_averaging(a,scratch)
      real a(*),scratch(*)
      call exitt0
      end

      subroutine x_avg_reshape(a,scratch)
      real a(*),scratch(*)
      call exitt0
      end

      subroutine surface_int(pint,awall,pwall,ie,iface)
      include 'SIZE'
      include 'FIXTURE'
      real pwall(lx1,ly1,lz1,lelt),pint,awall,w
      integer ie,iface,x,z
      surface_calls=surface_calls+1
      pint=0.0
      awall=0.0
      do z=1,2
      do x=1,2
         w=2*x-1
         pint=pint+w*pwall(x,1,z,ie)
         awall=awall+w
      enddo
      enddo
      end

      subroutine gop(a,work,op,n)
      include 'mpif.h'
      integer n,ierr
      real a(n),work(n)
      character*3 op
      call MPI_ALLREDUCE(a,work,n,MPI_DOUBLE_PRECISION,MPI_SUM,
     $                   MPI_COMM_WORLD,ierr)
      call copy(a,work,n)
      end

      subroutine exitt0
      include 'mpif.h'
      integer ierr
      call MPI_ABORT(MPI_COMM_WORLD,1,ierr)
      stop 1
      end

      subroutine exitt
      call exitt0
      end

      subroutine rzero(a,n)
      integer n
      real a(n)
      a=0.0
      end

      subroutine add2sxy(a,alpha,b,beta,n)
      integer n
      real a(n),b(n),alpha,beta
      a=alpha*a+beta*b
      end
"""


def main():
    for name in ("mpif77", "mpirun"):
        if shutil.which(name) is None:
            raise SystemExit(f"Required tool not found: {name}")
    # /tmp can be mounted noexec on the workstations used for this repo.
    with tempfile.TemporaryDirectory(prefix=".bewley-test-",
                                     dir=ROOT / "utils") as tmp:
        path = Path(tmp)
        for name, source in INCLUDES.items():
            (path / name).write_text(source)
        shutil.copyfile(CASE / "inc_src/DRL", path / "DRL")
        shutil.copyfile(CASE / "inc_src/BEWLEY", path / "BEWLEY")
        shutil.copyfile(CASE / "inc_src/STATS", path / "STATS")
        # Compile the actual production operators without linking the DNS.
        stat_source = (CASE / "stats/statistics_2D.f").read_text()
        operators = []
        for name in ("stat_reshufflev", "stat_compute_1Dav1"):
            start = stat_source.index("      subroutine " + name + "(")
            stop = stat_source.index("\nc---", start)
            operators.append(stat_source[start:stop])
        (path / "operators.f").write_text("\n".join(operators))
        (path / "driver.f").write_text(DRIVER)
        executable = path / "check_bewley"
        subprocess.run(
            ["mpif77", "-O0", "-g", "-std=legacy", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", "-fdefault-real-8",
             "-fdefault-double-8", f"-I{path}",
             str(CASE / "drl/drl_bewley.f"),
             str(CASE / "stats/statistics_bewley.f"),
             str(path / "operators.f"), str(path / "driver.f"),
             "-o", str(executable)], check=True, timeout=60,
        )
        for nranks in (1, 2, 3):
            subprocess.run(
                ["mpirun", "--oversubscribe", "-np", str(nranks),
                 str(executable)], check=True, timeout=30,
            )


if __name__ == "__main__":
    main()
