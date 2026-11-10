"""Exercise production Bewley reward/statistics on synthetic MPI wall data.

Run: python utils/test_bewley_reward.py
Requires mpif77 and mpirun. No mesh, policies, or simulation is needed.
The reward, spanwise pressure reference, statistics append routine, and
span/reshape/statistics operators are compiled directly from the wing case.
Four elements span two cross sections and two spanwise slabs, partitioned
across 1, 2, and 3 ranks. Other solver fields/operators are synthetic stubs.
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
      include 'WZ'
      include 'ZPER'
      integer ierr,eg,ie,ix,iz,il,lnvar,ex,ez
      real ref(lx1,ly1,lz1,lelt),expected,base_pw(2),base_v3(2)
      real cp(lx1,ly1,lz1,lelt),ck(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt)
      data base_pw /2.0,4.0/
      data base_v3 /0.5,4.0/
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
c Two cross sections at two spanwise slabs; p = P(s)+[-3,-1,1,3].
c Weights [1,3,1,3] give pbar(s)=P(s)+0.5 and mean |p'|=2.
c Split the span between ranks 0/1; rank 2 owns no elements.
c Unused local slots have lglel=0 and must never enter the averaging.
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
      call bewley_spanwise_pressure(pr,ref)
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
     $                        'spanwise reference preserves input')
         enddo
         enddo
      enddo

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
      stat_bm1d(1,:,:,:)=0.25
      stat_bm1d(2,:,:,:)=0.75
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
c A chord-dependent, spanwise-constant pressure shift must also cancel.
c A single patch-wide pbar would fail this check.
      do ie=1,nelv
         ex=mod(lglel(ie)-1,2)+1
         do iz=1,2
         do ix=1,2
            pr(ix,1,iz,ie)=pr(ix,1,iz,ie)+11*ex*ix
         enddo
         enddo
      enddo
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
      lnvar=66
      call bewley_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(1.0,1.0)
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
      ez=(lglel(ie)-1)/2+1
      do iz=1,2
      do ix=1,2
         call assert_close(ck(ix,1,iz,ie),abs(bw_suction)**3,
     $                     'uniform suction with zero DRL action')
         call assert_close(cp(ix,1,iz,ie),
     $          abs((4*(ez-1)+2*(iz-1)-3.5)*bw_suction),
     $          'uniform suction pressure cost')
         call assert_close(cp(ix,2,iz,ie),0.0,'unactuated pressure')
         call assert_close(ck(ix,2,iz,ie),0.0,'unactuated kinetic')
      enddo
      enddo
      enddo
c A pressure field with ONLY chordwise variation has zero fluctuation
c cost, even with nonzero uniform suction; kinetic cost stays nonzero.
      do ie=1,nelv
         ex=mod(lglel(ie)-1,2)+1
         do iz=1,2
         do ix=1,2
            pr(ix,1,iz,ie)=100.0*ex+10.0*ix
         enddo
         enddo
      enddo
      call bewley_cost_fields(cp,ck)
      do ie=1,nelv
      do iz=1,2
      do ix=1,2
         call assert_close(cp(ix,1,iz,ie),0.0,
     $                     'chordwise mean excluded from pressure cost')
         call assert_close(ck(ix,1,iz,ie),abs(bw_suction)**3,
     $                     'kinetic cost independent of pressure mean')
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
     $        base_pw(x)*pfactor,'statistics pressure cost')
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
         call assert_close(rwd_pw(a),base_pw(x)*pfactor,
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


def run_wing_checks(case, driver_source, formulation):
    """Compile a wing cost implementation against the shared MPI fixture."""
    for name in ("mpif77", "mpirun"):
        if shutil.which(name) is None:
            raise SystemExit(f"Required tool not found: {name}")
    # /tmp can be mounted noexec on the workstations used for this repo.
    with tempfile.TemporaryDirectory(prefix=f".{formulation}-test-",
                                     dir=ROOT / "utils") as tmp:
        path = Path(tmp)
        # Keep INCLUDES and the generic stubs compatible with channel tests.
        wing_includes = dict(INCLUDES)
        wing_includes["SIZE"] = INCLUDES["SIZE"].replace(
            "lelt=2", "lelt=4").replace("totctrl=8", "totctrl=16").replace(
            "znel=1", "znel=2")
        wing_includes["PARALLEL"] = INCLUDES["PARALLEL"].replace(
            "gllel(2)", "gllel(4)") + """
      integer lglel(lelt)
      common /test_lglel/ lglel
"""
        wing_includes["GEOM"] = "\n"
        wing_includes["WZ"] = """
      real wzm1(lz1)
      common /test_weights/ wzm1
"""
        wing_includes["ZPER"] = """
      integer nelz
      common /test_zper/ nelz
"""
        for name, source in wing_includes.items():
            (path / name).write_text(source)
        shutil.copyfile(case / "inc_src/DRL", path / "DRL")
        constants = formulation.upper()
        shutil.copyfile(case / "inc_src" / constants, path / constants)
        shutil.copyfile(case / "inc_src/STATS", path / "STATS")
        # Compile the actual production operators without linking the DNS.
        stat_source = (case / "stats/statistics_2D.f").read_text()
        operators = []
        for name in ("stat_reshufflev", "stat_compute_1Dav1"):
            start = stat_source.index("      subroutine " + name + "(")
            stop = stat_source.index("\nc---", start)
            operators.append(stat_source[start:stop])
        span_source = (case / "drl/drl_state.f").read_text()
        for name in ("z_averaging", "z_avg_reshape", "get_exyz_usr"):
            start = span_source.index("        subroutine " + name + "(")
            stop = span_source.index("        end subroutine " + name, start)
            operators.append(span_source[start:stop] + "        end\n")
        (path / "operators.f").write_text("\n".join(operators))
        driver = driver_source
        for name in ("z_averaging", "z_avg_reshape"):
            start = driver.index("      subroutine " + name + "(")
            stop = driver.index("\n      subroutine ", start + 1)
            driver = driver[:start] + driver[stop:]
        driver += """
      subroutine invcol2(a,b,n)
      integer n
      real a(n),b(n)
      a=a/b
      end

      integer function mod1(i,n)
      integer i,n
      if(i.le.0.or.n.le.0) call exitt0
      mod1=mod(i-1,n)+1
      end
"""
        (path / "driver.f").write_text(driver)
        executable = path / f"check_{formulation}"
        subprocess.run(
            ["mpif77", "-O0", "-g", "-std=legacy", "-fcheck=all",
             "-ffpe-trap=invalid,zero,overflow", "-fdefault-real-8",
             "-fdefault-double-8", f"-I{path}",
             str(case / f"drl/drl_{formulation}.f"),
             str(case / f"stats/statistics_{formulation}.f"),
             str(path / "operators.f"), str(path / "driver.f"),
             "-o", str(executable)], check=True, timeout=60,
        )
        for nranks in (1, 2, 3):
            subprocess.run(
                ["mpirun", "--oversubscribe", "-np", str(nranks),
                 str(executable)], check=True, timeout=30,
            )


def main():
    run_wing_checks(CASE, DRIVER, "bewley")


if __name__ == "__main__":
    main()
