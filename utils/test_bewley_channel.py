"""Compile/test the seven channel Bewley cases with synthetic MPI fields.

Run: python3 utils/test_bewley_channel.py
Uses production cost/reward/statistics routines, actual DRL/STATD declarations,
and production reshuffle/span-averaging operators. Solver geometry, gradients,
and planar averaging are supplied by a tiny two-face fixture.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile

from test_bewley_reward import INCLUDES as WING_INCLUDES, DRIVER as WING_DRIVER

ROOT = Path(__file__).resolve().parents[1]
CASES = ("lc_n7", "lc_n11", "lc_extra_n11", "lc_n7_onlyU", "lc_n7_onlyV",
         "lc_n7_UcombV", "mini_channel")
INCLUDES = dict(WING_INCLUDES)
INCLUDES["FRAMELP"] = "\n"
INCLUDES["MAP2D"] = """
      integer map2d_idir,map2d_lmap(lelt)
      common /test_map/ map2d_idir,map2d_lmap
"""

DRIVER = """
      program test_channel
      implicit none
      include 'mpif.h'
      include 'SIZE'
      include 'PARALLEL'
      include 'INPUT'
      include 'SOLN'
      include 'TSTEP'
      include 'DRL'
      include 'STATD'
      include 'MAP2D'
      include 'FIXTURE'
      integer ierr,eg,ie,x,z,lnvar
      real pbar,cp(lx1,ly1,lz1,lelt),ck(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),basep(2,2),basek(2)
      data basep /7.0,6.0,1.0,10.0/
      data basek /0.5,4.0/
      call MPI_INIT(ierr)
      call MPI_COMM_RANK(MPI_COMM_WORLD,nid,ierr)
      call MPI_COMM_SIZE(MPI_COMM_WORLD,np,ierr)
      nio=nid
      numctrl=0
      nelv=0
      gllel=0
      pr=999.0
      vx=0.0
      actions=99.0
      msk_act=0.0
      param=0.0
      param(1)=1.0
      param(2)=0.1
      if3d=.true.
c Two faces with unequal x quadrature weights (1,3), uniform z weights.
c The area mean is 9; rank 2 has no elements and still joins reductions.
      do eg=1,2
         if (np.eq.1.or.nid.eq.eg-1) then
            nelv=nelv+1
            ie=nelv
            gllel(eg)=ie
            do z=1,2
            do x=1,2
               pr(x,1,z,ie)=1+4*(x-1)+2*(z-1)+8*(eg-1)
               actions(x,1,z,ie)=x*(-1.0)**(x+z)
               msk_act(x,1,z,ie)=1.0
               numctrl=numctrl+1
               info_agt(1,numctrl)=eg
               info_agt(2,numctrl)=1
               info_agt(3,numctrl)=x
               info_agt(4,numctrl)=1
               info_agt(5,numctrl)=z
            enddo
            enddo
         endif
      enddo
      surface_calls=0
      call bewley_wall_pressure(pr,pbar)
      call assert_close(pbar,9.0,'area-weighted pressure reference')
      if (surface_calls.ne.nelv) call exitt

c Statistics: test native -> span-first reshuffle before averaging.
      map2d_idir=3
      map2d_lmap=-1
      do ie=1,nelv
         map2d_lmap(ie)=ie
      enddo
      stat_nm1=2
      stat_nm2=2
      stat_nm3=2
      stat_bm1d=0.5
      stat_ruavg=123.0
      lnvar=44
      call bewley_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      if (lnvar.ne.46) call exitt
      call check_stats(1.0,1.0)
      actions=2.0*actions
      lnvar=44
      call bewley_stat_compute(cp,ck,work,lnvar,0.25,0.75)
      call check_stats(1.75,6.25)
      lnvar=44
      call bewley_stat_compute(cp,ck,work,lnvar,0.0,1.0)
      call check_stats(2.0,8.0)
      actions=0.5*actions

c Reward: full wall means, local magnitudes cannot cancel in x or z.
      dt=1.0
      shear=2.0
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
      pr=pr+100.0
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
      actions=-actions
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,1.0,0.2)
      actions=2.0*actions
      dt=3.0
      shear=4.0
      call compute_netGain_Bewley(2)
      call check_rewards(1.75,6.25,0.35)
      call compute_netGain_Bewley(1)
      call check_rewards(2.0,8.0,0.4)
c Mechanical pressure and dynamic viscosity: density scales kinetic only.
      param(1)=2.0
      call compute_netGain_Bewley(1)
      call check_rewards(2.0,16.0,0.4)
c A fractional mask must affect velocity before the cubic cost.
      msk_act=0.5*msk_act
      call compute_netGain_Bewley(1)
      call check_rewards(1.0,2.0,0.4)
      call bewley_cost_fields(cp,ck)
      do ie=1,nelv
      do z=1,2
      do x=1,2
         call assert_close(cp(x,2,z,ie),0.0,'inactive pressure')
         call assert_close(ck(x,2,z,ie),0.0,'inactive kinetic')
      enddo
      enddo
      enddo
      actions=0.0
      call compute_netGain_Bewley(1)
      call check_rewards(0.0,0.0,0.4)
      if(nid.eq.0) write(*,*) 'PASS channel Bewley, MPI ranks:',np
      call MPI_FINALIZE(ierr)

      contains
      subroutine check_rewards(pfactor,kfactor,tau)
      real pfactor,kfactor,tau
      integer a
      do a=1,numctrl
         call assert_close(rwd_pw(a),7.0*pfactor,'pressure reward')
         call assert_close(rwd_v3(a),3.125*kfactor,'kinetic reward')
         call assert_close(rwd_tau(a),tau,'shear reward')
      enddo
      end subroutine

      subroutine check_stats(pfactor,kfactor)
      real pfactor,kfactor
      integer a,e,x,l
      do a=1,numctrl
         e=info_agt(1,a)
         l=gllel(e)
         x=info_agt(3,a)
         call assert_close(stat_ruavg(x,l,45),basep(x,e)*pfactor,
     $                     'pressure statistic')
         call assert_close(stat_ruavg(x,l,46),basek(x)*kfactor,
     $                     'kinetic statistic')
         call assert_close(stat_ruavg(x+2,l,45),0.0,
     $                     'off-wall statistic')
         call assert_close(stat_ruavg(x,l,44),123.0,
     $                     'original statistics unchanged')
      enddo
      end subroutine
      end

      subroutine gtpp_gs_setup(handle,nx,ny,nz,idir)
      integer handle,nx,ny,nz,idir
      handle=idir
      end

      subroutine planar_avg(out,a,handle)
      include 'SIZE'
      real out(lx1,ly1,lz1,lelt),a(lx1,ly1,lz1,lelt)
      real sums(2),work(2),w
      integer handle,e,x,y,z
      out=0.0
      if (handle.eq.3) then
         do e=1,nelv
         do y=1,2
         do x=1,2
            out(x,y,1,e)=0.5*(a(x,y,1,e)+a(x,y,2,e))
            out(x,y,2,e)=out(x,y,1,e)
         enddo
         enddo
         enddo
      elseif (handle.eq.1) then
         do z=1,2
         do y=1,2
            sums=0.0
            do e=1,nelv
            do x=1,2
               w=2*x-1
               sums(1)=sums(1)+w*a(x,y,z,e)
               sums(2)=sums(2)+w
            enddo
            enddo
            call gop(sums,work,'+  ',2)
            do e=1,nelv
            do x=1,2
               out(x,y,z,e)=sums(1)/sums(2)
            enddo
            enddo
         enddo
         enddo
      else
         call exitt
      endif
      end

      subroutine mntr_abort(id,message)
      integer id
      character*(*) message
      write(*,*) message
      call exitt
      end
"""

# Reuse only generic solver stubs (MPI reduction, quadrature, copy, mappr,
# gradient, assertions). All channel physics/averaging is compiled from cases.
STUBS = WING_DRIVER[WING_DRIVER.index("\n      subroutine assert_close"):]


def main():
    for name in CASES:
        case = ROOT / "envs/cases" / (name + "_Bewley")
        with tempfile.TemporaryDirectory(prefix=".channel-bewley-",
                                         dir=ROOT / "utils") as tmp:
            path = Path(tmp)
            for inc, source in INCLUDES.items():
                (path / inc).write_text(source)
            for inc in ("STATD", "inc_src/DRL"):
                shutil.copyfile(case / inc, path / Path(inc).name)
            source = (case / "stats/stat.f").read_text()
            operators = []
            for routine in ("stat_reshufflev", "stat_compute_1Dav1"):
                start = source.index("      subroutine " + routine + "(")
                stop = source.index("      end subroutine", start)
                operators.append(source[start:stop] + "      end\n")
            (path / "operators.f").write_text("\n".join(operators))
            (path / "driver.f").write_text(DRIVER + STUBS)
            executable = path / "check"
            subprocess.run([
                "mpif77", "-O0", "-g", "-std=legacy", "-fcheck=all",
                "-ffpe-trap=invalid,zero,overflow", "-fdefault-real-8",
                "-fdefault-double-8", "-x", "f77-cpp-input", f"-I{path}",
                str(case / "drl/drl_bewley.f"),
                str(case / "stats/statistics_bewley.f"),
                str(path / "operators.f"), str(path / "driver.f"),
                "-o", str(executable)], check=True, timeout=60)
            print(name + "_Bewley", flush=True)
            for ranks in (1, 2, 3):
                subprocess.run(["mpirun", "--oversubscribe", "-np", str(ranks),
                                str(executable)], check=True, timeout=30)


if __name__ == "__main__":
    main()
