c-----------------------------------------------------------------------
      subroutine nes_stat_compute(bp,bk,hp,hk,lnvar,alpha,beta)
c Append four costs after the 66 original statistics. Caller scratch is
c available after the filter terms. Each field has its own time weights.
      implicit none
      include 'SIZE'
      real bp(lx1,ly1,lz1,lelt),bk(lx1,ly1,lz1,lelt)
      real hp(lx1,ly1,lz1,lelt),hk(lx1,ly1,lz1,lelt)
      real work(lx1,ly1,lz1,lelt),alpha,beta
      integer lnvar
      save work
      if (lnvar.ne.66) then
         write(6,*) 'NES70: expected 66 original statistics',lnvar
         call exitt
      endif
      call nes_cost_fields(bp,bk,hp,hk)
      call nes_stat_add(bp,work,lnvar,alpha,beta)
      call nes_stat_add(bk,work,lnvar,alpha,beta)
      call nes_stat_add(hp,work,lnvar,alpha,beta)
      call nes_stat_add(hk,work,lnvar,alpha,beta)
      return
      end

c-----------------------------------------------------------------------
      subroutine nes_stat_add(cost,work,lnvar,alpha,beta)
      implicit none
      include 'SIZE'
      include 'INPUT'
      real cost(lx1,ly1,lz1,lelt),work(lx1,ly1,lz1,lelt)
      real alpha,beta
      integer lnvar
      if (if3d) then
         call stat_reshufflev(work,cost,nelv)
      else
         call copy(work,cost,lx1*ly1*lz1*nelv)
      endif
      lnvar=lnvar+1
      call stat_compute_1Dav1(work,lnvar,alpha,beta)
      return
      end
