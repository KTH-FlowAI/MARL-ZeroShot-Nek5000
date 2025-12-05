c==============================================
c DRL subroutines for ACTION 
c Those are inherented from OPPO control implementation 
c Yuning Wang 
c==============================================



c------------------------------------------------------------------
        subroutine drl_action
c=============================================
c       Define variable
c=============================================
        implicit none 
        include "SIZE"
        include "TSTEP"
        include 'PARALLEL'
        include 'INPUT'
        integer i_znmf
c=============================================
c       Function
c=============================================
        
      !   if (ISTEP.eq.0) then
cc STEP 1: Do nothing 
            if (NID.eq.0) then 
                  print *,"-------------------------"
                  print *,"[ACTION] STANDING BY"
                  print *,"-------------------------"
            endif 
cc STEP 3: Computing Fluctuation and Actuating on the wall 
c-----------------------------------------------
      !   else ! When the simulation is running: 
            ! call wall_aft_vel               ! TEST the actutation on the wall 
            call recv_Actions() 
            call nekgsync()

            i_znmf =UPARAM(2) 
            ! YW Comment here,since we use the unique action, the Weighted average will explode
            if (i_znmf.eq.1) then 
                  call znmf_avg_patch()!call znmf_avg()
                  !call znmf_avg()
                  ! call znmf_avg_new()
                  call znmf_check()
            endif 
      !   endif ! if (ISTEP.eq.0)

        return 
        end subroutine drl_action
c------------------------------------------------------------------



c------------------------------------------------------------------
      subroutine recv_Actions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TSTEP'
      include 'NEKUSE'
      include 'INPUT'
      include 'PARALLEL'
      include 'DRL'   
      include "SOLN"      
      include 'mpif.h'
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      !-----------------------
      integer totLine,ntot
      integer i_znmf
      integer glbid,fceid,nidid,nididnek,lclid
      integer ix,iy,iz,iel
      real    jetval
      !-----------------------
      
      !--------------------------
      ! MPI 
      integer parent_comm, ierr
      integer idx_buffer(totctrl)
      integer fce_buffer(totctrl)
      real    act_buffer(totctrl)
      integer sendLen 
      integer request_send, request_recv, status(mpi_status_size)
      !--------------------------
      logical ifexist
      character*13 fNAME
      integer ilx,ily,ilz
      character*4 str,str1

      ! wrking 
      real wrk_buff(LX1,LY1,LZ1,LELT),act_i
      ! Test 
      real diff_buff(LX1,LY1,LZ1,LELT),sts_buff(LX1,LY1,LZ1,LELT)

c=============================================
c       Function
c=============================================
      i_znmf = UPARAM(2) 
      if (NUMCTRL.ne.0) then 

      call MPI_RECV(act_buffer,TOTCTRL,MPI_DOUBLE,
     &            0,NID+90000,DRL_COMM,
     &            MPI_STATUS_IGNORE,ierr)

      ! print *,"[ACTION] UPDATE NID=",NID
      call nekgsync()
      else
      call nekgsync()
      ! print *,"[ACTION] UPDATE NID=",NID
      endif
      ! call nekgsync()
! Update 
!----------------------------------------
      ntot=LX1*LY1*LZ1*LELT
      ! INIT THE buffer with dumi value 
      call rzero(wrk_buff(1,1,1,1),ntot)
      call rzero(msk_act(1,1,1,1),ntot)
      ! Update 
      do il=1,NUMCTRL 
            glbid=info_agt(1,il)
            lclid=gllel(glbid)
            fceid=info_agt(2,il)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            act_i=act_buffer(il)
            wrk_buff(ix,iy,iz,lclid)=act_i
            if (i_znmf.ne.1) then 
            msk_act(ix,iy,iz,lclid)=1.0
            endif 
      enddo 
      

      call copy(ACTIONS(1,1,1,1),wrk_buff(1,1,1,1),NTOT)
      ! totLine=LX1*LY1*LZ1*6*LELT
      ! if(ISTEP.eq.1) call copy(old_ctrl_val,ctrl_val,totLine)
!----------------------------------------

      if (NID.eq.0) then
            print *, "-------------------------"
            print *, "[ACTION] UPDATED"
            print *, "-------------------------"
      endif

          
c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      ! if (NID.eq.0) print *, "[ACTION] ZNMF AVERAGED"
      if (ISTEP.le.3) then 
            if (numctrl.gt.0) then 
            write(str,"(i4.4)") NID
            write(str1,"(i4.4)") ISTEP
            open(10001,file="RECV-ACTION.txt"//str//str1)
            write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, " 
            do ilx = 1,numctrl
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            ix=info_agt(3,ilx)
            iy=info_agt(4,ilx)
            iz=info_agt(5,ilx)
            write(10001,*) iel,
     $      (pos_agt(ily,ilx), ily=1,NDIM),
     $       ACTIONS(ix,iy,iz,iel) 
            enddo
            close(10001)
            endif
      
      call rzero(diff_buff(1,1,1,1),ntot)
      call rzero(sts_buff(1,1,1,1),ntot)
      do il=1,NUMCTRL 
            glbid=info_agt(1,il)
            lclid=gllel(glbid)
            fceid=info_agt(2,il)
            ix=info_agt(3,il)
            iy=info_agt(4,il)
            iz=info_agt(5,il)
            act_i=val_obs(2,il)
            sts_buff(ix,iy,iz,lclid)=act_i
      enddo 
      
      diff_buff(:,:,:,:)=sts_buff(:,:,:,:)+wrk_buff(:,:,:,:)

      call outpost(diff_buff,sts_buff,wrk_buff,vy,t,'dif')
      endif ! if ISTEP.le.3 
#endif 

      end subroutine recv_Actions 
c------------------------------------------------------------------

c------------------------------------------------------------------
      subroutine ACTUATE_JET(vf,isfind,ix,iy,iz,iside,eg)
! Used for userbc: Impose velocity 
! c=============================================
! c       Define variable
! c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,iside,eg,iel
      integer ierr,IO_STATUS
      integer k,il,jl        ! Iteration
      integer len,recctrl ! Flag for counting 
      
      real vf, v_msk
      logical isfind
! c=============================================
! c       Function
! c=============================================
      iel = gllel(eg)
      vf = ACTIONS(ix,iy,iz,iel)
      v_msk = msk_act(ix,iy,iz,iel)
      vf = vf * v_msk ! Masking the action so that only control nodes as BC
      ! vf =0.0
      !isfind =.FALSE.
      !if (NUMCTRL.ne.0) then 
      !      recctrl=msk_act(ix,iy,iz,iel)
      !      if (recctrl.eq.1) then 
      !            vf=ACTIONS(ix,iy,iz,iel)
      !            isfind=.TRUE.
      !      endif
      !endif 

      end subroutine ACTUATE_JET
c--------------------------------------------------


c--------------------------------------------------

      subroutine znmf_avg() 
c Subroutine for average the Actions for ZNMF conditions
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         

      integer ix,iy,iz,ifs,eg,iel,iface
      integer ntot,il
      integer i_evolv,ndrl
      real velV(LX1,LY1,LZ1,LELT)
      real actV(LX1,LY1,LZ1,LELT)
      real wrk_buff(LX1,LY1,LZ1,LELT)
      real avgV(LX1,LY1,LZ1,LELT)
      real vf,va,vo
      real diff_buff(LX1,LY1,LZ1,LELT)
      ! Handlers for gop average
      integer igs_x,igs_z
      save igs_x,igs_z
      ! For test 
      integer ilx, ily
      character*4 str, str1
c=============================================
c       Functions
c=============================================
      ntot=LX1*LY1*LZ1*LELT

      ! STEP 1: Spatial Average based on X- Z-dir 
      ! NOTE: Again this should be modified if we move to NEK-V17 
      if (igs_z.eq.0.and.igs_x.eq.0) then 
      call gtpp_gs_setup(igs_z,xnel*ynel,1,znel,3) ! z-avx
      call gtpp_gs_setup(igs_x,xnel,ynel,znel,1) ! x-avx
      if (NID.eq.0) print *, "[ACTION] AVG HANDLE INIT!",igs_z,igs_x
      endif 

      call copy(velV(1,1,1,1),ACTIONS(1,1,1,1),ntot)
      call copy(actV(1,1,1,1),ACTIONS(1,1,1,1),ntot)
#ifdef YWDEBUG  
      if (ISTEP.eq.1) then 
      call outpost(velV,diff_buff,actV,vy,t,'dss')
      if (NID.eq.0) print *, "[ACTION] SAVED BUFFER FOR TEST"
      endif 
#endif    
      call copy(actV(1,1,1,1),velV(1,1,1,1),ntot)

      if (act_zavg) then
            call planar_avg(avgV,velV,igs_z)
            call copy(velV(1,1,1,1),avgV(1,1,1,1),ntot)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] Z-AVG"
#endif
      endif ! if (act_zavg)

            ! Do streamwise average if it allowed/defined
      if (act_xavg) then 
            call planar_avg(avgV,velV,igs_x)
            call copy(velV(1,1,1,1),avgV(1,1,1,1),ntot)
#ifdef YWDEBUG
            if (NID.eq.0) print *, "[ACTION] X-AVG!"
#endif
      endif 
      
      
      ! Step 2: Subtract the mean of the action 
      ! Write the mask for the action
      do ilx = 1,numctrl
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            iface=info_agt(2,ilx)
            call impose_rvalue(1.0,msk_act,iel,iface)
      enddo
      ! Subtract the mean of the action 
      call rzero(wrk_buff,ntot)
      call sub3(wrk_buff,actV,velV,ntot)
      call copy(ACTIONS(1,1,1,1),wrk_buff(1,1,1,1),NTOT)

      ! Save the result for testing
#ifdef YWDEBUG  
      if (ISTEP.eq.1) then 
      call outpost(velV,actV,wrk_buff,
     $             v1mask,t,'act')
      if (NID.eq.0) print *, "[ACTION] SAVED BUFFER FOR TEST"
      endif 
#endif 

c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG
      if (NID.eq.0) print *, "[ACTION] ZNMF AVERAGED"
      if (ISTEP.le.3) then 
            if (numctrl.gt.0) then 
            write(str,"(i4.4)") NID
            write(str1,"(i4.4)") ISTEP
            open(10001,file="ZNMF-ACTION.txt"//str//str1)
            write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, ","MEAN, ","BEFORE" 
            do ilx = 1,numctrl
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            ix=info_agt(3,ilx)
            iy=info_agt(4,ilx)
            iz=info_agt(5,ilx)
            write(10001,*) iel,
     $      (pos_agt(ily,ilx), ily=1,NDIM),
     $       ACTIONS(ix,iy,iz,iel), 
     $       velV(ix,iy,iz,iel),
     $       actV(ix,iy,iz,iel)
            enddo
            close(10001)
            endif
      endif 
#endif 

      end subroutine znmf_avg
c--------------------------------------------------

c==============================================
c Modified znmf_avg_new subroutine
c Using weighted spatial averaging with dssum
c==============================================

c--------------------------------------------------

      subroutine znmf_avg_patch() 
c Subroutine for average the Actions for ZNMF conditions
c=============================================
c       Define variable
c=============================================
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'         
      integer ix,iy,iz,ifs,eg,iel,iface
      integer ntot,il, nface, cel
      integer color, npatch
      integer KX1,KX2,KY1,KY2,KZ1,KZ2
      integer i_evolv,ndrl
      real velV(LX1,LY1,LZ1,LELT)
      real actV(LX1,LY1,LZ1,LELT)
      real wrk_buff(LX1,LY1,LZ1,LELT)
      real avgV(LX1,LY1,LZ1,LELT)
      real vf,va,vo, vm, vw 
      real diff_buff(LX1,LY1,LZ1,LELT)
      ! Handlers for gop average
      integer igs_x,igs_z
      save igs_x,igs_z
      ! For test 
      integer ilx, ily
      character*4 str, str1
c=============================================
c       Functions
c=============================================
      ntot=LX1*LY1*LZ1*LELT
      npatch = ptch_x*ptch_z
      call copy(velV,ACTIONS,ntot)
      call dssum(velV,lx1,ly1,lz1)
      
      call rzero(wrk_buff,ntot)
      call rzero(patch_mean,npatch)
      call rzero(patch_weight,npatch)
      cel = -1 
      do il=1,NUMCTRL 
        iel = info_agt(1,il)
        if (cel.ne.iel) then 
        cel = iel
        iel = gllel(cel)
        call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,1)
        do iz = KZ1,KZ2
        do iy = KY1,KY2
        do ix = KX1,KX2
         color = color_wall(ix,iy,iz,iel)
         vf = velV(ix,iy,iz,iel) ! vel 
         va = agent_own(ix,iy,iz,iel) ! weight
         vm = patch_mean(color)
         vw = patch_weight(color)
         patch_mean(color)= vm + vf * va 
         patch_weight(color)= vw + va
        enddo 
        enddo 
        enddo 
        endif

      enddo 

      do il=1,npatch
        vf = patch_mean(il)
        va = patch_weight(il)
        vf = glsum(vf,1)
        va = glsum(va,1)
        vo = vf/va
        patch_mean(il) = vo 
!        if (NID.eq.0) print *, "Mean", il, vo
      enddo

      cel = -1 
      do il=1,NUMCTRL 
       iel = info_agt(1,il)
       if (cel.ne.iel) then 
       cel = iel 
       iel = gllel(cel)
       call impose_rvalue(1.0,msk_act,iel,1)
       call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,1)
       do iz = KZ1,KZ2
       do iy = KY1,KY2
       do ix = KX1,KX2
         color = color_wall(ix,iy,iz,iel)
         vf = velV(ix,iy,iz,iel) ! vel 
         va = patch_mean(color)
         wrk_buff(ix,iy,iz,iel)=va 
         vo = vf - va 
         actV(ix,iy,iz,iel) = vo 
       enddo 
       enddo 
       enddo 
       endif 
      enddo 
      call copy(ACTIONS(1,1,1,1),actV(1,1,1,1),ntot)


c--------------------------
c TEST 
c--------------------------
#ifdef YWDEBUG  
      if (ISTEP.eq.1) then 
      call outpost(velV,actV,wrk_buff,
     $             v1mask,t,'pat')
      if (NID.eq.0) print *, "[ACTION] SAVED BUFFER FOR TEST"
      endif 
#endif 

#ifdef YWDEBUG
      if (NID.eq.0) print *, "[ACTION] ZNMF AVERAGED"
      if (ISTEP.le.3) then 
            if (numctrl.gt.0) then 
            write(str,"(i4.4)") NID
            write(str1,"(i4.4)") ISTEP
            open(10001,file="ZNMF-ACTION.txt"//str//str1)
            write(10001,*) "IGL, ", "X, ", "Y, ", "Z, ", 
     $                     "ACT, ","MEAN, ","BEFORE" 
            do ilx = 1,numctrl
            iel=info_agt(1,ilx)
            iel=gllel(iel)
            ix=info_agt(3,ilx)
            iy=info_agt(4,ilx)
            iz=info_agt(5,ilx)
            write(10001,*) iel,
     $      (pos_agt(ily,ilx), ily=1,NDIM),
     $       ACTIONS(ix,iy,iz,iel), 
     $       velV(ix,iy,iz,iel),
     $       actV(ix,iy,iz,iel)
            enddo
            close(10001)
            endif
      endif 
#endif 

      end subroutine znmf_avg_patch
c--------------------------------------------------



c--------------------------------------------------
      subroutine impose_rvalue(vf,buffer,iel,iface)
c Subroutine for imposing the action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      integer iel,iface
      integer ix,iy,iz
      integer KX1,KX2,KY1,KY2,KZ1,KZ2
      real buffer(LX1,LY1,LZ1,LELT)
      real vf
      call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
      do iz=KZ1,KZ2
      do iy=KY1,KY2
      do ix=KX1,KX2
      buffer(ix,iy,iz,iel) = vf
      enddo
      enddo
      enddo

      end subroutine impose_rvalue
c--------------------------------------------------


c--------------------------------------------------
      subroutine impose_ivalue(vf,buffer,iel,iface)
c Subroutine for imposing the action
c=============================================
c       Define variable
c=============================================
      implicit none 
      include 'SIZE'
      include 'TOTAL'
      integer iel,iface
      integer ix,iy,iz
      integer KX1,KX2,KY1,KY2,KZ1,KZ2
      integer buffer(LX1,LY1,LZ1,LELT)
      integer vf
      call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
      do iz=KZ1,KZ2
      do iy=KY1,KY2
      do ix=KX1,KX2
      buffer(ix,iy,iz,iel) = vf
      enddo
      enddo
      enddo

      end subroutine impose_ivalue
c--------------------------------------------------



c--------------------------------------------------
      subroutine znmf_check()
c Subroutine for checking the ZNMF flux
c=============================================
c       Define variable
c=============================================
      include 'SIZE'
      include 'TOTAL'
      include 'DRL'
      common /mystuff/ tx(lx1,ly1,lz1,lelt)
     $ , ty(lx1,ly1,lz1,lelt)
     $ , tz(lx1,ly1,lz1,lelt)
      integer e,f,counter
      integer ielist(TOTCTRL),flist(TOTCTRL)
c=============================================
c       Functions
c=============================================
      nface = 2*ndim
      a = 0.
      s = 0.
      
      ! call gradm1(tx,ty,tz,ACTIONS) ! grad T
      !call copy(tx,vx,lx1*ly1*lz1*nelv)
      !call copy(ty,vy,lx1*ly1*lz1*nelv)
      !call copy(tz,vz,lx1*ly1*lz1*nelv)
      call copy(tx,ACTIONS,lx1*ly1*lz1*nelv)
      call copy(ty,ACTIONS,lx1*ly1*lz1*nelv)
      call copy(tz,ACTIONS,lx1*ly1*lz1*nelv)
      e = -1 
      counter = 0 
      do il = 1, numctrl
      iel = info_agt(1,il)
      if (e.ne.iel) then 
      e = iel 
      iel = gllel(iel)
      ielist(il) = iel
      flist(il) = info_agt(2,il)
      counter = counter +1 
      endif 
      enddo

      do ie=1,counter
      e = ielist(ie)
      f = flist(ie)
      call facind(i0,i1,j0,j1,k0,k1,nx1,ny1,nz1,f)
      l=0
            do k=k0,k1 ! March over face f
            do j=j0,j1
            do i=i0,i1
            l = l + 1
            s = s + uny(l,1,f,e)*ty(i,j,k,e) 
     $  *area(l,1,f,e)
c            s = s + (unx(l,1,f,e)*tx(i,j,k,e)
c     $ + uny(l,1,f,e)*ty(i,j,k,e)
c     $ + unz(l,1,f,e)*tz(i,j,k,e))*area(l,1,f,e)
            a = a + area(l,1,f,e)
            enddo
            enddo
            enddo
      enddo
      
      a=glsum(a,1) ! Sum across processors
      s=glsum(s,1)
      abar = s/a
      
      if (nid.eq.0) then 
      print *, 'ZNMF Flux: ', abar
      endif 
      
      return
      end
c--------------------------------------------------

