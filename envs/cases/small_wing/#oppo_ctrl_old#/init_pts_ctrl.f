            subroutine ctrl_init 
cc YW: A subroutine for initialisation of ctrl points 
            include 'SIZE'
            include "OPPO_CTL"
            include 'PARALLEL'
            integer ntot, nfail
            real rdum1, rdum2 
            integer il, jl, kl 
            logical ifascii
            real ctlbuff(LDIM,totctrl)
            real walbuff(LDIM,totctrl)
cc YW: For interpolation of wall 
            integer inth_wall
            integer wrcode(totctrl)
            integer welid(totctrl)
            integer wproc(totctrl)
            real wdist(totctrl)
            real wrst(totctrl*LDIM)

cc Communication for the wall points 
            
            common /stat_wallsi/ inth_wall
            common /stat_walliv/ wrcode, welid, wproc
            common /stat_wallrv/ wdist, wrst
             
            ! functions
            integer irecv, iglsum
cc TEST
            character*2 str, str1 
            integer il1, il2 
cc=================================================================
            ifascii = .TRUE.
            if (NID.eq.0) print *,"YW: START INIT CTRL"
            call mask_all_pts ! Mask all coordinates as infty initially            
            ntot = LDIM*totctrl
            rdum1 = crdctl(1,1)
            rdum2 = crdwall(1,1)
            
            do il=1,totctrl 
                  proc(il) = NID
                  wproc(il) = NID
            enddo


            ! Read the profiles
            call read_ctrl_pts(ctlbuff,totctrl,uniqz,ifascii)
            call read_wall_pts(walbuff,totctrl,ifascii)
            
            call intpts_setup(-1.0,inth_hpts)

            call copy(crdctl,ctlbuff,ntot)
            call copy(crdwall,walbuff,ntot)
            ! Interpolation on the grid
            call findpts(inth_hpts,rcode,1,
     &                   proc,1,
     &                   elid,1,
     &                   rst,NDIM,
     &                   dist,1,
     &                   crdctl(1,1),NDIM,
     &                   crdctl(2,1),NDIM,
     &                   crdctl(3,1),NDIM,totctrl)
            ! Check if the results by the tolerance 
            nfail = 0
            do il=1,totctrl
               if(rcode(il).eq.1) then
                  if (dist(il).gt.1e-12) then
                     nfail = nfail + 1
                     if (nfail.le.5) write(6,'(a,1p4e15.7)') 
     &      ' WARNING: point on boundary or outside the mesh xy[z]d^2:'
     &              ,(crdctl(kl,il),kl=1,NDIM),dist(il)
                  endif   
               elseif(rcode(il).eq.2) then
                  nfail = nfail + 1
                  if (nfail.le.5) write(6,'(a,1p3e15.7)') 
     &          ' WARNING: point not within mesh xy[z]: !',
     &          (crdctl(kl,il),kl=1,NDIM),
     &           "Tolerance:",
     &          (dist(il))
               endif
            enddo

            nfail = iglsum(nfail,1)
            if (nfail.gt.0) then
               if (NIO.eq.0) write(6,*) 
     $             'Error: stat_pts; non-zero nfail'
               call exitt
            endif

            ! Interpolation for the control points
            call intpts_setup(-1.0,inth_wall)
            call findpts(inth_wall,wrcode,1,
     &                    wproc,1,
     &                    welid,1,
     &                    wrst,NDIM,
     &                    wdist,1,
     &                    crdwall(1,1),NDIM,
     &                    crdwall(2,1),NDIM,
     &                    crdwall(3,1),NDIM,totctrl)
            ! Check if the results meet tolerance
            nfail = 0
            do il=1,totctrl
            !    check return code 
                  if(wrcode(il).eq.1) then
                  if (wdist(il).gt.1e-12) then
                        nfail = nfail + 1
                        if (nfail.le.5) write(6,'(a,1p4e15.7)') 
     &      ' WARNING: point on boundary or outside the mesh xy[z]d^2:'
     &              ,(crdwall(kl,il),kl=1,NDIM),wdist(il)
                  endif   
                  elseif(wrcode(il).eq.2) then
                  nfail = nfail + 1
                  if (nfail.le.5) write(6,'(a,1p3e15.7)') 
     &          ' WARNING: point not within mesh xy[z]: !',
     &          (crdwall(kl,il),kl=1,NDIM),
     &           "Tolerance:",
     &          (wdist(il))
                  endif
            enddo

            nfail = iglsum(nfail,1)
            if (nfail.gt.0) then
                  if (NIO.eq.0) write(6,*) 
     $             'Error: stat_pts; non-zero nfail'
                  call exitt
            endif

            call wall_grid_search
            
            return
            end


c-----------------------------------------------------------------            
            subroutine mask_all_pts
            
            include 'SIZE'
            include 'OPPO_CTL'
   
            integer il, jl 
c=======================================
            do il = 1,totctrl
                  vwall(1,il) = inftr
                  vwall(2,il) = inftr
                  vwall(3,il) = inftr

                  rwall(1,il) = inftr
                  rwall(2,il) = inftr
                  rwall(3,il) = inftr
                  do jl = 1, nfeat
                  grdwall(jl,il) = infty
                  
                  enddo
            enddo

            return 
            end

c-----------------------------------------------------------------            
            subroutine read_ctrl_pts(pts,lpts,nz,ifascii)
cc YW      A subroutine to read the control points 
cc Args:
cc         pts     :     The array to store the coordinates
cc         lpts    :     Number of points to load


            implicit none 
            
            include 'SIZE'

            real pts(LDIM,lpts)
            integer lpts
            integer npt3d,nz,wdim ! read header 
            logical ifascii 
            integer idum ! dummy variable 
            real rdum1, rdum2 ! dummy variable 
            integer il 
c==============================================
            if (ifascii) then
                  open(50,file='ctl_all.in',status='old')
                  read(50,*) npt3d, nz, idum
                  if (lpts.gt.npt3d) print *,"ONLY read:",npt3d
                  read(50,*) rdum1,rdum2

                  do il=1,npt3d
                        read(50,*)
     $                  pts(1,il),pts(2,il),pts(3,il)
                  enddo
                  close(50)
            else                      ! ifascii
                  print *, "YW: Error:Please provide ASCII file!"
            endif                     ! ifascii
            return
            end
c-----------------------------------------------------------------            

            subroutine read_wall_pts(pts,lpts,ifascii)
              !!c      A subroutine to find the Number of Processor and the indicies of ctrl points
            implicit none 
            
            include 'SIZE'

            real pts(LDIM,lpts)
            integer lpts
            integer npt3d,nz,wdim ! read header 
            logical ifascii 
            integer idum ! dummy variable 
            real rdum1, rdum2 ! dummy variable 
            integer il      
c=================================================            
            if (ifascii) then
                  open(50,file='wall_all.in',status='old')
                  read(50,*) npt3d, nz, wdim 
                  if (lpts.gt.npt3d) print *,"ONLY read:",npt3d
                  read(50,*) rdum1,rdum2  
                  do il=1,npt3d
                        read(50,*)
     $                  pts(1,il),pts(2,il),pts(3,il)
                  enddo
                  close(50)
            else                      ! ifascii
                  print *, "YW: Error:Please provide ASCII file!"
            endif                     ! ifascii
            return
            end
        

c-----------------------------------------------------------------  
            subroutine wall_grid_search
cc A subroutine for searching the wall grids    
            implicit none 
            include 'SIZE'
            include 'TOTAL'
            ! include 'INPUT'
            include 'USERPAR'
            include 'NEKUSE'
            real xf,yf,zf
            real xw,yw,zw

            integer KX1,KX2,KY1,KY2,KZ1,KZ2
            integer nel,nfaces,iegl,fcount,wcount
            integer ie,iface, ix, iy, iz ! iterator
            logical isfind
            character ccb*3 

            character*2 str, str1 ! Test for writting the changes
c===========================================

            
            if (NID.eq.0) print *,"Switching B.C to Dirichlet"
            ! if (NID.eq.0) print *,"IFIELD=",IFIELD
            
            nfaces = 2*ndim ! A cubic 

            NEL=NELFLD(IFIELD)
            
            fcount = 0 
            wcount = 0
            
            ! write(str,'(i2.2)') NID
            ! open(1001, file='changeBC.txt'//str)

            do ie=1,NEL
            do iface=1,nfaces
            call facind(KX1,KX2,KY1,KY2,KZ1,KZ2,NX1,NY1,NZ1,iface)
            do iz=KZ1,KZ2
            do iy=KY1,KY2
            do ix=KX1,KX2
                  iegl = lglel(ie)
                  xf=xm1(ix,iy,iz,ie)
                  yf=ym1(ix,iy,iz,ie)
                  zf=zm1(ix,iy,iz,ie)
                  ccb=CBC(iface,ie,IFIELD)
                  isfind = .FALSE.
                  call find_wall_grid(xf,yf,zf,
     $                                ix,iy,iz,
     $                                iegl,NID,iface,
     $                                isfind)
                  if (isfind .and. ccb.eq.'v  ') then
                        ! print *,"FIND Wall grid:",xf,yf,zf 
                        ! print *,"BC is:",CBC(iface,ie,1)
                        ! CBC(iface,ie,1) = 'v  '
                        fcount = fcount + 1
!                         write(1001,*)
!      $                  NID,"NOCHANGE",xf,yf,zf,ix,iy,iz,iegl,iface
                        
                  elseif (isfind .and. ccb.ne.'v  ') then
                        CBC(iface,ie,ifield) = 'v  '
                        ! print *,"FIND Wall grid:",xf,yf,zf 
                        ! print *, "BUT BCTYPE=",cb
                        wcount = wcount + 1
!                         write(1001,*)
!      $                  NID,"ISCHANGE",xf,yf,zf,ix,iy,iz,iegl,iface
                  endif 
            enddo
            enddo
            enddo
            enddo
            enddo 
            print *, NID, "NOT CHANGE=",fcount,"CHANGED=",wcount
 
            return 
            end subroutine wall_grid_search
c-----------------------------------------------------------------  
            subroutine find_wall_grid(
     $                               xin,yin,zin,
     $                               ix,iy,iz,iel,rkid,f,
     $                               isfind)
             
            implicit none 
            include 'SIZE'
            include 'OPPO_CTL'
            include 'NEKUSE'
            integer iel, ieg, rkid,f
            integer ix,iy,iz
            integer sx,sy,sz,sel,sf,skid
            real xw,yw,zw ! Wall points 
            real xin,yin,zin
            real tole
            integer irepeat
            logical isfind
            parameter(tole=1e-13) 
            integer il,jl ! Iteration  
c============================================
            isfind = .FALSE.
            
            do il = 1,totctrl
                  xw=crdwall(1,il)
                  yw=crdwall(2,il)
                  zw=crdwall(3,il)
                  
                  skid=grdwall(1,il)
                  sx=grdwall(2,il)
                  sy=grdwall(3,il)
                  sz=grdwall(4,il)
                  sel=grdwall(5,il)
                  sf=grdwall(6,il)

                  if(xw.eq.xin .and.
     $               yw.eq.yin .and.   
     $               zw.eq.zin) then    
                        ! isfind = .TRUE.

                        if(skid.eq.rkid .or.
     $                         sx.eq.ix .or.
     $                         sy.eq.iy .or.
     $                         sz.eq.iz .or.
     $                         sel.eq.iel .or.
     $                         sf.eq.f ) then
                        isfind = .FALSE.
                        goto 201
                        else 
                        grdwall(1,il)= rkid       
                        grdwall(2,il)= ix
                        grdwall(3,il)= iy
                        grdwall(4,il)= iz
                        grdwall(5,il)= iel
                        grdwall(6,il)= f
                        isfind = .TRUE.
                        goto 101
                        endif ! if repeat
                  else
                        isfind = .FALSE.
                        goto 201
                  endif ! if matches

201         continue
            enddo 
   
101         return 
            end
             
             