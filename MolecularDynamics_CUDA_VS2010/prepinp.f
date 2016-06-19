      implicit double precision(a - h, o - z)
      dimension x(1000), y(1000), z(1000)
      character (len=*) line

      open(unit = 1, file = 'mdse.pic', status = 'old')
      open(unit = 2, file = 'mdse-yeni.inp', status = 'new')
      open(unit = 3, file = 'mdse.inp', status = 'old')

      read(1, 10) nnn
 10   format(5x, i3)
      do 40 i = 1, nnn
          read(1, *) x(i), y(i), z(i)
          z(i) = z(i) * 1.05
 40       continue

      read(3, '(A)') line
      write(2, *) line
      read(3, '(A)') line
      write(2, *) line
      write(2, *) '50000,100,500,2,0,1.0,35,35,1,0.0,0.0,Lz'
      write(2, *) '(W(J,K),K=1,6),(NO(J,K),K=1,3)'

      do 50 i = 1, nnn
          write(2, 60) x(i), y(i), z(i)
 60       format (3(1x, f10.5, ','), '1.0,1.0,1.0,1,1,1')
 50       continue

      close(unit = 1)
      close(unit = 2)

      stop
      end
