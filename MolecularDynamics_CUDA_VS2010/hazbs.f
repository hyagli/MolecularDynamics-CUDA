C     ------------------------------------------------------------------
C     PROGRAM TO PREPARE INPUT COORDINATES FOR PICTURE
C     ------------------------------------------------------------------
      DIMENSION X(1000),Y(1000),Z(1000)
      OPEN(UNIT=1,FILE='mdse.pic',STATUS='OLD')
      OPEN(UNIT=3,FILE='mdse.bs',STATUS='NEW')
C
      READ(1,2) n
    2 FORMAT(5x,i3)
      do 10 i=1,n
   10 read(1,*) x(i),y(i),z(i)
C
      DO 30 I=1,N
      WRITE(3,21) X(I),Y(I),Z(I)
   21 FORMAT('atom   Cu   ',F12.5,'   ',F12.5,'   ',F12.5)
   30 continue
C
      WRITE(3,33)
   33 FORMAT(/'spec   Cu   0.50   0.25',
     +      //'bonds  Cu  Cu   1.0   3.50   0.05   0.40',
     +      //'tmat  1.0  0.0  0.0  0.0  1.0  0.0  0.0  0.0  1.0',
     +       /'dist  100.0',
     +       /'inc     1.0',
     +       /'scale  40.0',
     +       /'rfac    1.0',
     +       /'bfac    1.0',
     +       /'switches 1 0 1 0 0 1 1 0 0')
C
      CLOSE(UNIT=3)
      CLOSE(UNIT=1)
      STOP
      END

