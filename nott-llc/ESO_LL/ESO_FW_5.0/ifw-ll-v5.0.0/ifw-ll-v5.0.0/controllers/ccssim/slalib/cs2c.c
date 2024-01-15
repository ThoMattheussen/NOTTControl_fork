#include "TcPch.h"
#pragma hdrstop

#include "slalib.h"
#include "slamac.h"
void slaCs2c ( float a, float b, float v[3] )
/*
**  - - - - - - - -
**   s l a C s 2 c
**  - - - - - - - -
**
**  Spherical coordinates to direction cosines (single precision)
**
**  Given:
**     a,b      float     spherical coordinates in radians
**                           (RA,Dec), (long,lat) etc.
**
**  Returned:
**     v        float[3]  x,y,z unit vector
**
**  The spherical coordinates are longitude (+ve anticlockwise looking
**  from the +ve latitude pole) and latitude.  The Cartesian coordinates
**  are right handed, with the x axis at zero longitude and latitude,
**  and the z axis at the +ve latitude pole.
**
**  Last revision:   22 July 2004
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   float cosb;

   cosb = (float) cos_ ( b );
   v[0] = (float) cos_ ( a ) * cosb;
   v[1] = (float) sin_ ( a ) * cosb;
   v[2] = (float) sin_ ( b );
}
