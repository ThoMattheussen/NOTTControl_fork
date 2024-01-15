#include "../TcPch.h"
#pragma hdrstop

#include "slalib.h"
#include "slamac.h"
void slaS2tp ( float ra, float dec, float raz, float decz,
               float *xi, float *eta, int *j )
/*
**  - - - - - - - -
**   s l a S 2 t p
**  - - - - - - - -
**
**  Projection of spherical coordinates onto tangent plane
**  ('gnomonic' projection - 'standard coordinates').
**
**  (single precision)
**
**  Given:
**     ra,dec     float  spherical coordinates of point to be projected
**     raz,decz   float  spherical coordinates of tangent point
**
**  Returned:
**     *xi,*eta   float  rectangular coordinates on tangent plane
**     *j         int    status:   0 = OK, star on tangent plane
**                                 1 = error, star too far from axis
**                                 2 = error, antistar on tangent plane
**                                 3 = error, antistar too far from axis
**
**  Last revision:   17 August 1999
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
#define TINY 1e-6f
{
   float sdecz, sdec, cdecz, cdec, radif, sradif, cradif, denom;


/* Trig functions */
   sdecz = (float) sin_ ( decz );
   sdec = (float) sin_ ( dec );
   cdecz = (float) cos_ ( decz );
   cdec = (float) cos_ ( dec );
   radif = ra - raz;
   sradif = (float) sin_ ( radif );
   cradif = (float) cos_ ( radif );

/* Reciprocal of star vector length to tangent plane */
   denom = sdec * sdecz + cdec * cdecz * cradif;

/* Handle vectors too far from axis */
   if ( denom > TINY ) {
      *j = 0;
   } else if ( denom >= 0.0f ) {
      *j = 1;
      denom = TINY;
   } else if ( denom > -TINY ) {
      *j = 2;
      denom = -TINY;
   } else {
      *j = 3;
   }

/* Compute tangent plane coordinates (even in dubious cases) */
   *xi  = cdec * sradif / denom;
   *eta = ( sdec * cdecz - cdec * sdecz * cradif ) / denom;
}
