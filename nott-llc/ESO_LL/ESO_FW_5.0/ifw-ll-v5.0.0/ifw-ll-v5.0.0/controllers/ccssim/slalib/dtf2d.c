#include "TcPch.h"
#pragma hdrstop

#include "slalib.h"
#include "slamac.h"
void slaDtf2d ( int ihour, int imin, double sec, double *days, int *j )
/*
**  - - - - - - - - -
**   s l a D t f 2 d
**  - - - - - - - - -
**
**  Convert hours, minutes, seconds to days.
**
**  (double precision)
**
**  Given:
**     ihour       int           hours
**     imin        int           minutes
**     sec         double        seconds
**
**  Returned:
**     *days       double        interval in days
**     *j          int           status:  0 = OK
**                                        1 = ihour outside range 0-23
**                                        2 = imin outside range 0-59
**                                        3 = sec outside range 0-59.999...
**
**  Notes:
**
**     1)  The result is computed even if any of the range checks fail.
**
**     2)  The sign must be dealt with outside this function.
**
**  Last revision:   22 October 2006
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Seconds per day */
#define D2S 86400.0

{
   int jstat;


/* Preset status. */
   jstat = 0;

/* Validate sec, min, hour. */
   if ( ( sec < 0.0 ) || ( sec >= 60.0 ) ) jstat = 3;
   if ( ( imin < 0 ) || ( imin > 59 ) ) jstat = 2;
   if ( ( ihour < 0 ) || ( ihour > 23 ) ) jstat = 1;

/* Compute interval (irrespective of validation) and return status. */
   *days = ( 60.0 * ( 60.0 * (double) ihour
                           + (double) imin )
                                    + sec ) / D2S;
   *j = jstat;
}
