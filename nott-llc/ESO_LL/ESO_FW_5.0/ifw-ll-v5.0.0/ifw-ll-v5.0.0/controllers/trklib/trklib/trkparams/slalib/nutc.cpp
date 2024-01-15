#include "../TcPch.h"
#pragma hdrstop

#include "slalib.h"
#include "slamac.h"
void slaNutc ( double date, double *dpsi, double *deps, double *eps0 )
/*
**  - - - - - - - -
**   s l a N u t c
**  - - - - - - - -
**
**  Nutation:  longitude & obliquity components and mean obliquity,
**  using the Shirai & Fukushima (2001) theory.
**
**  Given:
**     date        double    TDB (loosely ET) as Modified Julian Date
**                                                 (JD-2400000.5)
**  Returned:
**     dpsi,deps   double    nutation in longitude,obliquity
**     eps0        double    mean obliquity
**
**  Notes:
**
**  1  The function predicts forced nutation (but not free core
**     nutation) plus corrections to the IAU 1976 precession model.
**
**  2  Earth attitude predictions made by combining the present nutation
**     model with IAU 1976 precession are accurate to 1 mas (with
**     respect to the ICRS) for a few decades around 2000.
**
**  3  The slaNutc80 function is the equivalent of the present function
**     but using the IAU 1980 nutation theory.  The older theory is less
**     accurate, leading to errors as large as 350 mas over the interval
**     1900-2100, mainly because of the error in the IAU 1976
**     precession.
**
**  References:
**
**     Shirai, T. & Fukushima, T., 2001, Astron.J. 121, 3270-3283
**
**     Fukushima, T., 1991, Astron.Astrophys. 244, L11
**
**     Simon, J. L., Bretagnon, P., Chapront, J., Chapront-Touze, M.,
**     Francou, G. & Laskar, J., 1994, Astron.Astrophys. 282, 663
**
**  Last revision:   22 October 2006
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

#define TURNAS 1296000.0          /* arc seconds in a full circle */
#define DJM0 51544.5              /* reference epoch (J2000), MJD */
#define DJC 36525.0               /* days per Julian century */

{

/*
** --------------------------------
** The SF2001 forced nutation model
** --------------------------------
*/

/* Coefficients of fundamental angles */
   static int na[][9] = {
      {  0,    0,    0,    0,   -1,    0,    0,    0,    0 },
      {  0,    0,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  0,    0,    0,    0,   -2,    0,    0,    0,    0 },
      {  0,    1,    0,    0,    0,    0,    0,    0,    0 },
      {  0,    1,    2,   -2,    2,    0,    0,    0,    0 },
      {  1,    0,    0,    0,    0,    0,    0,    0,    0 },
      {  0,    0,    2,    0,    1,    0,    0,    0,    0 },
      {  1,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  0,   -1,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    0,    2,   -2,    1,    0,    0,    0,    0 },
      { -1,    0,    2,    0,    2,    0,    0,    0,    0 },
      { -1,    0,    0,    2,    0,    0,    0,    0,    0 },
      {  1,    0,    0,    0,    1,    0,    0,    0,    0 },
      {  1,    0,    0,    0,   -1,    0,    0,    0,    0 },
      { -1,    0,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    0,    2,    0,    1,    0,    0,    0,    0 },
      { -2,    0,    2,    0,    1,    0,    0,    0,    0 },
      {  0,    0,    0,    2,    0,    0,    0,    0,    0 },
      {  0,    0,    2,    2,    2,    0,    0,    0,    0 },
      {  2,    0,    0,   -2,    0,    0,    0,    0,    0 },
      {  2,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  1,    0,    2,   -2,    2,    0,    0,    0,    0 },
      { -1,    0,    2,    0,    1,    0,    0,    0,    0 },
      {  2,    0,    0,    0,    0,    0,    0,    0,    0 },
      {  0,    0,    2,    0,    0,    0,    0,    0,    0 },
      {  0,    1,    0,    0,    1,    0,    0,    0,    0 },
      { -1,    0,    0,    2,    1,    0,    0,    0,    0 },
      {  0,    2,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    0,    2,   -2,    0,    0,    0,    0,    0 },
      { -1,    0,    0,    2,   -1,    0,    0,    0,    0 },
      {  0,    1,    0,    0,   -1,    0,    0,    0,    0 },
      {  0,    2,    0,    0,    0,    0,    0,    0,    0 },
      { -1,    0,    2,    2,    1,    0,    0,    0,    0 },
      {  1,    0,    2,    2,    2,    0,    0,    0,    0 },
      {  0,    1,    2,    0,    2,    0,    0,    0,    0 },
      { -2,    0,    2,    0,    0,    0,    0,    0,    0 },
      {  0,    0,    2,    2,    1,    0,    0,    0,    0 },
      {  0,   -1,    2,    0,    2,    0,    0,    0,    0 },
      {  0,    0,    0,    2,    1,    0,    0,    0,    0 },
      {  1,    0,    2,   -2,    1,    0,    0,    0,    0 },
      {  2,    0,    0,   -2,   -1,    0,    0,    0,    0 },
      {  2,    0,    2,   -2,    2,    0,    0,    0,    0 },
      {  2,    0,    2,    0,    1,    0,    0,    0,    0 },
      {  0,    0,    0,    2,   -1,    0,    0,    0,    0 },
      {  0,   -1,    2,   -2,    1,    0,    0,    0,    0 },
      { -1,   -1,    0,    2,    0,    0,    0,    0,    0 },
      {  2,    0,    0,   -2,    1,    0,    0,    0,    0 },
      {  1,    0,    0,    2,    0,    0,    0,    0,    0 },
      {  0,    1,    2,   -2,    1,    0,    0,    0,    0 },
      {  1,   -1,    0,    0,    0,    0,    0,    0,    0 },
      { -2,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  0,   -1,    0,    2,    0,    0,    0,    0,    0 },
      {  3,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  0,    0,    0,    1,    0,    0,    0,    0,    0 },
      {  1,   -1,    2,    0,    2,    0,    0,    0,    0 },
      {  1,    0,    0,   -1,    0,    0,    0,    0,    0 },
      { -1,   -1,    2,    2,    2,    0,    0,    0,    0 },
      { -1,    0,    2,    0,    0,    0,    0,    0,    0 },
      {  2,    0,    0,    0,   -1,    0,    0,    0,    0 },
      {  0,   -1,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    1,    2,    0,    2,    0,    0,    0,    0 },
      {  2,    0,    0,    0,    1,    0,    0,    0,    0 },
      {  1,    1,    0,    0,    0,    0,    0,    0,    0 },
      {  1,    0,   -2,    2,   -1,    0,    0,    0,    0 },
      {  1,    0,    2,    0,    0,    0,    0,    0,    0 },
      { -1,    1,    0,    1,    0,    0,    0,    0,    0 },
      {  1,    0,    0,    0,    2,    0,    0,    0,    0 },
      { -1,    0,    1,    0,    1,    0,    0,    0,    0 },
      {  0,    0,    2,    1,    2,    0,    0,    0,    0 },
      { -1,    1,    0,    1,    1,    0,    0,    0,    0 },
      { -1,    0,    2,    4,    2,    0,    0,    0,    0 },
      {  0,   -2,    2,   -2,    1,    0,    0,    0,    0 },
      {  1,    0,    2,    2,    1,    0,    0,    0,    0 },
      {  1,    0,    0,    0,   -2,    0,    0,    0,    0 },
      { -2,    0,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    1,    2,   -2,    2,    0,    0,    0,    0 },
      { -2,    0,    2,    4,    2,    0,    0,    0,    0 },
      { -1,    0,    4,    0,    2,    0,    0,    0,    0 },
      {  2,    0,    2,   -2,    1,    0,    0,    0,    0 },
      {  1,    0,    0,   -1,   -1,    0,    0,    0,    0 },
      {  2,    0,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    0,    0,    2,    1,    0,    0,    0,    0 },
      {  3,    0,    0,    0,    0,    0,    0,    0,    0 },
      {  0,    0,    2,   -2,   -1,    0,    0,    0,    0 },
      {  3,    0,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    0,    4,   -2,    2,    0,    0,    0,    0 },
      { -1,    0,    0,    4,    0,    0,    0,    0,    0 },
      {  0,    1,    2,    0,    1,    0,    0,    0,    0 },
      {  0,    0,    2,   -2,    3,    0,    0,    0,    0 },
      { -2,    0,    0,    4,    0,    0,    0,    0,    0 },
      { -1,   -1,    0,    2,    1,    0,    0,    0,    0 },
      { -2,    0,    2,    0,   -1,    0,    0,    0,    0 },
      {  0,    0,    2,    0,   -1,    0,    0,    0,    0 },
      {  0,   -1,    2,    0,    1,    0,    0,    0,    0 },
      {  0,    1,    0,    0,    2,    0,    0,    0,    0 },
      {  0,    0,    2,   -1,    2,    0,    0,    0,    0 },
      {  2,    1,    0,   -2,    0,    0,    0,    0,    0 },
      {  0,    0,    2,    4,    2,    0,    0,    0,    0 },
      { -1,   -1,    0,    2,   -1,    0,    0,    0,    0 },
      { -1,    1,    0,    2,    0,    0,    0,    0,    0 },
      {  1,   -1,    0,    0,    1,    0,    0,    0,    0 },
      {  0,   -1,    2,   -2,    0,    0,    0,    0,    0 },
      {  0,    1,    0,    0,   -2,    0,    0,    0,    0 },
      {  1,   -1,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    0,    0,    2,   -1,    0,    0,    0,    0 },
      { -1,    1,    2,    2,    2,    0,    0,    0,    0 },
      {  3,    0,    2,    0,    1,    0,    0,    0,    0 },
      {  0,    1,    2,    2,    2,    0,    0,    0,    0 },
      {  1,    0,    2,   -2,    0,    0,    0,    0,    0 },
      { -1,    0,   -2,    4,   -1,    0,    0,    0,    0 },
      { -1,   -1,    2,    2,    1,    0,    0,    0,    0 },
      {  0,   -1,    2,    2,    1,    0,    0,    0,    0 },
      {  2,   -1,    2,    0,    2,    0,    0,    0,    0 },
      {  0,    0,    0,    2,    2,    0,    0,    0,    0 },
      {  1,   -1,    2,    0,    1,    0,    0,    0,    0 },
      { -1,    1,    2,    0,    2,    0,    0,    0,    0 },
      {  0,    1,    0,    2,    0,    0,    0,    0,    0 },
      {  0,    1,    2,   -2,    0,    0,    0,    0,    0 },
      {  0,    3,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    0,    0,    1,    1,    0,    0,    0,    0 },
      { -1,    0,    2,    2,    0,    0,    0,    0,    0 },
      {  2,    1,    2,    0,    2,    0,    0,    0,    0 },
      {  1,    1,    0,    0,    1,    0,    0,    0,    0 },
      {  2,    0,    0,    2,    0,    0,    0,    0,    0 },
      {  1,    1,    2,    0,    1,    0,    0,    0,    0 },
      { -1,    0,    0,    2,    2,    0,    0,    0,    0 },
      {  1,    0,   -2,    2,    0,    0,    0,    0,    0 },
      {  0,   -1,    0,    2,   -1,    0,    0,    0,    0 },
      { -1,    0,    1,    0,    2,    0,    0,    0,    0 },
      {  0,    1,    0,    1,    0,    0,    0,    0,    0 },
      {  1,    0,   -2,    2,   -2,    0,    0,    0,    0 },
      {  0,    0,    0,    1,   -1,    0,    0,    0,    0 },
      {  1,   -1,    0,    0,   -1,    0,    0,    0,    0 },
      {  0,    0,    0,    4,    0,    0,    0,    0,    0 },
      {  1,   -1,    0,    2,    0,    0,    0,    0,    0 },
      {  1,    0,    2,    1,    2,    0,    0,    0,    0 },
      {  1,    0,    2,   -1,    2,    0,    0,    0,    0 },
      { -1,    0,    0,    2,   -2,    0,    0,    0,    0 },
      {  0,    0,    2,    1,    1,    0,    0,    0,    0 },
      { -1,    0,    2,    0,   -1,    0,    0,    0,    0 },
      { -1,    0,    2,    4,    1,    0,    0,    0,    0 },
      {  0,    0,    2,    2,    0,    0,    0,    0,    0 },
      {  1,    1,    2,   -2,    1,    0,    0,    0,    0 },
      {  0,    0,    1,    0,    1,    0,    0,    0,    0 },
      { -1,    0,    2,   -1,    1,    0,    0,    0,    0 },
      { -2,    0,    2,    2,    1,    0,    0,    0,    0 },
      {  2,   -1,    0,    0,    0,    0,    0,    0,    0 },
      {  4,    0,    2,    0,    2,    0,    0,    0,    0 },
      {  2,    1,    2,   -2,    2,    0,    0,    0,    0 },
      {  0,    1,    2,    1,    2,    0,    0,    0,    0 },
      {  1,    0,    4,   -2,    2,    0,    0,    0,    0 },
      {  1,    1,    0,    0,   -1,    0,    0,    0,    0 },
      { -2,    0,    2,    4,    1,    0,    0,    0,    0 },
      {  2,    0,    2,    0,    0,    0,    0,    0,    0 },
      { -1,    0,    1,    0,    0,    0,    0,    0,    0 },
      {  1,    0,    0,    1,    0,    0,    0,    0,    0 },
      {  0,    1,    0,    2,    1,    0,    0,    0,    0 },
      { -1,    0,    4,    0,    1,    0,    0,    0,    0 },
      { -1,    0,    0,    4,    1,    0,    0,    0,    0 },
      {  2,    0,    2,    2,    1,    0,    0,    0,    0 },
      {  2,    1,    0,    0,    0,    0,    0,    0,    0 },
      {  0,    0,    5,   -5,    5,   -3,    0,    0,    0 },
      {  0,    0,    0,    0,    0,    0,    0,    2,    0 },
      {  0,    0,    1,   -1,    1,    0,    0,   -1,    0 },
      {  0,    0,   -1,    1,   -1,    1,    0,    0,    0 },
      {  0,    0,   -1,    1,    0,    0,    2,    0,    0 },
      {  0,    0,    3,   -3,    3,    0,    0,   -1,    0 },
      {  0,    0,   -8,    8,   -7,    5,    0,    0,    0 },
      {  0,    0,   -1,    1,   -1,    0,    2,    0,    0 },
      {  0,    0,   -2,    2,   -2,    2,    0,    0,    0 },
      {  0,    0,   -6,    6,   -6,    4,    0,    0,    0 },
      {  0,    0,   -2,    2,   -2,    0,    8,   -3,    0 },
      {  0,    0,    6,   -6,    6,    0,   -8,    3,    0 },
      {  0,    0,    4,   -4,    4,   -2,    0,    0,    0 },
      {  0,    0,   -3,    3,   -3,    2,    0,    0,    0 },
      {  0,    0,    4,   -4,    3,    0,   -8,    3,    0 },
      {  0,    0,   -4,    4,   -5,    0,    8,   -3,    0 },
      {  0,    0,    0,    0,    0,    2,    0,    0,    0 },
      {  0,    0,   -4,    4,   -4,    3,    0,    0,    0 },
      {  0,    1,   -1,    1,   -1,    0,    0,    1,    0 },
      {  0,    0,    0,    0,    0,    0,    0,    1,    0 },
      {  0,    0,    1,   -1,    1,    1,    0,    0,    0 },
      {  0,    0,    2,   -2,    2,    0,   -2,    0,    0 },
      {  0,   -1,   -7,    7,   -7,    5,    0,    0,    0 },
      { -2,    0,    2,    0,    2,    0,    0,   -2,    0 },
      { -2,    0,    2,    0,    1,    0,    0,   -3,    0 },
      {  0,    0,    2,   -2,    2,    0,    0,   -2,    0 },
      {  0,    0,    1,   -1,    1,    0,    0,    1,    0 },
      {  0,    0,    0,    0,    0,    0,    0,    0,    2 },
      {  0,    0,    0,    0,    0,    0,    0,    0,    1 },
      {  2,    0,   -2,    0,   -2,    0,    0,    3,    0 },
      {  0,    0,    1,   -1,    1,    0,    0,   -2,    0 },
      {  0,    0,   -7,    7,   -7,    5,    0,    0,    0 }
   };

/* Nutation series:  longitude. */
   static double psi[][4] = {
      {  3341.5,  17206241.8,    3.1,  17409.5 },
      { -1716.8,  -1317185.3,    1.4,   -156.8 },
      {   285.7,   -227667.0,    0.3,    -23.5 },
      {   -68.6,   -207448.0,    0.0,    -21.4 },
      {   950.3,    147607.9,   -2.3,   -355.0 },
      {   -66.7,    -51689.1,    0.2,    122.6 },
      {  -108.6,     71117.6,    0.0,      7.0 },
      {    35.6,    -38740.2,    0.1,    -36.2 },
      {    85.4,    -30127.6,    0.0,     -3.1 },
      {     9.0,     21583.0,    0.1,    -50.3 },
      {    22.1,     12822.8,    0.0,     13.3 },
      {     3.4,     12350.8,    0.0,      1.3 },
      {   -21.1,     15699.4,    0.0,      1.6 },
      {     4.2,      6313.8,    0.0,      6.2 },
      {   -22.8,      5796.9,    0.0,      6.1 },
      {    15.7,     -5961.1,    0.0,     -0.6 },
      {    13.1,     -5159.1,    0.0,     -4.6 },
      {     1.8,      4592.7,    0.0,      4.5 },
      {   -17.5,      6336.0,    0.0,      0.7 },
      {    16.3,     -3851.1,    0.0,     -0.4 },
      {    -2.8,      4771.7,    0.0,      0.5 },
      {    13.8,     -3099.3,    0.0,     -0.3 },
      {     0.2,      2860.3,    0.0,      0.3 },
      {     1.4,      2045.3,    0.0,      2.0 },
      {    -8.6,      2922.6,    0.0,      0.3 },
      {    -7.7,      2587.9,    0.0,      0.2 },
      {     8.8,     -1408.1,    0.0,      3.7 },
      {     1.4,      1517.5,    0.0,      1.5 },
      {    -1.9,     -1579.7,    0.0,      7.7 },
      {     1.3,     -2178.6,    0.0,     -0.2 },
      {    -4.8,      1286.8,    0.0,      1.3 },
      {     6.3,      1267.2,    0.0,     -4.0 },
      {    -1.0,      1669.3,    0.0,     -8.3 },
      {     2.4,     -1020.0,    0.0,     -0.9 },
      {     4.5,      -766.9,    0.0,      0.0 },
      {    -1.1,       756.5,    0.0,     -1.7 },
      {    -1.4,     -1097.3,    0.0,     -0.5 },
      {     2.6,      -663.0,    0.0,     -0.6 },
      {     0.8,      -714.1,    0.0,      1.6 },
      {     0.4,      -629.9,    0.0,     -0.6 },
      {     0.3,       580.4,    0.0,      0.6 },
      {    -1.6,       577.3,    0.0,      0.5 },
      {    -0.9,       644.4,    0.0,      0.0 },
      {     2.2,      -534.0,    0.0,     -0.5 },
      {    -2.5,       493.3,    0.0,      0.5 },
      {    -0.1,      -477.3,    0.0,     -2.4 },
      {    -0.9,       735.0,    0.0,     -1.7 },
      {     0.7,       406.2,    0.0,      0.4 },
      {    -2.8,       656.9,    0.0,      0.0 },
      {     0.6,       358.0,    0.0,      2.0 },
      {    -0.7,       472.5,    0.0,     -1.1 },
      {    -0.1,      -300.5,    0.0,      0.0 },
      {    -1.2,       435.1,    0.0,     -1.0 },
      {     1.8,      -289.4,    0.0,      0.0 },
      {     0.6,      -422.6,    0.0,      0.0 },
      {     0.8,      -287.6,    0.0,      0.6 },
      {   -38.6,      -392.3,    0.0,      0.0 },
      {     0.7,      -281.8,    0.0,      0.6 },
      {     0.6,      -405.7,    0.0,      0.0 },
      {    -1.2,       229.0,    0.0,      0.2 },
      {     1.1,      -264.3,    0.0,      0.5 },
      {    -0.7,       247.9,    0.0,     -0.5 },
      {    -0.2,       218.0,    0.0,      0.2 },
      {     0.6,      -339.0,    0.0,      0.8 },
      {    -0.7,       198.7,    0.0,      0.2 },
      {    -1.5,       334.0,    0.0,      0.0 },
      {     0.1,       334.0,    0.0,      0.0 },
      {    -0.1,      -198.1,    0.0,      0.0 },
      {  -106.6,         0.0,    0.0,      0.0 },
      {    -0.5,       165.8,    0.0,      0.0 },
      {     0.0,       134.8,    0.0,      0.0 },
      {     0.9,      -151.6,    0.0,      0.0 },
      {     0.0,      -129.7,    0.0,      0.0 },
      {     0.8,      -132.8,    0.0,     -0.1 },
      {     0.5,      -140.7,    0.0,      0.0 },
      {    -0.1,       138.4,    0.0,      0.0 },
      {     0.0,       129.0,    0.0,     -0.3 },
      {     0.5,      -121.2,    0.0,      0.0 },
      {    -0.3,       114.5,    0.0,      0.0 },
      {    -0.1,       101.8,    0.0,      0.0 },
      {    -3.6,      -101.9,    0.0,      0.0 },
      {     0.8,      -109.4,    0.0,      0.0 },
      {     0.2,       -97.0,    0.0,      0.0 },
      {    -0.7,       157.3,    0.0,      0.0 },
      {     0.2,       -83.3,    0.0,      0.0 },
      {    -0.3,        93.3,    0.0,      0.0 },
      {    -0.1,        92.1,    0.0,      0.0 },
      {    -0.5,       133.6,    0.0,      0.0 },
      {    -0.1,        81.5,    0.0,      0.0 },
      {     0.0,       123.9,    0.0,      0.0 },
      {    -0.3,       128.1,    0.0,      0.0 },
      {     0.1,        74.1,    0.0,     -0.3 },
      {    -0.2,       -70.3,    0.0,      0.0 },
      {    -0.4,        66.6,    0.0,      0.0 },
      {     0.1,       -66.7,    0.0,      0.0 },
      {    -0.7,        69.3,    0.0,     -0.3 },
      {     0.0,       -70.4,    0.0,      0.0 },
      {    -0.1,       101.5,    0.0,      0.0 },
      {     0.5,       -69.1,    0.0,      0.0 },
      {    -0.2,        58.5,    0.0,      0.2 },
      {     0.1,       -94.9,    0.0,      0.2 },
      {     0.0,        52.9,    0.0,     -0.2 },
      {     0.1,        86.7,    0.0,     -0.2 },
      {    -0.1,       -59.2,    0.0,      0.2 },
      {     0.3,       -58.8,    0.0,      0.1 },
      {    -0.3,        49.0,    0.0,      0.0 },
      {    -0.2,        56.9,    0.0,     -0.1 },
      {     0.3,       -50.2,    0.0,      0.0 },
      {    -0.2,        53.4,    0.0,     -0.1 },
      {     0.1,       -76.5,    0.0,      0.0 },
      {    -0.2,        45.3,    0.0,      0.0 },
      {     0.1,       -46.8,    0.0,      0.0 },
      {     0.2,       -44.6,    0.0,      0.0 },
      {     0.2,       -48.7,    0.0,      0.0 },
      {     0.1,       -46.8,    0.0,      0.0 },
      {     0.1,       -42.0,    0.0,      0.0 },
      {     0.0,        46.4,    0.0,     -0.1 },
      {     0.2,       -67.3,    0.0,      0.1 },
      {     0.0,       -65.8,    0.0,      0.2 },
      {    -0.1,       -43.9,    0.0,      0.3 },
      {     0.0,       -38.9,    0.0,      0.0 },
      {    -0.3,        63.9,    0.0,      0.0 },
      {    -0.2,        41.2,    0.0,      0.0 },
      {     0.0,       -36.1,    0.0,      0.2 },
      {    -0.3,        58.5,    0.0,      0.0 },
      {    -0.1,        36.1,    0.0,      0.0 },
      {     0.0,       -39.7,    0.0,      0.0 },
      {     0.1,       -57.7,    0.0,      0.0 },
      {    -0.2,        33.4,    0.0,      0.0 },
      {    36.4,         0.0,    0.0,      0.0 },
      {    -0.1,        55.7,    0.0,     -0.1 },
      {     0.1,       -35.4,    0.0,      0.0 },
      {     0.1,       -31.0,    0.0,      0.0 },
      {    -0.1,        30.1,    0.0,      0.0 },
      {    -0.3,        49.2,    0.0,      0.0 },
      {    -0.2,        49.1,    0.0,      0.0 },
      {    -0.1,        33.6,    0.0,      0.0 },
      {     0.1,       -33.5,    0.0,      0.0 },
      {     0.1,       -31.0,    0.0,      0.0 },
      {    -0.1,        28.0,    0.0,      0.0 },
      {     0.1,       -25.2,    0.0,      0.0 },
      {     0.1,       -26.2,    0.0,      0.0 },
      {    -0.2,        41.5,    0.0,      0.0 },
      {     0.0,        24.5,    0.0,      0.1 },
      {   -16.2,         0.0,    0.0,      0.0 },
      {     0.0,       -22.3,    0.0,      0.0 },
      {     0.0,        23.1,    0.0,      0.0 },
      {    -0.1,        37.5,    0.0,      0.0 },
      {     0.2,       -25.7,    0.0,      0.0 },
      {     0.0,        25.2,    0.0,      0.0 },
      {     0.1,       -24.5,    0.0,      0.0 },
      {    -0.1,        24.3,    0.0,      0.0 },
      {     0.1,       -20.7,    0.0,      0.0 },
      {     0.1,       -20.8,    0.0,      0.0 },
      {    -0.2,        33.4,    0.0,      0.0 },
      {    32.9,         0.0,    0.0,      0.0 },
      {     0.1,       -32.6,    0.0,      0.0 },
      {     0.0,        19.9,    0.0,      0.0 },
      {    -0.1,        19.6,    0.0,      0.0 },
      {     0.0,       -18.7,    0.0,      0.0 },
      {     0.1,       -19.0,    0.0,      0.0 },
      {     0.1,       -28.6,    0.0,      0.0 },
      {     4.0,       178.8,  -11.8,      0.3 },
      {    39.8,      -107.3,   -5.6,     -1.0 },
      {     9.9,       164.0,   -4.1,      0.1 },
      {    -4.8,      -135.3,   -3.4,     -0.1 },
      {    50.5,        75.0,    1.4,     -1.2 },
      {    -1.1,       -53.5,    1.3,      0.0 },
      {   -45.0,        -2.4,   -0.4,      6.6 },
      {   -11.5,       -61.0,   -0.9,      0.4 },
      {     4.4,       -68.4,   -3.4,      0.0 },
      {     7.7,       -47.1,   -4.7,     -1.0 },
      {   -42.9,       -12.6,   -1.2,      4.2 },
      {   -42.8,        12.7,   -1.2,     -4.2 },
      {    -7.6,       -44.1,    2.1,     -0.5 },
      {   -64.1,         1.7,    0.2,      4.5 },
      {    36.4,       -10.4,    1.0,      3.5 },
      {    35.6,        10.2,    1.0,     -3.5 },
      {    -1.7,        39.5,    2.0,      0.0 },
      {    50.9,        -8.2,   -0.8,     -5.0 },
      {     0.0,        52.3,    1.2,      0.0 },
      {   -42.9,       -17.8,    0.4,      0.0 },
      {     2.6,        34.3,    0.8,      0.0 },
      {    -0.8,       -48.6,    2.4,     -0.1 },
      {    -4.9,        30.5,    3.7,      0.7 },
      {     0.0,       -43.6,    2.1,      0.0 },
      {     0.0,       -25.4,    1.2,      0.0 },
      {     2.0,        40.9,   -2.0,      0.0 },
      {    -2.1,        26.1,    0.6,      0.0 },
      {    22.6,        -3.2,   -0.5,     -0.5 },
      {    -7.6,        24.9,   -0.4,     -0.2 },
      {    -6.2,        34.9,    1.7,      0.3 },
      {     2.0,        17.4,   -0.4,      0.1 },
      {    -3.9,        20.5,    2.4,      0.6 }
   };

/* Nutation series:  obliquity */
   static double eps[][4] = {
      { 9205365.8,  -1506.2,   885.7,  -0.2 },
      {  573095.9,   -570.2,  -305.0,  -0.3 },
      {   97845.5,    147.8,   -48.8,  -0.2 },
      {  -89753.6,     28.0,    46.9,   0.0 },
      {    7406.7,   -327.1,   -18.2,   0.8 },
      {   22442.3,    -22.3,   -67.6,   0.0 },
      {    -683.6,     46.8,     0.0,   0.0 },
      {   20070.7,     36.0,     1.6,   0.0 },
      {   12893.8,     39.5,    -6.2,   0.0 },
      {   -9593.2,     14.4,    30.2,  -0.1 },
      {   -6899.5,      4.8,    -0.6,   0.0 },
      {   -5332.5,     -0.1,     2.7,   0.0 },
      {    -125.2,     10.5,     0.0,   0.0 },
      {   -3323.4,     -0.9,    -0.3,   0.0 },
      {    3142.3,      8.9,     0.3,   0.0 },
      {    2552.5,      7.3,    -1.2,   0.0 },
      {    2634.4,      8.8,     0.2,   0.0 },
      {   -2424.4,      1.6,    -0.4,   0.0 },
      {    -123.3,      3.9,     0.0,   0.0 },
      {    1642.4,      7.3,    -0.8,   0.0 },
      {      47.9,      3.2,     0.0,   0.0 },
      {    1321.2,      6.2,    -0.6,   0.0 },
      {   -1234.1,     -0.3,     0.6,   0.0 },
      {   -1076.5,     -0.3,     0.0,   0.0 },
      {     -61.6,      1.8,     0.0,   0.0 },
      {     -55.4,      1.6,     0.0,   0.0 },
      {     856.9,     -4.9,    -2.1,   0.0 },
      {    -800.7,     -0.1,     0.0,   0.0 },
      {     685.1,     -0.6,    -3.8,   0.0 },
      {     -16.9,     -1.5,     0.0,   0.0 },
      {     695.7,      1.8,     0.0,   0.0 },
      {     642.2,     -2.6,    -1.6,   0.0 },
      {      13.3,      1.1,    -0.1,   0.0 },
      {     521.9,      1.6,     0.0,   0.0 },
      {     325.8,      2.0,    -0.1,   0.0 },
      {    -325.1,     -0.5,     0.9,   0.0 },
      {      10.1,      0.3,     0.0,   0.0 },
      {     334.5,      1.6,     0.0,   0.0 },
      {     307.1,      0.4,    -0.9,   0.0 },
      {     327.2,      0.5,     0.0,   0.0 },
      {    -304.6,     -0.1,     0.0,   0.0 },
      {     304.0,      0.6,     0.0,   0.0 },
      {    -276.8,     -0.5,     0.1,   0.0 },
      {     268.9,      1.3,     0.0,   0.0 },
      {     271.8,      1.1,     0.0,   0.0 },
      {     271.5,     -0.4,    -0.8,   0.0 },
      {      -5.2,      0.5,     0.0,   0.0 },
      {    -220.5,      0.1,     0.0,   0.0 },
      {     -20.1,      0.3,     0.0,   0.0 },
      {    -191.0,      0.1,     0.5,   0.0 },
      {      -4.1,      0.3,     0.0,   0.0 },
      {     130.6,     -0.1,     0.0,   0.0 },
      {       3.0,      0.3,     0.0,   0.0 },
      {     122.9,      0.8,     0.0,   0.0 },
      {       3.7,     -0.3,     0.0,   0.0 },
      {     123.1,      0.4,    -0.3,   0.0 },
      {     -52.7,     15.3,     0.0,   0.0 },
      {     120.7,      0.3,    -0.3,   0.0 },
      {       4.0,     -0.3,     0.0,   0.0 },
      {     126.5,      0.5,     0.0,   0.0 },
      {     112.7,      0.5,    -0.3,   0.0 },
      {    -106.1,     -0.3,     0.3,   0.0 },
      {    -112.9,     -0.2,     0.0,   0.0 },
      {       3.6,     -0.2,     0.0,   0.0 },
      {     107.4,      0.3,     0.0,   0.0 },
      {     -10.9,      0.2,     0.0,   0.0 },
      {      -0.9,      0.0,     0.0,   0.0 },
      {      85.4,      0.0,     0.0,   0.0 },
      {       0.0,    -88.8,     0.0,   0.0 },
      {     -71.0,     -0.2,     0.0,   0.0 },
      {     -70.3,      0.0,     0.0,   0.0 },
      {      64.5,      0.4,     0.0,   0.0 },
      {      69.8,      0.0,     0.0,   0.0 },
      {      66.1,      0.4,     0.0,   0.0 },
      {     -61.0,     -0.2,     0.0,   0.0 },
      {     -59.5,     -0.1,     0.0,   0.0 },
      {     -55.6,      0.0,     0.2,   0.0 },
      {      51.7,      0.2,     0.0,   0.0 },
      {     -49.0,     -0.1,     0.0,   0.0 },
      {     -52.7,     -0.1,     0.0,   0.0 },
      {     -49.6,      1.4,     0.0,   0.0 },
      {      46.3,      0.4,     0.0,   0.0 },
      {      49.6,      0.1,     0.0,   0.0 },
      {      -5.1,      0.1,     0.0,   0.0 },
      {     -44.0,     -0.1,     0.0,   0.0 },
      {     -39.9,     -0.1,     0.0,   0.0 },
      {     -39.5,     -0.1,     0.0,   0.0 },
      {      -3.9,      0.1,     0.0,   0.0 },
      {     -42.1,     -0.1,     0.0,   0.0 },
      {     -17.2,      0.1,     0.0,   0.0 },
      {      -2.3,      0.1,     0.0,   0.0 },
      {     -39.2,      0.0,     0.0,   0.0 },
      {     -38.4,      0.1,     0.0,   0.0 },
      {      36.8,      0.2,     0.0,   0.0 },
      {      34.6,      0.1,     0.0,   0.0 },
      {     -32.7,      0.3,     0.0,   0.0 },
      {      30.4,      0.0,     0.0,   0.0 },
      {       0.4,      0.1,     0.0,   0.0 },
      {      29.3,      0.2,     0.0,   0.0 },
      {      31.6,      0.1,     0.0,   0.0 },
      {       0.8,     -0.1,     0.0,   0.0 },
      {     -27.9,      0.0,     0.0,   0.0 },
      {       2.9,      0.0,     0.0,   0.0 },
      {     -25.3,      0.0,     0.0,   0.0 },
      {      25.0,      0.1,     0.0,   0.0 },
      {      27.5,      0.1,     0.0,   0.0 },
      {     -24.4,     -0.1,     0.0,   0.0 },
      {      24.9,      0.2,     0.0,   0.0 },
      {     -22.8,     -0.1,     0.0,   0.0 },
      {       0.9,     -0.1,     0.0,   0.0 },
      {      24.4,      0.1,     0.0,   0.0 },
      {      23.9,      0.1,     0.0,   0.0 },
      {      22.5,      0.1,     0.0,   0.0 },
      {      20.8,      0.1,     0.0,   0.0 },
      {      20.1,      0.0,     0.0,   0.0 },
      {      21.5,      0.1,     0.0,   0.0 },
      {     -20.0,      0.0,     0.0,   0.0 },
      {       1.4,      0.0,     0.0,   0.0 },
      {      -0.2,     -0.1,     0.0,   0.0 },
      {      19.0,      0.0,    -0.1,   0.0 },
      {      20.5,      0.0,     0.0,   0.0 },
      {      -2.0,      0.0,     0.0,   0.0 },
      {     -17.6,     -0.1,     0.0,   0.0 },
      {      19.0,      0.0,     0.0,   0.0 },
      {      -2.4,      0.0,     0.0,   0.0 },
      {     -18.4,     -0.1,     0.0,   0.0 },
      {      17.1,      0.0,     0.0,   0.0 },
      {       0.4,      0.0,     0.0,   0.0 },
      {      18.4,      0.1,     0.0,   0.0 },
      {       0.0,     17.4,     0.0,   0.0 },
      {      -0.6,      0.0,     0.0,   0.0 },
      {     -15.4,      0.0,     0.0,   0.0 },
      {     -16.8,     -0.1,     0.0,   0.0 },
      {      16.3,      0.0,     0.0,   0.0 },
      {      -2.0,      0.0,     0.0,   0.0 },
      {      -1.5,      0.0,     0.0,   0.0 },
      {     -14.3,     -0.1,     0.0,   0.0 },
      {      14.4,      0.0,     0.0,   0.0 },
      {     -13.4,      0.0,     0.0,   0.0 },
      {     -14.3,     -0.1,     0.0,   0.0 },
      {     -13.7,      0.0,     0.0,   0.0 },
      {      13.1,      0.1,     0.0,   0.0 },
      {      -1.7,      0.0,     0.0,   0.0 },
      {     -12.8,      0.0,     0.0,   0.0 },
      {       0.0,    -14.4,     0.0,   0.0 },
      {      12.4,      0.0,     0.0,   0.0 },
      {     -12.0,      0.0,     0.0,   0.0 },
      {      -0.8,      0.0,     0.0,   0.0 },
      {      10.9,      0.1,     0.0,   0.0 },
      {     -10.8,      0.0,     0.0,   0.0 },
      {      10.5,      0.0,     0.0,   0.0 },
      {     -10.4,      0.0,     0.0,   0.0 },
      {     -11.2,      0.0,     0.0,   0.0 },
      {      10.5,      0.1,     0.0,   0.0 },
      {      -1.4,      0.0,     0.0,   0.0 },
      {       0.0,      0.1,     0.0,   0.0 },
      {       0.7,      0.0,     0.0,   0.0 },
      {     -10.3,      0.0,     0.0,   0.0 },
      {     -10.0,      0.0,     0.0,   0.0 },
      {       9.6,      0.0,     0.0,   0.0 },
      {       9.4,      0.1,     0.0,   0.0 },
      {       0.6,      0.0,     0.0,   0.0 },
      {     -87.7,      4.4,    -0.4,  -6.3 },
      {      46.3,     22.4,     0.5,  -2.4 },
      {      15.6,     -3.4,     0.1,   0.4 },
      {       5.2,      5.8,     0.2,  -0.1 },
      {     -30.1,     26.9,     0.7,   0.0 },
      {      23.2,     -0.5,     0.0,   0.6 },
      {       1.0,     23.2,     3.4,   0.0 },
      {     -12.2,     -4.3,     0.0,   0.0 },
      {      -2.1,     -3.7,    -0.2,   0.1 },
      {     -18.6,     -3.8,    -0.4,   1.8 },
      {       5.5,    -18.7,    -1.8,  -0.5 },
      {      -5.5,    -18.7,     1.8,  -0.5 },
      {      18.4,     -3.6,     0.3,   0.9 },
      {      -0.6,      1.3,     0.0,   0.0 },
      {      -5.6,    -19.5,     1.9,   0.0 },
      {       5.5,    -19.1,    -1.9,   0.0 },
      {     -17.3,     -0.8,     0.0,   0.9 },
      {      -3.2,     -8.3,    -0.8,   0.3 },
      {      -0.1,      0.0,     0.0,   0.0 },
      {      -5.4,      7.8,    -0.3,   0.0 },
      {     -14.8,      1.4,     0.0,   0.3 },
      {      -3.8,      0.4,     0.0,  -0.2 },
      {      12.6,      3.2,     0.5,  -1.5 },
      {       0.1,      0.0,     0.0,   0.0 },
      {     -13.6,      2.4,    -0.1,   0.0 },
      {       0.9,      1.2,     0.0,   0.0 },
      {     -11.9,     -0.5,     0.0,   0.3 },
      {       0.4,     12.0,     0.3,  -0.2 },
      {       8.3,      6.1,    -0.1,   0.1 },
      {       0.0,      0.0,     0.0,   0.0 },
      {       0.4,    -10.8,     0.3,   0.0 },
      {       9.6,      2.2,     0.3,  -1.2 }
   };

/* Number of terms in the model */
   static int nterms = sizeof ( na ) / sizeof ( int ) /9;

   int j;
   double t, el, elp, f, d, om, ve, ma, ju, sa, theta, c, s, dp, de;



/* Interval between fundamental epoch J2000.0 and given epoch (JC). */
   t  =  ( date - DJM0 ) / DJC;

/* Mean anomaly of the Moon. */
   el  = 134.96340251 * DD2R +
         fmod_ ( t * ( 1717915923.2178 +
                t * (         31.8792 +
                t * (          0.051635 +
                t * (        - 0.00024470 ) ) ) ), TURNAS ) * DAS2R;

/* Mean anomaly of the Sun. */
   elp = 357.52910918 * DD2R +
         fmod_ ( t * (  129596581.0481 +
                t * (        - 0.5532 +
                t * (          0.000136 +
                t * (        - 0.00001149 ) ) ) ), TURNAS ) * DAS2R;

/* Mean argument of the latitude of the Moon. */
   f   =  93.27209062 * DD2R +
         fmod_ ( t * ( 1739527262.8478 +
                t * (       - 12.7512 +
                t * (        - 0.001037 +
                t * (          0.00000417 ) ) ) ), TURNAS ) * DAS2R;

/* Mean elongation of the Moon from the Sun. */
   d   = 297.85019547 * DD2R +
         fmod_ ( t * ( 1602961601.2090 +
                t * (        - 6.3706 +
                t * (          0.006539 +
                t * (        - 0.00003169 ) ) ) ), TURNAS ) * DAS2R;

/* Mean longitude of the ascending node of the Moon. */
   om  = 125.04455501 * DD2R +
         fmod_ ( t * (  - 6962890.5431 +
                t * (          7.4722 +
                t * (          0.007702 +
                t * (        - 0.00005939 ) ) ) ), TURNAS ) * DAS2R;

/* Mean longitude of Venus. */
   ve  = 181.97980085 * DD2R +
         fmod_ ( 210664136.433548 * t, TURNAS ) * DAS2R;

/* Mean longitude of Mars. */
   ma  = 355.43299958 * DD2R +
         fmod_ (  68905077.493988 * t, TURNAS ) * DAS2R;

/* Mean longitude of Jupiter. */
   ju  =  34.35151874 * DD2R +
         fmod_ (  10925660.377991 * t, TURNAS ) * DAS2R;

/* Mean longitude of Saturn. */
   sa  =  50.07744430 * DD2R +
         fmod_ (   4399609.855732 * t, TURNAS ) * DAS2R;

/* Geodesic nutation (Fukushima 1991) in microarcsec. */
   dp = - 153.1 * sin_ ( elp ) - 1.9 * sin_( 2.0 * elp );
   de = 0.0;

/* Shirai & Fukushima (2001) nutation series. */
   for ( j = nterms - 1; j >= 0; j-- ) {
      theta = ( (double) na[j][0] ) * el +
              ( (double) na[j][1] ) * elp +
              ( (double) na[j][2] ) * f +
              ( (double) na[j][3] ) * d +
              ( (double) na[j][4] ) * om +
              ( (double) na[j][5] ) * ve +
              ( (double) na[j][6] ) * ma +
              ( (double) na[j][7] ) * ju +
              ( (double) na[j][8] ) * sa;
      c = cos_ ( theta );
      s = sin_ ( theta );
      dp += ( psi[j][0] + psi[j][2] * t ) * c +
              ( psi[j][1] + psi[j][3] * t ) * s;
      de += ( eps[j][0] + eps[j][2] * t ) * c +
              ( eps[j][1] + eps[j][3] * t ) * s;
   }

/* Change of units, and addition of the precession correction. */
   *dpsi = ( dp * 1e-6 - 0.042888 - 0.29856 * t ) * DAS2R;
   *deps = ( de * 1e-6 - 0.005171 - 0.02408 * t ) * DAS2R;

/* Mean obliquity of date (Simon et al. 1994). */
   *eps0  =  ( 84381.412 +
              ( - 46.80927 +
               ( - 0.000152 +
                 ( 0.0019989 +
               ( - 0.00000051 +
               ( - 0.000000025 ) * t ) * t ) * t ) * t ) * t ) * DAS2R;

}
