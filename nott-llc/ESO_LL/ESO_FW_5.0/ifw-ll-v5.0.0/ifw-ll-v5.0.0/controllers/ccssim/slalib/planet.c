#include "TcPch.h"
#pragma hdrstop

#include "slalib.h"
#include "slamac.h"
void slaPlanet ( double date, int np, double pv[6], int *jstat )
/*
**  - - - - - - - - - -
**   s l a P l a n e t
**  - - - - - - - - - -
**
**  Approximate heliocentric position and velocity of a specified
**  planet.
**
**  Given:
**     date     double      TDB (loosely ET) as a Modified Julian Date
**                                                  (JD-2400000.5)
**     np       int         body (1=Mercury, 2=Venus, 3=EMB,...9=Pluto)
**
**  Returned:
**     pv       double[6]   heliocentric x,y,z,xdot,ydot,zdot, J2000
**                                           equatorial triad (AU,AU/s)
**
**     *jstat   int         status: +1 = warning: date outside 1000-3000
**     *jstat   int         status:  0 = OK
**                                  -1 = illegal NP (outside 1-9)
**                                  -2 = solution didn't converge
**
**  Called:  slaPlanel
**
**  Notes
**
**  1  The epoch, date, is in the TDB timescale and is a Modified
**     Julian Date (JD-2400000.5).
**
**  2  The reference frame is equatorial and is with respect to the
**     mean equinox and ecliptic of epoch J2000.
**
**  3  If an np value outside the range 1-9 is supplied, an error
**     status (jstat = -1) is returned and the pv vector set to zeroes.
**
**  4  The algorithm for obtaining the mean elements of the planets
**     (Mercury to Neptune) is due to J.L. Simon, P. Bretagnon,
**     J. Chapront, M. Chapront-Touze, G. Francou and J. Laskar
**     (Bureau des Longitudes, Paris).  The (completely different)
**     algorithm for calculating the ecliptic coordinates of the dwarf
**     planet Pluto is by Meeus.
**
**  5  Comparisons of the present function with the JPL DE200 ephemeris
**     give the following RMS errors over the interval 1960-2025:
**
**                      position (km)     speed (metre/sec)
**
**        Mercury            334               0.437
**        Venus             1060               0.855
**        EMB               2010               0.815
**        Mars              7690               1.98
**        Jupiter          71700               7.70
**        Saturn          199000              19.4
**        Uranus          564000              16.4
**        Neptune         158000              14.4
**
**     From comparisons with DE102, Simon et al quote the following
**     longitude accuracies over the interval 1800-2200:
**
**        Mercury                 4"
**        Venus                   5"
**        EMB                     6"
**        Mars                   17"
**        Jupiter                71"
**        Saturn                 81"
**        Uranus                 86"
**        Neptune                11"
**
**     In the case of Pluto, Meeus quotes an accuracy of 0.6 arcsec
**     in longitude and 0.2 arcsec in latitude for the period
**     1885-2099.
**
**     For all except Pluto, over the period 1000-3000 the accuracy
**     is better than 1.5 times that over 1800-2200.  Outside the
**     period 1000-3000 the accuracy declines.  For Pluto the
**     accuracy declines rapidly outside the period 1885-2099.
**     Outside these ranges (1885-2099 for Pluto, 1000-3000 for
**     the rest) a "date out of range" warning status (JSTAT=+1)
**     is returned.
**
**  6  The algorithms for (i) Mercury through Neptune and (ii) Pluto
**     are completely independent.  In the Mercury through Neptune
**     case, the present SLALIB C implementation follows the original
**     Simon et al Fortran code closely, and delivers essentially
**     the same results.  The changes are these:
**
**     *  The date is supplied as a Modified Julian Date rather
**        than a Julian Date (MJD = JD - 2400000.5).
**
**     *  The result is returned only in equatorial Cartesian form;
**        the ecliptic longitude, latitude and radius vector are not
**        returned.
**
**     *  The velocity is in AU per second, not AU per day.
**
**     *  Different error/warning status values are used.
**
**     *  Kepler's equation is not solved inline.
**
**     *  Polynomials in T are nested to minimize rounding errors.
**
**     *  Explicit double-precision constants are used to avoid
**        mixed-mode expressions.
**
**  7  For np=3 the result is for the Earth-Moon Barycentre.  To
**     obtain the heliocentric position and velocity of the Earth,
**     either use the SLALIB function slaEvp (or slaEpv) or use slaDmoon
**     and subtract 0.012150581 times the geocentric Moon vector from
**     the EMB vector produced by the present function.  (The Moon
**     vector should be precessed to J2000 first, but this can
**     be omitted for modern epochs without introducing significant
**     inaccuracy.)
**
**  References:  Simon et al., Astron. Astrophys. 282, 663 (1994).
**               Meeus, Astronomical Algorithms, Willmann-Bell (1991).
**
**  Defined in slamac.h:  D2PI, DAS2R, DD2R, dmod
**
**  Last revision:   10 February 2009
**
**  Copyright P.T.Wallace.  All rights reserved.
*/

/* Gaussian gravitational constant (exact) */
#define GCON 0.01720209895

/* Canonical days to seconds */
#define CD2S ( GCON / 86400.0 )

/* Seconds per Julian century */
#define SPC ( 36525.0 * 86400.0 )

/* Sin and cos of J2000 mean obliquity (IAU 1976) */
#define SE 0.3977771559319137
#define CE 0.9174820620691818

{
   int ip, i, j;
   double t, da, de, dpe, di, dom, dmu, arga, argl, dm,
          dj, ds, dp, wlbr[3], wlbrd[3],
          wj, ws, wp, al, ald, sal, cal,
          ac, bc, dl, dld, db, dbd, dr, drd,
          sl, cl, sb, cb, slcb, clcb, x, y, z, xd, yd, zd;

/*
** -----------------------
** Mercury through Neptune
** -----------------------
*/

/* Planetary inverse masses */
   static double amas[] = {
      6023600.0,
       408523.5,
       328900.5,
      3098710.0,
       1047.355,
         3498.5,
        22869.0,
        19314.0
   };

/*
**    Tables giving the mean Keplerian elements, limited to T^2 terms:
**
**    a       semi-major axis (AU)
**    dlm     mean longitude (degree and arcsecond)
**    e       eccentricity
**    pi      longitude of the perihelion (degree and arcsecond)
**    dinc    inclination (degree and arcsecond)
**    omega   longitude of the ascending node (degree and arcsecond)
*/
   static double a[8][3] = {
      {  0.3870983098,           0.0,     0.0 },
      {  0.7233298200,           0.0,     0.0 },
      {  1.0000010178,           0.0,     0.0 },
      {  1.5236793419,         3e-10,     0.0 },
      {  5.2026032092,     19132e-10, -39e-10 },
      {  9.5549091915, -0.0000213896, 444e-10 },
      { 19.2184460618,     -3716e-10, 979e-10 },
      { 30.1103868694,    -16635e-10, 686e-10 }
   };
   static double dlm[8][3] = {
      { 252.25090552, 5381016286.88982,  -1.92789 },
      { 181.97980085, 2106641364.33548,   0.59381 },
      { 100.46645683, 1295977422.83429,  -2.04411 },
      { 355.43299958,  689050774.93988,   0.94264 },
      {  34.35151874,  109256603.77991, -30.60378 },
      {  50.07744430,   43996098.55732,  75.61614 },
      { 314.05500511,   15424811.93933,  -1.75083 },
      { 304.34866548,    7865503.20744,   0.21103 }
   };
   static double e[8][3] = {
      { 0.2056317526,  0.0002040653,      -28349e-10 },
      { 0.0067719164, -0.0004776521,       98127e-10 },
      { 0.0167086342, -0.0004203654,   -0.0000126734 },
      { 0.0934006477,  0.0009048438,      -80641e-10 },
      { 0.0484979255,  0.0016322542,   -0.0000471366 },
      { 0.0555481426, -0.0034664062,   -0.0000643639 },
      { 0.0463812221, -0.0002729293,    0.0000078913 },
      { 0.0094557470,  0.0000603263,             0.0 }
   };
   static double pi[8][3] = {
      {  77.45611904,  5719.11590,   -4.83016 },
      { 131.56370300,   175.48640, -498.48184 },
      { 102.93734808, 11612.35290,   53.27577 },
      { 336.06023395, 15980.45908,  -62.32800 },
      {  14.33120687,  7758.75163,  259.95938 },
      {  93.05723748, 20395.49439,  190.25952 },
      { 173.00529106,  3215.56238,  -34.09288 },
      {  48.12027554,  1050.71912,   27.39717 }
   };
   static double dinc[8][3] = {
      { 7.00498625, -214.25629,   0.28977 },
      { 3.39466189,  -30.84437, -11.67836 },
      {        0.0,  469.97289,  -3.35053 },
      { 1.84972648, -293.31722,  -8.11830 },
      { 1.30326698,  -71.55890,  11.95297 },
      { 2.48887878,   91.85195, -17.66225 },
      { 0.77319689,  -60.72723,   1.25759 },
      { 1.76995259,    8.12333,   0.08135 }
   };
   static double omega[8][3] = {
      {  48.33089304,  -4515.21727,  -31.79892 },
      {  76.67992019, -10008.48154,  -51.32614 },
      { 174.87317577,  -8679.27034,   15.34191 },
      {  49.55809321, -10620.90088, -230.57416 },
      { 100.46440702,   6362.03561,  326.52178 },
      { 113.66550252,  -9240.19942,  -66.23743 },
      {  74.00595701,   2669.15033,  145.93964 },
      { 131.78405702,   -221.94322,   -0.78728 }
   };

/*
**    Tables for trigonometric terms to be added to the mean elements
**    of the semi-major axes.
*/
   static double dkp[8][9] = {
      { 69613.0, 75645.0, 88306.0, 59899.0, 15746.0, 71087.0,
                                                142173.0,  3086.0,    0.0 },
      { 21863.0, 32794.0, 26934.0, 10931.0, 26250.0, 43725.0,
                                                 53867.0, 28939.0,    0.0 },
      { 16002.0, 21863.0, 32004.0, 10931.0, 14529.0, 16368.0,
                                                 15318.0, 32794.0,    0.0 },
      {  6345.0,  7818.0, 15636.0,  7077.0,  8184.0, 14163.0,
                                                  1107.0,  4872.0,    0.0 },
      {  1760.0,  1454.0,  1167.0,   880.0,   287.0,  2640.0,
                                                    19.0,  2047.0, 1454.0 },
      {   574.0,     0.0,   880.0,   287.0,    19.0,  1760.0,
                                                  1167.0,   306.0,  574.0 },
      {   204.0,     0.0,   177.0,  1265.0,     4.0,   385.0,
                                                   200.0,   208.0,  204.0 },
      {     0.0,   102.0,   106.0,     4.0,    98.0,  1367.0,
                                                   487.0,   204.0,    0.0 }
   };
   static double ca[8][9] = {
    {       4.0,    -13.0,    11.0,    -9.0,    -9.0,    -3.0,
                                                    -1.0,     4.0,    0.0 },
    {    -156.0,     59.0,   -42.0,     6.0,    19.0,   -20.0,
                                                   -10.0,   -12.0,    0.0 },
    {      64.0,   -152.0,    62.0,    -8.0,    32.0,   -41.0,
                                                    19.0,   -11.0,    0.0 },
    {     124.0,    621.0,  -145.0,   208.0,    54.0,   -57.0,
                                                    30.0,    15.0,    0.0 },
    {  -23437.0,  -2634.0,  6601.0,  6259.0, -1507.0, -1821.0,
                                                  2620.0, -2115.0,-1489.0 },
    {   62911.0,-119919.0, 79336.0, 17814.0,-24241.0, 12068.0,
                                                  8306.0, -4893.0, 8902.0 },
    {  389061.0,-262125.0,-44088.0,  8387.0,-22976.0, -2093.0,
                                                  -615.0, -9720.0, 6633.0 },
    { -412235.0,-157046.0,-31430.0, 37817.0, -9740.0,   -13.0,
                                                 -7449.0,  9644.0,    0.0 }
   };
   static double sa[8][9] = {
      {     -29.0,    -1.0,     9.0,     6.0,    -6.0,     5.0,
                                                     4.0,     0.0,    0.0 },
      {     -48.0,  -125.0,   -26.0,   -37.0,    18.0,   -13.0,
                                                   -20.0,    -2.0,    0.0 },
      {    -150.0,   -46.0,    68.0,    54.0,    14.0,    24.0,
                                                   -28.0,    22.0,    0.0 },
      {    -621.0,   532.0,  -694.0,   -20.0,   192.0,   -94.0,
                                                    71.0,   -73.0,    0.0 },
      {  -14614.0,-19828.0, -5869.0,  1881.0, -4372.0, -2255.0,
                                                   782.0,   930.0,  913.0 },
      {  139737.0,     0.0, 24667.0, 51123.0, -5102.0,  7429.0,
                                                 -4095.0, -1976.0,-9566.0 },
      { -138081.0,     0.0, 37205.0,-49039.0,-41901.0,-33872.0,
                                                -27037.0,-12474.0,18797.0 },
      {       0.0, 28492.0,133236.0, 69654.0, 52322.0,-49577.0,
                                                -26430.0, -3593.0,    0.0 }
   };

/*
**    Tables giving the trigonometric terms to be added to the mean
**    elements of the mean longitudes.
*/
   static double dkq[8][10] = {
      {  3086.0, 15746.0, 69613.0, 59899.0, 75645.0,
                                      88306.0, 12661.0, 2658.0,  0.0,   0.0 },
      { 21863.0, 32794.0, 10931.0,    73.0,  4387.0,
                                      26934.0,  1473.0, 2157.0,  0.0,   0.0 },
      {    10.0, 16002.0, 21863.0, 10931.0,  1473.0,
                                      32004.0,  4387.0,   73.0,  0.0,   0.0 },
      {    10.0,  6345.0,  7818.0,  1107.0, 15636.0,
                                       7077.0,  8184.0,  532.0, 10.0,   0.0 },
      {    19.0,  1760.0,  1454.0,   287.0,  1167.0,
                                        880.0,   574.0, 2640.0, 19.0,1454.0 },
      {    19.0,   574.0,   287.0,   306.0,  1760.0,
                                         12.0,    31.0,   38.0, 19.0, 574.0 },
      {     4.0,   204.0,   177.0,     8.0,    31.0,
                                        200.0,  1265.0,  102.0,  4.0, 204.0 },
      {     4.0,   102.0,   106.0,     8.0,    98.0,
                                       1367.0,   487.0,  204.0,  4.0, 102.0 }
   };
   static double clo[8][10] = {
    {      21.0,    -95.0,  -157.0,    41.0,    -5.0,
                                      42.0,   23.0,   30.0,     0.0,    0.0 },
    {    -160.0,   -313.0,  -235.0,    60.0,   -74.0,
                                     -76.0,  -27.0,   34.0,     0.0,    0.0 },
    {    -325.0,   -322.0,   -79.0,   232.0,   -52.0,
                                      97.0,   55.0,  -41.0,     0.0,    0.0 },
    {    2268.0,   -979.0,   802.0,   602.0,  -668.0,
                                     -33.0,  345.0,  201.0,   -55.0,    0.0 },
    {    7610.0,  -4997.0, -7689.0, -5841.0, -2617.0,
                                    1115.0, -748.0, -607.0,  6074.0,  354.0 },
    {  -18549.0,  30125.0, 20012.0,  -730.0,   824.0,
                                      23.0, 1289.0, -352.0,-14767.0,-2062.0 },
    { -135245.0, -14594.0,  4197.0, -4030.0, -5630.0,
                                   -2898.0, 2540.0, -306.0,  2939.0, 1986.0 },
    {   89948.0,   2103.0,  8963.0,  2695.0,  3682.0,
                                    1648.0,  866.0, -154.0, -1963.0, -283.0 }
   };
   static double slo[8][10] = {
    {   -342.0,    136.0,   -23.0,    62.0,    66.0,
                                 -52.0,   -33.0,    17.0,     0.0,     0.0 },
    {    524.0,   -149.0,   -35.0,   117.0,   151.0,
                                 122.0,   -71.0,   -62.0,     0.0,     0.0 },
    {   -105.0,   -137.0,   258.0,    35.0,  -116.0,
                                 -88.0,  -112.0,   -80.0,     0.0,     0.0 },
    {    854.0,   -205.0,  -936.0,  -240.0,   140.0,
                                -341.0,   -97.0,  -232.0,   536.0,     0.0 },
    { -56980.0,   8016.0,  1012.0,  1448.0, -3024.0,
                               -3710.0,   318.0,   503.0,  3767.0,   577.0 },
    { 138606.0, -13478.0, -4964.0,  1441.0, -1319.0,
                               -1482.0,   427.0,  1236.0, -9167.0, -1918.0 },
    {  71234.0, -41116.0,  5334.0, -4935.0, -1848.0,
                                  66.0,   434.0, -1748.0,  3780.0,  -701.0 },
    { -47645.0,  11647.0,  2166.0,  3194.0,   679.0,
                                   0.0,  -244.0,  -419.0, -2531.0,    48.0 }
   };

/*
** -----
** Pluto
** -----
*/

/*
** Coefficients for fundamental arguments:  mean longitudes (degrees)
** and mean rate of change of longitude (degrees per Julian century)
** for Jupiter, Saturn and Pluto
*/
   static double dj0 = 34.35, djd = 3034.9057,
                 ds0 = 50.08, dsd = 1222.1138,
                 dp0 = 238.96, dpd = 144.9600;

/* Coefficients for latitude, longitude, radius vector */
   static double dl0 = 238.956785, dld0 = 144.96,
                 db0 = -3.908202,
                 dr0 = 40.7247248;

/*
** Coefficients for periodic terms (Meeus's Table 36.A)
*/
   struct ab {
      double a;           /* sine component */
      double b;           /* cosine component */
   };
   struct tm {
      int ij;             /* Jupiter contribution to argument */
      int is;             /* Saturn contribution to argument */
      int ip;             /* Pluto contribution to argument */
      struct ab dlbr[3];  /* longitude (degrees),
                             latitude (degrees),
                             radius vector (AU) */
   };
   static struct tm term[] = {

   /*  1 */   { 0,  0,  1, { { -19798886e-6,  19848454e-6 },
                             {  -5453098e-6, -14974876e-6 },
                             {  66867334e-7,  68955876e-7 } } },
   /*  2 */   { 0,  0,  2, { {    897499e-6,  -4955707e-6 },
                             {   3527363e-6,   1672673e-6 },
                             { -11826086e-7,   -333765e-7 } } },
   /*  3 */   { 0,  0,  3, { {    610820e-6,   1210521e-6 },
                             {  -1050939e-6,    327763e-6 },
                             {   1593657e-7,  -1439953e-7 } } },
   /*  4 */   { 0,  0,  4, { {   -341639e-6,   -189719e-6 },
                             {    178691e-6,   -291925e-6 },
                             {    -18948e-7,    482443e-7 } } },
   /*  5 */   { 0,  0,  5, { {    129027e-6,    -34863e-6 },
                             {     18763e-6,    100448e-6 },
                             {    -66634e-7,    -85576e-7 } } },
   /*  6 */   { 0,  0,  6, { {    -38215e-6,     31061e-6 },
                             {    -30594e-6,    -25838e-6 },
                             {     30841e-7,     -5765e-7 } } },
   /*  7 */   { 0,  1, -1, { {     20349e-6,     -9886e-6 },
                             {      4965e-6,     11263e-6 },
                             {     -6140e-7,     22254e-7 } } },
   /*  8 */   { 0,  1,  0, { {     -4045e-6,     -4904e-6 },
                             {       310e-6,      -132e-6 },
                             {      4434e-7,      4443e-7 } } },
   /*  9 */   { 0,  1,  1, { {     -5885e-6,     -3238e-6 },
                             {      2036e-6,      -947e-6 },
                             {     -1518e-7,       641e-7 } } },
   /* 10 */   { 0,  1,  2, { {     -3812e-6,      3011e-6 },
                             {        -2e-6,      -674e-6 },
                             {        -5e-7,       792e-7 } } },
   /* 11 */   { 0,  1,  3, { {      -601e-6,      3468e-6 },
                             {      -329e-6,      -563e-6 },
                             {       518e-7,       518e-7 } } },
   /* 12 */   { 0,  2, -2, { {      1237e-6,       463e-6 },
                             {       -64e-6,        39e-6 },
                             {       -13e-7,      -221e-7 } } },
   /* 13 */   { 0,  2, -1, { {      1086e-6,      -911e-6 },
                             {       -94e-6,       210e-6 },
                             {       837e-7,      -494e-7 } } },
   /* 14 */   { 0,  2,  0, { {       595e-6,     -1229e-6 },
                             {        -8e-6,      -160e-6 },
                             {      -281e-7,       616e-7 } } },
   /* 15 */   { 1, -1,  0, { {      2484e-6,      -485e-6 },
                             {      -177e-6,       259e-6 },
                             {       260e-7,      -395e-7 } } },
   /* 16 */   { 1, -1,  1, { {       839e-6,     -1414e-6 },
                             {        17e-6,       234e-6 },
                             {      -191e-7,      -396e-7 } } },
   /* 17 */   { 1,  0, -3, { {      -964e-6,      1059e-6 },
                             {       582e-6,      -285e-6 },
                             {     -3218e-7,       370e-7 } } },
   /* 18 */   { 1,  0, -2, { {     -2303e-6,     -1038e-6 },
                             {      -298e-6,       692e-6 },
                             {      8019e-7,     -7869e-7 } } },
   /* 19 */   { 1,  0, -1, { {      7049e-6,       747e-6 },
                             {       157e-6,       201e-6 },
                             {       105e-7,     45637e-7 } } },
   /* 20 */   { 1,  0,  0, { {      1179e-6,      -358e-6 },
                             {       304e-6,       825e-6 },
                             {      8623e-7,      8444e-7 } } },
   /* 21 */   { 1,  0,  1, { {       393e-6,       -63e-6 },
                             {      -124e-6,       -29e-6 },
                             {      -896e-7,      -801e-7 } } },
   /* 22 */   { 1,  0,  2, { {       111e-6,      -268e-6 },
                             {        15e-6,         8e-6 },
                             {       208e-7,      -122e-7 } } },
   /* 23 */   { 1,  0,  3, { {       -52e-6,      -154e-6 },
                             {         7e-6,        15e-6 },
                             {      -133e-7,        65e-7 } } },
   /* 24 */   { 1,  0,  4, { {       -78e-6,       -30e-6 },
                             {         2e-6,         2e-6 },
                             {       -16e-7,         1e-7 } } },
   /* 25 */   { 1,  1, -3, { {       -34e-6,       -26e-6 },
                             {         4e-6,         2e-6 },
                             {       -22e-7,         7e-7 } } },
   /* 26 */   { 1,  1, -2, { {       -43e-6,         1e-6 },
                             {         3e-6,         0e-6 },
                             {        -8e-7,        16e-7 } } },
   /* 27 */   { 1,  1, -1, { {       -15e-6,        21e-6 },
                             {         1e-6,        -1e-6 },
                             {         2e-7,         9e-7 } } },
   /* 28 */   { 1,  1,  0, { {        -1e-6,        15e-6 },
                             {         0e-6,        -2e-6 },
                             {        12e-7,         5e-7 } } },
   /* 29 */   { 1,  1,  1, { {         4e-6,         7e-6 },
                             {         1e-6,         0e-6 },
                             {         1e-7,        -3e-7 } } },
   /* 30 */   { 1,  1,  3, { {         1e-6,         5e-6 },
                             {         1e-6,        -1e-6 },
                             {         1e-7,         0e-7 } } },
   /* 31 */   { 2,  0, -6, { {         8e-6,         3e-6 },
                             {        -2e-6,        -3e-6 },
                             {         9e-7,         5e-7 } } },
   /* 32 */   { 2,  0, -5, { {        -3e-6,         6e-6 },
                             {         1e-6,         2e-6 },
                             {         2e-7,        -1e-7 } } },
   /* 33 */   { 2,  0, -4, { {         6e-6,       -13e-6 },
                             {        -8e-6,         2e-6 },
                             {        14e-7,        10e-7 } } },
   /* 34 */   { 2,  0, -3, { {        10e-6,        22e-6 },
                             {        10e-6,        -7e-6 },
                             {       -65e-7,        12e-7 } } },
   /* 35 */   { 2,  0, -2, { {       -57e-6,       -32e-6 },
                             {         0e-6,        21e-6 },
                             {       126e-7,      -233e-7 } } },
   /* 36 */   { 2,  0, -1, { {       157e-6,       -46e-6 },
                             {         8e-6,         5e-6 },
                             {       270e-7,      1068e-7 } } },
   /* 37 */   { 2,  0,  0, { {        12e-6,       -18e-6 },
                             {        13e-6,        16e-6 },
                             {       254e-7,       155e-7 } } },
   /* 38 */   { 2,  0,  1, { {        -4e-6,         8e-6 },
                             {        -2e-6,        -3e-6 },
                             {       -26e-7,        -2e-7 } } },
   /* 39 */   { 2,  0,  2, { {        -5e-6,         0e-6 },
                             {         0e-6,         0e-6 },
                             {         7e-7,         0e-7 } } },
   /* 40 */   { 2,  0,  3, { {         3e-6,         4e-6 },
                             {         0e-6,         1e-6 },
                             {       -11e-7,         4e-7 } } },
   /* 41 */   { 3,  0, -2, { {        -1e-6,        -1e-6 },
                             {         0e-6,         1e-6 },
                             {         4e-7,       -14e-7 } } },
   /* 42 */   { 3,  0, -1, { {         6e-6,        -3e-6 },
                             {         0e-6,         0e-6 },
                             {        18e-7,        35e-7 } } },
   /* 43 */   { 3,  0,  0, { {        -1e-6,        -2e-6 },
                             {         0e-6,         1e-6 },
                             {        13e-7,         3e-7 } } } };



/* Validate the planet number. */
   if ( np < 1 || np > 9 ) {
      *jstat = -1;
      for ( i = 0; i <= 5; i++ ) pv[i] = 0.0;
      return;
   } else {
      ip = np - 1;
   }

/* Separate algorithms for Pluto and the rest. */
   if ( np != 9 ) {

   /* ----------------------- */
   /* Mercury through Neptune */
   /* ----------------------- */

   /* Time: Julian millennia since J2000. */
      t = ( date - 51544.5 ) / 365250.0;

   /* OK status unless remote epoch. */
      *jstat = ( fabs_ ( t ) <= 1.0 ) ? 0 : 1;

   /* Compute the mean elements. */
      da = a[ip][0] + ( a[ip][1] + a[ip][2] * t ) * t;
      dl = ( 3600.0 * dlm[ip][0] + ( dlm[ip][1] + dlm[ip][2] * t ) * t )
                                                                  * DAS2R;
      de = e[ip][0] + ( e[ip][1] + e[ip][2] * t ) * t;
      dpe = dmod ( ( 3600.0 * pi[ip][0] + ( pi[ip][1] + pi[ip][2] * t ) * t )
                                                              * DAS2R,D2PI );
      di = ( 3600.0 * dinc[ip][0] + ( dinc[ip][1] + dinc[ip][2] * t ) * t )
                                                                     * DAS2R;
      dom = dmod( ( 3600.0 * omega[ip][0] + ( omega[ip][1]
                                  + omega[ip][2] * t ) * t ) * DAS2R, D2PI );

   /* Apply the trigonometric terms. */
      dmu = 0.35953620 * t;
      for ( j = 0; j <= 7; j++ ) {
         arga = dkp[ip][j] * dmu;
         argl = dkq[ip][j] * dmu;
         da += ( ca[ip][j] * cos_ ( arga ) +
                 sa[ip][j] * sin_ ( arga ) ) * 1e-7;
         dl += ( clo[ip][j] * cos_ ( argl ) +
                 slo[ip][j] * sin_ ( argl ) ) * 1e-7;
      }
      arga = dkp[ip][8] * dmu;
      da += t * ( ca[ip][8] * cos_ ( arga ) +
                  sa[ip][8] * sin_ ( arga ) ) * 1e-7;
      for ( j = 8; j <= 9; j++ ) {
         argl = dkq[ip][j] * dmu;
         dl += t * ( clo[ip][j] * cos_ ( argl ) +
                     slo[ip][j] * sin_ ( argl ) ) * 1e-7;
      }
      dl = dmod ( dl, D2PI );

   /* Daily motion. */
      dm = GCON * sqrt_ ( ( 1.0 + 1.0 / amas[ip] ) / ( da * da * da ) );

   /* Make the prediction. */
      slaPlanel ( date, 1, date, di, dom, dpe, da, de, dl, dm, pv, &j );
      if ( j < 0 ) *jstat = -2;


   } else {

   /* ----- */
   /* Pluto */
   /* ----- */

   /* Time: Julian centuries since J2000. */
      t = ( date - 51544.5 ) / 36525.0;

   /* OK status unless remote epoch. */
      *jstat = t >= -1.15 && t <= 1.0 ? 0 : -1;

   /* Fundamental arguments (radians). */
      dj = ( dj0 + djd * t ) * DD2R;
      ds = ( ds0 + dsd * t ) * DD2R;
      dp = ( dp0 + dpd * t ) * DD2R;

   /* Initialize coefficients and derivatives. */
      for ( i = 0; i < 3; i++ ) {
         wlbr[i] = 0.0;
         wlbrd[i] = 0.0;
      }

   /* Term by term through Meeus Table 36.A. */
      for ( j = 0; j < (int) ( sizeof term / sizeof term[0] ); j++ ) {

      /* Argument and derivative (radians, radians per century). */
         wj = (double) ( term[j].ij );
         ws = (double) ( term[j].is );
         wp = (double) ( term[j].ip );
         al = wj * dj + ws * ds + wp * dp;
         ald = ( wj * djd + ws * dsd + wp * dpd ) * DD2R;

      /* Functions of argument. */
         sal = sin_ ( al );
         cal = cos_ ( al );

      /* Periodic terms in longitude, latitude, radius vector. */
         for ( i = 0; i < 3; i++ ) {

         /* A and B coefficients (deg, AU). */
            ac = term[j].dlbr[i].a;
            bc = term[j].dlbr[i].b;

         /* Periodic terms (deg, AU, deg/Jc, AU/Jc). */
            wlbr[i] = wlbr[i] + ac * sal + bc * cal;
            wlbrd[i] = wlbrd[i] + ( ac * cal - bc * sal ) * ald;
         }
      }

   /* Heliocentric longitude and derivative (radians, radians/sec). */
      dl = ( dl0 + dld0 * t + wlbr[0] ) * DD2R;
      dld = ( dld0 + wlbrd[0] ) * DD2R / SPC;

   /* Heliocentric latitude and derivative (radians, radians/sec). */
      db = ( db0 + wlbr[1] ) * DD2R;
      dbd = wlbrd[1] * DD2R / SPC;

   /* Heliocentric radius vector and derivative (AU, AU/sec). */
      dr = dr0 + wlbr[2];
      drd = wlbrd[2] / SPC;

   /* Functions of latitude, longitude, radius vector. */
      sl = sin_ ( dl );
      cl = cos_ ( dl );
      sb = sin_ ( db );
      cb = cos_ ( db );
      slcb = sl * cb;
      clcb = cl * cb;

   /* Heliocentric vector and derivative, J2000 ecliptic and equinox. */
      x = dr * clcb;
      y = dr * slcb;
      z = dr * sb;
      xd = drd * clcb - dr * ( cl * sb * dbd + slcb * dld );
      yd = drd * slcb + dr * ( - sl * sb * dbd + clcb * dld );
      zd = drd * sb + dr * cb * dbd;

   /* Transform to J2000 equator and equinox. */
      pv[0] = x;
      pv[1] = y * CE - z * SE;
      pv[2] = y * SE + z * CE;
      pv[3] = xd;
      pv[4] = yd * CE - zd * SE;
      pv[5] = yd * SE + zd * CE;
   }
}
