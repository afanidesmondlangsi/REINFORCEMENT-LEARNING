# Final Consolidated Experimental Results

## Experimental design

- Four environments: Ant, HalfCheetah, Hopper and Walker2d.
- Four friction shifts: 0.1, 0.5, 2.0 and 5.0.
- Three seeds: 0, 1 and 2.
- Six algorithms.
- 288 total experimental records.
- Evaluation checkpoints: 1,000 through 5,000 steps.

## RQ1: Negative transfer

| environment   |   shift | method              |   mean_auc_advantage |   std_auc_advantage |   mean_return_advantage |   positive_transfer |   negative_transfer |   paired_conditions |   algorithm_rank_auc |   positive_transfer_rank |
|:--------------|--------:|:--------------------|---------------------:|--------------------:|------------------------:|--------------------:|--------------------:|--------------------:|---------------------:|-------------------------:|
| ant           |     0.1 | BC_PAR              |          2.05705e+06 |    359624           |               710.512   |            100      |              0      |                   3 |                    9 |                        1 |
| ant           |     0.1 | BC_SAC              |          1.53668e+06 |    415680           |               963.028   |            100      |              0      |                   3 |                   18 |                        1 |
| ant           |     0.1 | EA_NTC_NORMALIZED   |          1.49413e+06 |    452213           |               877.922   |            100      |              0      |                   3 |                   19 |                        1 |
| ant           |     0.1 | EA_NTC_UNNORMALIZED |          1.54143e+06 |    412220           |               972.525   |            100      |              0      |                   3 |                   17 |                        1 |
| ant           |     0.1 | RLPD                |         -2.13937e+06 |    525367           |              -198.737   |              0      |            100      |                   3 |                   80 |                       72 |
| ant           |     0.5 | BC_PAR              |          3.1301e+06  |         1.52411e+06 |               976.933   |            100      |              0      |                   3 |                    4 |                        1 |
| ant           |     0.5 | BC_SAC              |          1.60753e+06 |         2.35965e+06 |               367.428   |             66.6667 |             33.3333 |                   3 |                   15 |                       43 |
| ant           |     0.5 | EA_NTC_NORMALIZED   |          1.6456e+06  |         2.29095e+06 |               443.573   |            100      |              0      |                   3 |                   11 |                        1 |
| ant           |     0.5 | EA_NTC_UNNORMALIZED |          1.61857e+06 |         2.32835e+06 |               389.497   |             66.6667 |             33.3333 |                   3 |                   12 |                       43 |
| ant           |     0.5 | RLPD                |         -1.07201e+06 |         1.61875e+06 |                48.5055  |             33.3333 |             66.6667 |                   3 |                   78 |                       64 |
| ant           |     2   | BC_PAR              |          4.05474e+06 |         1.52875e+06 |              1683.6     |            100      |              0      |                   3 |                    1 |                        1 |
| ant           |     2   | BC_SAC              |          1.60168e+06 |         1.48237e+06 |               824.977   |            100      |              0      |                   3 |                   16 |                        1 |
| ant           |     2   | EA_NTC_NORMALIZED   |          1.61594e+06 |         1.44136e+06 |               853.493   |            100      |              0      |                   3 |                   13 |                        1 |
| ant           |     2   | EA_NTC_UNNORMALIZED |          1.60776e+06 |         1.44564e+06 |               837.13    |            100      |              0      |                   3 |                   14 |                        1 |
| ant           |     2   | RLPD                |    -295593           |         1.00247e+06 |               708.253   |             33.3333 |             66.6667 |                   3 |                   73 |                       64 |
| ant           |     5   | BC_PAR              |          2.95135e+06 |     52137.8         |               989.273   |            100      |              0      |                   3 |                    5 |                        1 |
| ant           |     5   | BC_SAC              |     160688           |    105965           |                 5.44181 |            100      |              0      |                   3 |                   46 |                        1 |
| ant           |     5   | EA_NTC_NORMALIZED   |     156604           |    106997           |                -2.72549 |            100      |              0      |                   3 |                   47 |                        1 |
| ant           |     5   | EA_NTC_UNNORMALIZED |     169700           |    101606           |                23.4659  |            100      |              0      |                   3 |                   45 |                        1 |
| ant           |     5   | RLPD                |         -1.14225e+06 |    145507           |                -5.09313 |              0      |            100      |                   3 |                   79 |                       72 |
| halfcheetah   |     0.1 | BC_PAR              |     174029           |    140534           |               187.132   |            100      |              0      |                   3 |                   44 |                        1 |
| halfcheetah   |     0.1 | BC_SAC              |          2.35343e+06 |    221958           |               171.76    |            100      |              0      |                   3 |                    7 |                        1 |
| halfcheetah   |     0.1 | EA_NTC_NORMALIZED   |          2.26167e+06 |    175085           |               363.919   |            100      |              0      |                   3 |                    8 |                        1 |
| halfcheetah   |     0.1 | EA_NTC_UNNORMALIZED |          1.81833e+06 |    134526           |               144.108   |            100      |              0      |                   3 |                   10 |                        1 |
| halfcheetah   |     0.1 | RLPD                |    -370230           |    722291           |                94.0047  |             33.3333 |             66.6667 |                   3 |                   75 |                       64 |
| halfcheetah   |     0.5 | BC_PAR              |     324077           |    107875           |               179.83    |            100      |              0      |                   3 |                   31 |                        1 |
| halfcheetah   |     0.5 | BC_SAC              |          2.59554e+06 |    536228           |               907.675   |            100      |              0      |                   3 |                    6 |                        1 |
| halfcheetah   |     0.5 | EA_NTC_NORMALIZED   |          3.35973e+06 |    434354           |              1658.8     |            100      |              0      |                   3 |                    2 |                        1 |
| halfcheetah   |     0.5 | EA_NTC_UNNORMALIZED |          3.34659e+06 |    618633           |              1338.45    |            100      |              0      |                   3 |                    3 |                        1 |
| halfcheetah   |     0.5 | RLPD                |     -39582.2         |    247432           |                 6.65464 |             66.6667 |             33.3333 |                   3 |                   65 |                       43 |
| halfcheetah   |     2   | BC_PAR              |     177705           |    208156           |               181.274   |             66.6667 |             33.3333 |                   3 |                   43 |                       43 |
| halfcheetah   |     2   | BC_SAC              |     695958           |         1.09034e+06 |               418.631   |             66.6667 |             33.3333 |                   3 |                   27 |                       43 |
| halfcheetah   |     2   | EA_NTC_NORMALIZED   |     294007           |    446803           |               679.926   |             66.6667 |             33.3333 |                   3 |                   32 |                       43 |
| halfcheetah   |     2   | EA_NTC_UNNORMALIZED |     338821           |    367343           |               639.195   |             66.6667 |             33.3333 |                   3 |                   28 |                       43 |
| halfcheetah   |     2   | RLPD                |      50646.1         |    728233           |               339.456   |             33.3333 |             66.6667 |                   3 |                   57 |                       64 |
| halfcheetah   |     5   | BC_PAR              |     325285           |    138064           |               141.251   |            100      |              0      |                   3 |                   30 |                        1 |
| halfcheetah   |     5   | BC_SAC              |     132219           |    122124           |                21.2978  |            100      |              0      |                   3 |                   48 |                        1 |
| halfcheetah   |     5   | EA_NTC_NORMALIZED   |     194267           |    273329           |               148.478   |             66.6667 |             33.3333 |                   3 |                   42 |                       43 |
| halfcheetah   |     5   | EA_NTC_UNNORMALIZED |     260166           |    334179           |               279.549   |            100      |              0      |                   3 |                   35 |                        1 |
| halfcheetah   |     5   | RLPD                |    -274887           |    143393           |                 8.72868 |              0      |            100      |                   3 |                   71 |                       72 |
| hopper        |     0.1 | BC_PAR              |    -160347           |     82733.9         |                17.8865  |              0      |            100      |                   3 |                   67 |                       72 |
| hopper        |     0.1 | BC_SAC              |     129773           |     64978.5         |                42.0809  |            100      |              0      |                   3 |                   49 |                        1 |
| hopper        |     0.1 | EA_NTC_NORMALIZED   |     109585           |     58701.7         |                19.0137  |            100      |              0      |                   3 |                   52 |                        1 |
| hopper        |     0.1 | EA_NTC_UNNORMALIZED |     120037           |     59275.6         |                 4.38466 |            100      |              0      |                   3 |                   51 |                        1 |
| hopper        |     0.1 | RLPD                |      -6450.12        |    128304           |                -3.27015 |             66.6667 |             33.3333 |                   3 |                   62 |                       43 |
| hopper        |     0.5 | BC_PAR              |    -172423           |     99681.8         |                -6.28487 |              0      |            100      |                   3 |                   69 |                       72 |
| hopper        |     0.5 | BC_SAC              |     128855           |    183711           |               184.546   |             66.6667 |             33.3333 |                   3 |                   50 |                       43 |
| hopper        |     0.5 | EA_NTC_NORMALIZED   |      68881.3         |    101146           |                55.9734  |             66.6667 |             33.3333 |                   3 |                   55 |                       43 |
| hopper        |     0.5 | EA_NTC_UNNORMALIZED |      49837.5         |     79657.7         |                26.8225  |             66.6667 |             33.3333 |                   3 |                   58 |                       43 |
| hopper        |     0.5 | RLPD                |      -6450.12        |    128304           |                -3.27015 |             66.6667 |             33.3333 |                   3 |                   62 |                       43 |
| hopper        |     2   | BC_PAR              |    -249621           |    168440           |                -9.14184 |              0      |            100      |                   3 |                   70 |                       72 |
| hopper        |     2   | BC_SAC              |     276195           |     83730.3         |               104.939   |            100      |              0      |                   3 |                   34 |                        1 |
| hopper        |     2   | EA_NTC_NORMALIZED   |     250529           |    139910           |                59.0432  |            100      |              0      |                   3 |                   36 |                        1 |
| hopper        |     2   | EA_NTC_UNNORMALIZED |     287333           |    109703           |               128.618   |            100      |              0      |                   3 |                   33 |                        1 |
| hopper        |     2   | RLPD                |     -84311.8         |    139444           |               -23.3457  |             33.3333 |             66.6667 |                   3 |                   66 |                       64 |
| hopper        |     5   | BC_PAR              |    -321266           |     60440.6         |               -62.9499  |              0      |            100      |                   3 |                   74 |                       72 |
| hopper        |     5   | BC_SAC              |       4247.57        |     58252.7         |               -62.7218  |             66.6667 |             33.3333 |                   3 |                   61 |                       43 |
| hopper        |     5   | EA_NTC_NORMALIZED   |      12389.1         |     72158.3         |               -29.9376  |             66.6667 |             33.3333 |                   3 |                   60 |                       43 |
| hopper        |     5   | EA_NTC_UNNORMALIZED |      55977.8         |     51647           |                37.1951  |            100      |              0      |                   3 |                   56 |                        1 |
| hopper        |     5   | RLPD                |     -17819.4         |    131687           |               -72.3144  |             33.3333 |             66.6667 |                   3 |                   64 |                       64 |
| walker2d      |     0.1 | BC_PAR              |    -278878           |    256193           |              -164.973   |             33.3333 |             66.6667 |                   3 |                   72 |                       64 |
| walker2d      |     0.1 | BC_SAC              |     829230           |    140677           |               -40.7743  |            100      |              0      |                   3 |                   24 |                        1 |
| walker2d      |     0.1 | EA_NTC_NORMALIZED   |     911404           |    147741           |               -49.7001  |            100      |              0      |                   3 |                   22 |                        1 |
| walker2d      |     0.1 | EA_NTC_UNNORMALIZED |     798170           |    167971           |              -130.597   |            100      |              0      |                   3 |                   25 |                        1 |
| walker2d      |     0.1 | RLPD                |     248403           |    200770           |               -74.61    |            100      |              0      |                   3 |                   37 |                        1 |
| walker2d      |     0.5 | BC_PAR              |         -1.00895e+06 |    716059           |              -135.707   |              0      |            100      |                   3 |                   77 |                       72 |
| walker2d      |     0.5 | BC_SAC              |      45553.7         |    490722           |                99.5654  |             66.6667 |             33.3333 |                   3 |                   59 |                       43 |
| walker2d      |     0.5 | EA_NTC_NORMALIZED   |     230248           |    418236           |                45.5422  |             66.6667 |             33.3333 |                   3 |                   39 |                       43 |
| walker2d      |     0.5 | EA_NTC_UNNORMALIZED |     104898           |    475863           |                99.0572  |             66.6667 |             33.3333 |                   3 |                   53 |                       43 |
| walker2d      |     0.5 | RLPD                |    -170700           |    724429           |                 0.67919 |             33.3333 |             66.6667 |                   3 |                   68 |                       64 |
| walker2d      |     2   | BC_PAR              |      86476.7         |    173163           |               132.194   |             66.6667 |             33.3333 |                   3 |                   54 |                       43 |
| walker2d      |     2   | BC_SAC              |     935062           |    462331           |               226.76    |            100      |              0      |                   3 |                   21 |                        1 |
| walker2d      |     2   | EA_NTC_NORMALIZED   |     878803           |    332667           |               128.283   |            100      |              0      |                   3 |                   23 |                        1 |
| walker2d      |     2   | EA_NTC_UNNORMALIZED |          1.03952e+06 |    583813           |               267.501   |            100      |              0      |                   3 |                   20 |                        1 |
| walker2d      |     2   | RLPD                |     791837           |     73940.4         |               176.56    |            100      |              0      |                   3 |                   26 |                        1 |
| walker2d      |     5   | BC_PAR              |    -506458           |    261481           |               -18.0846  |              0      |            100      |                   3 |                   76 |                       72 |
| walker2d      |     5   | BC_SAC              |     208160           |    334036           |               112.774   |             66.6667 |             33.3333 |                   3 |                   41 |                       43 |
| walker2d      |     5   | EA_NTC_NORMALIZED   |     223937           |    313788           |                97.9187  |             66.6667 |             33.3333 |                   3 |                   40 |                       43 |
| walker2d      |     5   | EA_NTC_UNNORMALIZED |     235988           |    269096           |               104.663   |            100      |              0      |                   3 |                   38 |                        1 |
| walker2d      |     5   | RLPD                |     333739           |    158405           |                84.4761  |            100      |              0      |                   3 |                   29 |                        1 |

## RQ2: Task relatedness

| environment   |   shift | method              |        mean |         std |   count |   rank |
|:--------------|--------:|:--------------------|------------:|------------:|--------:|-------:|
| ant           |     0.1 | BC_PAR              |  994.102    |    2.40064  |       3 |      4 |
| ant           |     0.1 | BC_SAC              | 1246.62     |  823.285    |       3 |      2 |
| ant           |     0.1 | EA_NTC_NORMALIZED   | 1161.51     |  944.863    |       3 |      3 |
| ant           |     0.1 | EA_NTC_UNNORMALIZED | 1256.12     |  810.25     |       3 |      1 |
| ant           |     0.1 | RLPD                |   84.8537   |  235.192    |       3 |      6 |
| ant           |     0.1 | SAC_TARGET_ONLY     |  283.591    |  120.765    |       3 |      5 |
| ant           |     0.5 | BC_PAR              |  995.016    |    3.16573  |       3 |      1 |
| ant           |     0.5 | BC_SAC              |  385.51     |  167.79     |       3 |      4 |
| ant           |     0.5 | EA_NTC_NORMALIZED   |  461.656    |  324.835    |       3 |      2 |
| ant           |     0.5 | EA_NTC_UNNORMALIZED |  407.579    |  222.775    |       3 |      3 |
| ant           |     0.5 | RLPD                |   66.5881   |  206.823    |       3 |      5 |
| ant           |     0.5 | SAC_TARGET_ONLY     |   18.0826   |   79.5188   |       3 |      6 |
| ant           |     2   | BC_PAR              |  994.558    |    3.54025  |       3 |      1 |
| ant           |     2   | BC_SAC              |  135.934    |   29.0607   |       3 |      4 |
| ant           |     2   | EA_NTC_NORMALIZED   |  164.45     |  123.188    |       3 |      2 |
| ant           |     2   | EA_NTC_UNNORMALIZED |  148.086    |   45.7499   |       3 |      3 |
| ant           |     2   | RLPD                |   19.2095   |   52.4009   |       3 |      5 |
| ant           |     2   | SAC_TARGET_ONLY     | -689.043    | 1050.44     |       3 |      6 |
| ant           |     5   | BC_PAR              |  994.92     |    2.88665  |       3 |      1 |
| ant           |     5   | BC_SAC              |   11.0884   |   61.9295   |       3 |      3 |
| ant           |     5   | EA_NTC_NORMALIZED   |    2.92106  |   63.2136   |       3 |      5 |
| ant           |     5   | EA_NTC_UNNORMALIZED |   29.1124   |   46.4278   |       3 |      2 |
| ant           |     5   | RLPD                |    0.553422 |   12.1907   |       3 |      6 |
| ant           |     5   | SAC_TARGET_ONLY     |    5.64656  |   12.5725   |       3 |      4 |
| halfcheetah   |     0.1 | BC_PAR              |   -0.870231 |    1.00547  |       3 |      2 |
| halfcheetah   |     0.1 | BC_SAC              |  -16.2414   |  203.173    |       3 |      3 |
| halfcheetah   |     0.1 | EA_NTC_NORMALIZED   |  175.918    |  168.834    |       3 |      1 |
| halfcheetah   |     0.1 | EA_NTC_UNNORMALIZED |  -43.8937   |  133.243    |       3 |      4 |
| halfcheetah   |     0.1 | RLPD                |  -93.9971   |   58.2409   |       3 |      5 |
| halfcheetah   |     0.1 | SAC_TARGET_ONLY     | -188.002    |  200.363    |       3 |      6 |
| halfcheetah   |     0.5 | BC_PAR              |   -1.47654  |    0.459216 |       3 |      4 |
| halfcheetah   |     0.5 | BC_SAC              |  726.369    |  607.692    |       3 |      3 |
| halfcheetah   |     0.5 | EA_NTC_NORMALIZED   | 1477.49     |  596.917    |       3 |      1 |
| halfcheetah   |     0.5 | EA_NTC_UNNORMALIZED | 1157.14     |  554.944    |       3 |      2 |
| halfcheetah   |     0.5 | RLPD                | -174.652    |   47.2075   |       3 |      5 |
| halfcheetah   |     0.5 | SAC_TARGET_ONLY     | -181.307    |  118.142    |       3 |      6 |
| halfcheetah   |     2   | BC_PAR              |   -1.55303  |    0.204008 |       3 |      5 |
| halfcheetah   |     2   | BC_SAC              |  235.804    |  297.124    |       3 |      3 |
| halfcheetah   |     2   | EA_NTC_NORMALIZED   |  497.099    |  408.711    |       3 |      1 |
| halfcheetah   |     2   | EA_NTC_UNNORMALIZED |  456.368    |  135.739    |       3 |      2 |
| halfcheetah   |     2   | RLPD                |  156.629    |  510.006    |       3 |      4 |
| halfcheetah   |     2   | SAC_TARGET_ONLY     | -182.827    |  358.32     |       3 |      6 |
| halfcheetah   |     5   | BC_PAR              |   -1.35292  |    0.164695 |       3 |      3 |
| halfcheetah   |     5   | BC_SAC              | -121.306    |  104.007    |       3 |      4 |
| halfcheetah   |     5   | EA_NTC_NORMALIZED   |    5.87406  |  212.071    |       3 |      2 |
| halfcheetah   |     5   | EA_NTC_UNNORMALIZED |  136.944    |  328.349    |       3 |      1 |
| halfcheetah   |     5   | RLPD                | -133.876    |  112.622    |       3 |      5 |
| halfcheetah   |     5   | SAC_TARGET_ONLY     | -142.604    |   56.7889   |       3 |      6 |
| hopper        |     0.1 | BC_PAR              |   94.065    |   41.3649   |       3 |      3 |
| hopper        |     0.1 | BC_SAC              |  118.259    |   69.9012   |       3 |      1 |
| hopper        |     0.1 | EA_NTC_NORMALIZED   |   95.1922   |   36.405    |       3 |      2 |
| hopper        |     0.1 | EA_NTC_UNNORMALIZED |   80.5632   |   26.7105   |       3 |      4 |
| hopper        |     0.1 | RLPD                |   72.9084   |   18.0803   |       3 |      6 |
| hopper        |     0.1 | SAC_TARGET_ONLY     |   76.1785   |   43.6933   |       3 |      5 |
| hopper        |     0.5 | BC_PAR              |   69.8937   |   23.9632   |       3 |      6 |
| hopper        |     0.5 | BC_SAC              |  260.724    |  343.808    |       3 |      1 |
| hopper        |     0.5 | EA_NTC_NORMALIZED   |  132.152    |  149.043    |       3 |      2 |
| hopper        |     0.5 | EA_NTC_UNNORMALIZED |  103.001    |   75.6571   |       3 |      3 |
| hopper        |     0.5 | RLPD                |   72.9084   |   18.0803   |       3 |      5 |
| hopper        |     0.5 | SAC_TARGET_ONLY     |   76.1785   |   43.6933   |       3 |      4 |
| hopper        |     2   | BC_PAR              |   87.844    |   36.3265   |       3 |      5 |
| hopper        |     2   | BC_SAC              |  201.925    |  120.136    |       3 |      2 |
| hopper        |     2   | EA_NTC_NORMALIZED   |  156.029    |  136.894    |       3 |      3 |
| hopper        |     2   | EA_NTC_UNNORMALIZED |  225.604    |  138.285    |       3 |      1 |
| hopper        |     2   | RLPD                |   73.6401   |   28.4342   |       3 |      6 |
| hopper        |     2   | SAC_TARGET_ONLY     |   96.9858   |  126.611    |       3 |      4 |
| hopper        |     5   | BC_PAR              |   46.1473   |   19.447    |       3 |      5 |
| hopper        |     5   | BC_SAC              |   46.3754   |   32.0789   |       3 |      4 |
| hopper        |     5   | EA_NTC_NORMALIZED   |   79.1596   |   30.5636   |       3 |      3 |
| hopper        |     5   | EA_NTC_UNNORMALIZED |  146.292    |   90.0081   |       3 |      1 |
| hopper        |     5   | RLPD                |   36.7828   |   11.44     |       3 |      6 |
| hopper        |     5   | SAC_TARGET_ONLY     |  109.097    |   87.1396   |       3 |      2 |
| walker2d      |     0.1 | BC_PAR              |   77.5952   |  105.098    |       3 |      6 |
| walker2d      |     0.1 | BC_SAC              |  201.793    |  164.387    |       3 |      2 |
| walker2d      |     0.1 | EA_NTC_NORMALIZED   |  192.868    |  172.493    |       3 |      3 |
| walker2d      |     0.1 | EA_NTC_UNNORMALIZED |  111.971    |  115.623    |       3 |      5 |
| walker2d      |     0.1 | RLPD                |  167.958    |   96.0825   |       3 |      4 |
| walker2d      |     0.1 | SAC_TARGET_ONLY     |  242.568    |   40.6055   |       3 |      1 |
| walker2d      |     0.5 | BC_PAR              |  122.893    |  193.525    |       3 |      6 |
| walker2d      |     0.5 | BC_SAC              |  358.166    |  309.565    |       3 |      1 |
| walker2d      |     0.5 | EA_NTC_NORMALIZED   |  304.142    |  137.851    |       3 |      3 |
| walker2d      |     0.5 | EA_NTC_UNNORMALIZED |  357.658    |  282.324    |       3 |      2 |
| walker2d      |     0.5 | RLPD                |  259.279    |  156.442    |       3 |      4 |
| walker2d      |     0.5 | SAC_TARGET_ONLY     |  258.6      |  202.718    |       3 |      5 |
| walker2d      |     2   | BC_PAR              |  127.266    |  197.873    |       3 |      4 |
| walker2d      |     2   | BC_SAC              |  221.833    |  333.065    |       3 |      2 |
| walker2d      |     2   | EA_NTC_NORMALIZED   |  123.356    |  110.257    |       3 |      5 |
| walker2d      |     2   | EA_NTC_UNNORMALIZED |  262.574    |  297.368    |       3 |      1 |
| walker2d      |     2   | RLPD                |  171.633    |   64.9474   |       3 |      3 |
| walker2d      |     2   | SAC_TARGET_ONLY     |   -4.92711  |   11.8628   |       3 |      6 |
| walker2d      |     5   | BC_PAR              |   98.1611   |  152        |       3 |      6 |
| walker2d      |     5   | BC_SAC              |  229.019    |  164.622    |       3 |      1 |
| walker2d      |     5   | EA_NTC_NORMALIZED   |  214.164    |  152.044    |       3 |      3 |
| walker2d      |     5   | EA_NTC_UNNORMALIZED |  220.908    |   94.2541   |       3 |      2 |
| walker2d      |     5   | RLPD                |  200.722    |  116.549    |       3 |      4 |
| walker2d      |     5   | SAC_TARGET_ONLY     |  116.246    |  171.401    |       3 |      5 |

## RQ3: Normalized versus unnormalized EA-NTC

| environment   | checkpoint   |   mean_difference |   std_difference |   normalized_wins |   unnormalized_wins |
|:--------------|:-------------|------------------:|-----------------:|------------------:|--------------------:|
| ant           | eval_1000    |            0      |           0      |                 0 |                   0 |
| ant           | eval_2000    |            0      |           0      |                 0 |                   0 |
| ant           | eval_3000    |            0      |           0      |                 0 |                   0 |
| ant           | eval_4000    |            0      |           0      |                 0 |                   0 |
| ant           | eval_5000    |          -12.5886 |         110.755  |                 2 |                   7 |
| halfcheetah   | eval_1000    |            0      |           0      |                 0 |                   0 |
| halfcheetah   | eval_2000    |            0      |           0      |                 0 |                   0 |
| halfcheetah   | eval_3000    |            0      |           0      |                 0 |                   0 |
| halfcheetah   | eval_4000    |           30.2161 |         317.731  |                 5 |                   7 |
| halfcheetah   | eval_5000    |          112.456  |         528.805  |                 4 |                   8 |
| hopper        | eval_1000    |            0      |           0      |                 0 |                   0 |
| hopper        | eval_2000    |            0      |           0      |                 0 |                   0 |
| hopper        | eval_3000    |            0      |           0      |                 0 |                   0 |
| hopper        | eval_4000    |           -6.334  |          13.2441 |                 3 |                   9 |
| hopper        | eval_5000    |          -23.232  |          88.695  |                 5 |                   7 |
| walker2d      | eval_1000    |            0      |           0      |                 0 |                   0 |
| walker2d      | eval_2000    |            0      |           0      |                 0 |                   0 |
| walker2d      | eval_3000    |            0      |           0      |                 0 |                   0 |
| walker2d      | eval_4000    |           31.2777 |         164.468  |                 8 |                   4 |
| walker2d      | eval_5000    |          -29.6452 |         172.553  |                 3 |                   9 |

## Cross-environment rankings

| method              |   mean_final_return |   mean_rank |   conditions_won |   conditions |
|:--------------------|--------------------:|------------:|-----------------:|-------------:|
| EA_NTC_UNNORMALIZED |           316.001   |      2.3125 |                5 |           16 |
| BC_SAC              |           265.117   |      2.5    |                4 |           16 |
| EA_NTC_NORMALIZED   |           327.749   |      2.5625 |                3 |           16 |
| BC_PAR              |           293.576   |      3.875  |                3 |           16 |
| SAC_TARGET_ONLY     |            -6.59601 |      4.8125 |                1 |           16 |
| RLPD                |            61.3213  |      4.9375 |                0 |           16 |

## Statistical interpretation

All reported means and standard deviations are calculated from the three available seeds. The paired EA-NTC comparisons use identical environment, shift and seed conditions. No statistical-significance claims are made from these descriptive comparisons alone.

AUC transfer advantages in RQ1 are retained from the existing RQ1 analysis to preserve the original integration convention.