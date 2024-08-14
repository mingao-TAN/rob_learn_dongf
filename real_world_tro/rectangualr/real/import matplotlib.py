# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
inf = 3.5;
# 数组数据
ranges = [ 1.21300006,  1.21399999,  1.21200001,  1.21300006,  1.21200001,
        1.21399999,  1.21200001,  1.21300006,  1.21200001,  1.21800005,
        1.21399999,  1.21099997,  1.21200001,  1.21300006,  1.21300006,
        1.21200001,  1.21700001,  1.21899998,  1.21399999,  1.21200001,
        1.21200001,  1.21399999,  1.21300006,  1.21200001,  1.21300006,
        1.21300006,  1.21300006,  1.21399999,  1.21399999,  1.22399998,
        1.21800005,  1.22300005,  1.21599996,  1.21800005,  1.21700001,
        1.21200001,  1.20700002,  1.19799995,         inf,         inf,
        0.523     ,         inf,         inf,         inf,         inf,
        0.35699999,  0.347     ,  0.352     ,  0.35699999,  0.359     ,
        0.35600001,  0.35600001,  0.354     ,  0.35600001,  0.354     ,
        0.345     ,  0.34900001,  0.345     ,  0.34799999,  0.34999999,
        0.34900001,         inf,  0.34799999,  0.33899999,  0.34900001,
        0.35100001,  0.33500001,  0.33899999,  0.338     ,  0.33700001,
        0.338     ,  0.33700001,  0.33700001,  0.34      ,  0.33700001,
        0.33500001,  0.33899999,  0.34299999,  0.329     ,  0.33500001,
        0.333     ,  0.333     ,  0.336     ,  0.33500001,  0.317     ,
        0.324     ,  0.33000001,  0.331     ,  0.329     ,  0.331     ,
        0.33000001,  0.31799999,  0.322     ,  0.31999999,  0.32100001,
        0.31999999,  0.324     ,  0.32100001,  0.31600001,  0.32600001,
        0.32100001,  0.30899999,  0.324     ,  0.32100001,  0.322     ,
        0.31099999,  0.30899999,  0.30899999,  0.324     ,  0.30700001,
        0.31200001,  0.31799999,  0.31999999,  0.30899999,  0.31099999,
        0.30899999,  0.303     ,  0.30899999,  0.30899999,  0.30500001,
        0.30399999,  0.30500001,  0.308     ,  0.30899999,  0.30899999,
        0.30700001,  0.31299999,  0.31299999,  0.301     ,  0.31200001,
        0.31099999,  0.30899999,  0.29800001,  0.308     ,  0.30000001,
        0.30000001,  0.29800001,  0.29699999,  0.29899999,  0.31200001,
        0.31099999,  0.29699999,  0.29800001,  0.29800001,  0.29699999,
        0.30199999,  0.294     ,  0.29699999,  0.30000001,  0.29899999,
        0.29699999,  0.303     ,  0.301     ,  0.301     ,  0.301     ,
        0.303     ,  0.301     ,  0.296     ,  0.30000001,  0.29899999,
        0.30000001,  0.29899999,  0.30000001,  0.303     ,  0.303     ,
        0.29899999,  0.30000001,  0.30000001,  0.301     ,  0.289     ,
        0.28999999,  0.28999999,  0.292     ,  0.29100001,  0.289     ,
        0.29100001,  0.29300001,  0.29100001,  0.29100001,  0.289     ,
        0.29300001,  0.29100001,  0.29499999,  0.28999999,  0.29300001,
               inf,  0.29499999,  0.29499999,  0.292     ,  0.294     ,
        0.294     ,  0.296     ,  0.292     ,  0.29100001,  0.29499999,
        0.29699999,  0.29800001,  0.29499999,  0.285     ,  0.29800001,
        0.287     ,  0.287     ,  0.294     ,  0.29699999,  0.294     ,
        0.296     ,  0.29800001,  0.29699999,  0.28600001,  0.287     ,
        0.30000001,  0.29499999,  0.29499999,  0.29499999,  0.294     ,
        0.29499999,  0.29499999,  0.29300001,  0.292     ,  0.292     ,
        0.289     ,  0.292     ,  0.294     ,  0.29300001,  0.29699999,
        0.29300001,  0.29699999,  0.29499999,  0.29300001,  0.29300001,
        0.29499999,  0.29300001,  0.294     ,  0.29499999,  0.292     ,
        0.29300001,  0.296     ,  0.29300001,  0.294     ,  0.29699999,
        0.29100001,  0.292     ,  0.294     ,  0.29300001,  0.289     ,
        0.29300001,  0.29300001,  0.29100001,  0.285     ,  0.289     ,
        0.30000001,  0.28999999,  0.303     ,  0.303     ,  0.30000001,
        0.303     ,  0.301     ,  0.28600001,  0.29800001,  0.29899999,
        0.30000001,  0.29699999,  0.29499999,  0.301     ,  0.29899999,
        0.29800001,  0.30000001,  0.29800001,  0.296     ,         inf,
        0.29699999,  0.29699999,  0.29499999,  0.296     ,  0.29800001,
        0.29699999,  0.29800001,  0.30000001,  0.29800001,  0.30899999,
        0.296     ,  0.296     ,  0.31200001,  0.29899999,  0.296     ,
        0.31      ,  0.308     ,  0.294     ,  0.29499999,  0.30899999,
        0.30899999,  0.30899999,  0.31099999,  0.308     ,  0.30899999,
        0.308     ,  0.31099999,  0.30899999,  0.30399999,  0.31099999,
        0.308     ,  0.30399999,  0.30700001,  0.308     ,  0.308     ,
        0.31200001,  0.30500001,  0.32100001,  0.308     ,  0.322     ,
        0.31099999,  0.32600001,  0.32100001,  0.32300001,  0.31999999,
        0.317     ,  0.322     ,  0.32100001,  0.322     ,  0.32100001,
        0.31799999,  0.322     ,  0.322     ,  0.33500001,  0.31900001,
        0.322     ,  0.331     ,  0.322     ,  0.33199999,  0.333     ,
        0.31799999,  0.33199999,  0.333     ,  0.333     ,  0.33000001,
        0.33000001,  0.33000001,  0.333     ,  0.333     ,  0.33500001,
        0.33199999,  0.338     ,  0.33500001,  0.331     ,  0.33500001,
        0.347     ,  0.33700001,  0.33700001,  0.34799999,  0.338     ,
        0.35100001,  0.336     ,  0.34      ,  0.333     ,  0.33899999,
        0.33700001,  0.347     ,  0.32800001,         inf,  0.29699999,
        0.28099999,         inf,  0.26100001,  0.242     ,  0.235     ,
        0.23      ,  0.23800001,  0.228     ,  0.229     ,  0.24600001,
        0.236     ,  0.23800001,  0.237     ,  0.23800001,  0.23899999,
        0.241     ,  0.23199999,  0.23199999,  0.23100001,  0.23199999,
        0.236     ,  0.235     ,  0.234     ,  0.234     ,  0.235     ,
        0.227     ,  0.23800001,  0.23899999,  0.23      ,  0.23999999,
        0.245     ,  0.23899999,  0.23199999,  0.227     ,  0.23      ,
        0.228     ,  0.23      ,  0.234     ,  0.233     ,  0.236     ,
        0.229     ,  0.235     ,  0.235     ,  0.23199999,  0.23      ,
        0.23199999,  0.234     ,  0.235     ,  0.23199999,  0.235     ,
        0.23100001,  0.233     ,  0.233     ,  0.23199999,         inf,
               inf,  0.235     ,  0.235     ,  0.227     ,  0.236     ,
        0.23800001,  0.236     ,  0.23100001,  0.237     ,  0.234     ,
        0.23100001,  0.237     ,  0.234     ,  0.234     ,  0.235     ,
        0.235     ,  0.23800001,  0.234     ,  0.235     ,  0.233     ,
        0.23199999,  0.23      ,  0.233     ,  0.228     ,  0.234     ,
        0.233     ,  0.23100001,  0.23100001,  0.23      ,  0.227     ,
        0.233     ,  0.23199999,  0.23199999,  0.23      ,  0.228     ,
        0.241     ,  0.23      ,  0.23      ,  0.229     ,  0.242     ,
        0.228     ,  0.228     ,  0.229     ,  0.227     ,  0.241     ,
        0.243     ,  0.23899999,  0.23      ,  0.226     ,  0.235     ,
        0.235     ,  0.23999999,  0.23800001,  0.233     ,  0.235     ,
        0.23899999,  0.23899999,  0.233     ,  0.237     ,  0.236     ,
        0.23800001,  0.233     ,  0.237     ,         inf,  0.236     ,
        0.23800001,  0.236     ,  0.233     ,  0.243     ,  0.235     ,
        0.236     ,  0.233     ,  0.24699999,  0.235     ,  0.24699999,
        0.24699999,  0.245     ,  0.244     ,  0.244     ,  0.245     ,
        0.241     ,  0.23999999,  0.241     ,  0.241     ,  0.242     ,
        0.23999999,  0.244     ,  0.24600001,  0.241     ,  0.241     ,
        0.23999999,  0.243     ,  0.23800001,  0.23999999,  0.237     ,
        0.23899999,  0.25299999,  0.244     ,  0.23899999,  0.25799999,
        0.248     ,  0.248     ,  0.249     ,  0.25299999,  0.252     ,
        0.248     ,  0.248     ,  0.249     ,  0.24699999,  0.249     ,
        0.248     ,  0.248     ,  0.25400001,  0.249     ,  0.25299999,
        0.25299999,  0.24699999,  0.26199999,  0.24699999,  0.25799999,
        0.25999999,  0.259     ,  0.26100001,  0.26100001,  0.259     ,
        0.26300001,  0.25799999,  0.25799999,  0.259     ,  0.255     ,
        0.255     ,  0.25299999,  0.26699999,  0.25600001,  0.271     ,
        0.25799999,  0.255     ,  0.26100001,  0.255     ,  0.26699999,
        0.264     ,  0.26699999,  0.26899999,  0.26699999,  0.26800001,
        0.26199999,  0.264     ,  0.264     ,  0.28099999,  0.26800001,
        0.28      ,  0.28099999,  0.27599999,  0.27599999,  0.27900001,
        0.28099999,  0.27599999,  0.27900001,  0.27700001,  0.27500001,
        0.27599999,  0.27500001,  0.27599999,  0.27599999,  0.27500001,
        0.27700001,  0.27700001,  0.26800001,  0.28799999,  0.28099999,
        0.28200001,  0.28799999,  0.28999999,  0.287     ,  0.27700001,
        0.287     ,  0.28999999,  0.28999999,  0.301     ,  0.28999999,
        0.289     ,  0.30000001,  0.303     ,  0.29899999,  0.303     ,
        0.30000001,  0.301     ,  0.29800001,  0.29300001,  0.29300001,
        0.29899999,  0.30700001,  0.301     ,  0.301     ,  0.301     ,
        0.30000001,  0.30399999,  0.31600001,  0.30199999,  0.30500001,
        0.31400001,  0.31900001,  0.315     ,  0.31600001,  0.317     ,
        0.308     ,  0.31900001,  0.32499999,  0.32800001,  0.32699999,
        0.33000001,  0.32699999,  0.32800001,  0.32800001,  0.33000001,
        0.33500001,  0.33500001,  0.336     ,  0.336     ,  0.33899999,
        0.34900001,  0.338     ,  0.336     ,  0.34999999,  0.352     ,
        0.34999999,  0.35499999,  0.35800001,  0.35499999,  0.35600001,
        0.35499999,  0.36399999,  0.36000001,  0.37200001,  0.36899999,
        0.375     ,  0.37400001,  0.36700001,  0.37      ,  0.37099999,
        0.366     ,  0.38100001,  0.37799999,  0.38299999,  0.37900001,
        0.375     ,  0.39199999,  0.38      ,  0.38100001,  0.382     ,
        0.396     ,  0.40400001,  0.40099999,  0.40400001,  0.40000001,
        0.40700001,  0.40200001,  0.41999999,  0.41999999,  0.421     ,
        0.421     ,  0.426     ,  0.42500001,  0.42399999,  0.428     ,
        0.43200001,  0.428     ,  0.42899999,  0.43000001,  0.447     ,
        0.44499999,         inf,  0.44100001,  0.43799999,  0.45300001,
        0.456     ,  0.45300001,  0.456     ,  0.46000001,  0.47      ,
        0.47099999,  0.47600001,  0.472     ,  0.477     ,  0.479     ,
        0.47600001,  0.49200001,  0.49200001,  0.491     ,  0.49700001,
        0.50400001,  0.50400001,  0.50599998,  0.505     ,  0.51800001,
        0.51300001,  0.51999998,  0.53100002,  0.53299999,  0.53500003,
        0.53500003,  0.54000002,  0.55000001,  0.53600001,  0.551     ,
        0.56300002,  0.56800002,  0.56300002,  0.57499999,  0.57499999,
        0.56999999,  0.59200001,  0.58899999,  0.60299999,  0.61400002,
        0.61500001,  0.61699998,  0.62599999,  0.62400001,  0.639     ,
        0.639     ,  0.64300001,  0.65399998,  0.65499997,         inf,
        0.67900002,  0.68199998,  0.68099999,  0.699     ,  0.699     ,
        0.70999998,  0.722     ,  0.73500001,  0.736     ,  0.745     ,
        0.75      ,  0.759     ,  0.773     ,  0.77600002,  0.78399998,
        0.79699999,         inf,         inf,         inf,  2.64899993,
        2.65199995,  2.64100003,  2.6500001 ,  2.63800001,  2.6400001 ,
        2.63899994,  2.64499998,  2.62599993,  2.63000011,  2.63000011,
        2.62800002,  2.62899995,  2.61800003,  2.61500001,  2.61899996,
        2.61599994,  2.61999989,  2.61500001,  2.61500001,  2.6099999 ,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,  2.87199998,  2.875     ,
        2.88400006,  2.87400007,  2.88100004,  2.87599993,  2.87100005,
        2.87199998,  2.87899995,  2.87199998,  2.87299991,  2.875     ,
        2.86500001,  2.86299992,  2.86100006,  2.86199999,  2.86400008,
        2.86199999,  2.852     ,  2.86800003,  2.86899996,  2.85599995,
        2.85100007,         inf,         inf,  3.36100006,  3.35400009,
        3.36500001,  3.36299992,  3.352     ,  3.36800003,  3.35500002,
        3.35400009,  3.3599999 ,  3.35299993,  3.35599995,  3.34500003,
        3.35400009,  3.352     ,  3.35500002,  3.35599995,  3.34100008,
        3.35400009,  3.34699988,  3.34599996,  3.34899998,  3.35299993,
        3.35400009,  3.3499999 ,  3.35400009,  3.35100007,  3.34899998,
        3.3599999 ,  3.352     ,  3.35599995,  3.35700011,  3.34699988,
        3.35400009,  3.3599999 ,  3.35700011,  3.35500002,  3.34800005,
        3.35100007,  3.36500001,  3.3499999 ,  3.36500001,  3.36800003,
        3.36400008,  3.36400008,  3.35800004,  3.35899997,  3.37100005,
        3.37100005,  3.36800003,  3.36599994,  3.36899996,  3.36899996,
        3.36500001,  3.37800002,  3.38000011,  3.38199997,  3.38000011,
        3.38700008,  3.38599992,  3.39400005,  3.3900001 ,  3.39299989,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,         inf,
               inf,  3.95000005,  3.95300007,  3.88100004,  3.91599989,
        3.9059999 ,  3.91700006,  3.91899991,  3.93199992,  3.93400002,
        3.9289999 ,  3.94099998,  3.93799996,  3.94899988,  3.95000005,
        3.96199989,  3.96399999,  3.96099997,  3.96300006,  3.9690001 ,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,  6.5       ,
               inf,         inf,         inf,         inf,         inf,
               inf,  6.09200001,         inf,  5.96799994,  5.9289999 ,
               inf,         inf,         inf,         inf,  5.65899992,
        5.62099981,         inf,  5.48000002,  5.4460001 ,  5.41900015,
        5.375     ,  5.33300018,         inf,  5.25      ,  5.21099997,
        5.16900015,  5.13199997,  5.09100008,  5.046     ,  5.02400017,
        4.97900009,  4.94199991,  4.89799976,         inf,  4.40600014,
               inf,         inf,         inf,  4.3499999 ,         inf,
               inf,         inf,         inf,  4.39599991,  4.38800001,
        4.40100002,  4.39799976,  4.40700006,  4.41400003,  4.43400002,
        4.44299984,  4.44000006,  4.46700001,  4.46500015,  4.48000002,
        4.49300003,  4.50099993,         inf,         inf,         inf,
               inf,         inf,         inf,         inf,         inf,
               inf,  1.60899997,  1.58099997,         inf,  1.51699996,
        1.50899994,  1.50999999,  1.5       ,  1.48300004,  1.48300004,
        1.47899997,  1.47899997,  1.48599994,  1.45700002,  1.454     ,
        1.42299998,  1.43200004,  1.43900001,  1.43400002,  1.41900003,
        1.40199995,  1.40699995,  1.39499998,  1.39600003,  1.39499998,
        1.38900006,  1.37800002,  1.37899995,  1.37800002,  1.38100004,
        1.37199998,  1.36099994,  1.36500001,  1.33899999,  1.33200002,
        1.33700001,  1.36099994,  1.33099997,  1.34300005,  1.32000005,
        1.31799996,  1.296     ,  1.31200004,  1.31500006,  1.35000002,
        1.32700002,  1.32000005,  1.35500002,  1.329     ,  1.33500004,
        1.34899998,         inf,  1.39699996,  1.40199995,  1.40900004,
        1.40499997,  1.41100001,  1.41100001,  1.42299998,  1.42900002,
        1.43099999,  1.44000006,  1.44299996,  1.44500005,  1.49600005,
               inf,  1.51800001,  1.52100003,         inf,         inf,
               inf,         inf,         inf,         inf,  1.90699995,
        1.91299999,  1.91999996,  1.91299999,  1.921     ,  1.93499994,
        1.93900001,  1.95200002,  1.95099998,  1.94799995,         inf,
               inf,         inf,         inf,  1.72000003,  1.71200001,
        1.71200001,  1.71700001,  1.70799994,  1.71200001,  1.70700002,
        1.69700003,  1.70099998,  1.70299995,  1.69099998,  1.69400001,
        1.69099998,  1.70599997,  1.70099998,  1.71300006,  1.71000004,
        1.722     ,  1.72000003,         inf,  1.73399997,  1.74300003,
        1.74399996,  1.76400006,  1.78299999,         inf,         inf,
               inf,  3.171     ,  3.14899993,  3.18099999,  3.18300009,
        3.18499994,         inf,  6.4289999 ,  6.41699982,  6.40899992,
               inf,  1.70500004,  1.70700002,  1.699     ,  1.722     ,
        1.68599999,  1.70299995,  1.68400002,  1.69500005,  1.68799996,
        1.68099999,  1.68499994,  1.66600001,  1.67799997,  1.68700004,
        1.67499995,  1.67499995,  1.66299999,  1.65100002,  1.66100001,
        1.68400002,  1.69200003,         inf,  1.73199999,         inf,
        1.63999999,  1.64900005,  1.63900006,  1.62300003,  1.62399995,
        1.59399998,  1.59899998,  1.59800005,  1.59800005,  1.59599996,
        1.58800006,  1.58800006,  1.58800006,  1.58700001,  1.58899999,
        1.574     ,  1.57599998,  1.574     ,  1.57299995,  1.57500005,
        1.56599998,  1.57599998,  1.57599998,  1.56200004,  1.56099999,
        1.56099999,  1.56799996,  1.55299997,  1.56700003,  1.55499995,
        1.55400002,  1.55200005,  1.55400002,  1.54999995,  1.55599999,
        1.55200005,  1.54299998,  1.54100001,  1.53999996,  1.54100001,
        1.546     ,  1.54400003,  1.54299998,  1.54499996,  1.53900003,
        1.51400006,  1.53999996,  1.53999996,  1.53100002,  1.53400004,
        1.52900004,  1.53600001,  1.53100002,  1.52900004,  1.52999997,
        1.53299999,  1.53400004,  1.51999998,  1.52199996,  1.52499998,
        1.52199996,  1.52199996,  1.51900005,  1.52100003,  1.52199996,
        1.523     ,  1.51300001,  1.523     ,  1.51100004,  1.51300001,
        1.51999998,  1.51199996,  1.51499999,         inf,         inf,
               inf,         inf,         inf,  0.90700001,  0.90100002,
        0.91100001,  0.90499997,  0.89499998,  0.89200002,  0.89899999,
        0.89499998,  0.89099997,  0.88999999,  0.884     ,  0.88200003,
        0.89600003,  0.884     ,  0.88700002,  0.88      ,  0.87      ,
        0.88300002,  0.87699997,  0.87400001,  0.88      ,  0.88700002,
        0.88      ,  0.87199998,  0.89399999,  0.87800002,         inf,
               inf,         inf,  1.04499996,  0.99900001,  1.01699996,
               inf,         inf,  6.38700008,  6.375     ,  6.41400003,
        6.41099977,         inf,         inf,         inf,  1.08500004,
        1.08299994,  1.06799996,  1.07000005,  1.06400001,  1.08299994,
        1.08299994,         inf,         inf,  1.24800003,  1.25199997,
               inf,  1.06400001,         inf,  1.23300004,         inf,
               inf,  1.13699996,         inf,  1.15199995,  1.15100002,
        1.17799997,         inf,  1.23399997,  1.24100006,  1.227     ,
               inf,  1.13100004,         inf,  1.12300003,         inf,
               inf,  1.08099997,  1.07700002,  1.08599997,  1.10399997,
               inf,         inf,         inf,         inf,         inf,
        1.199     ,         inf,  1.13900006,  1.13100004,  1.10699999,
               inf,         inf,         inf,         inf,  1.68599999,
        1.70799994,  1.70000005,  1.69799995,  1.704     ,  1.69200003,
               inf,         inf,         inf,  0.94599998,  0.94499999,
        0.92799997,  0.91900003,  0.903     ,  0.90399998,         inf,
        0.90399998,  0.89600003,  0.903     ,  0.90100002,  0.91299999,
        0.90799999,  0.91299999,  0.91000003,  0.921     ,  0.90600002,
        0.912     ,  0.926     ,  0.92000002,  0.92000002,  0.93099999,
        0.94      ,  0.93900001,  0.95599997,  0.95200002,         inf,
               inf,  1.62199998,  1.64499998,  1.66100001,  1.65999997,
        1.66600001,  1.67200005,  1.68200004,  1.67900002,  1.67700005,
        1.68200004,  1.68799996,  1.68200004,  1.69200003,  1.69099998,
        1.70299995,  1.71000004,  1.71099997,  1.70599997,  1.71599996,
        1.72399998,  1.71300006,  1.71500003,  1.72599995,  1.73199999,
        1.71300006,  1.72800004,  1.73300004,  1.727     ,  1.71899998,
        1.73399997,  1.73500001,  1.73699999,  1.74699998,  1.74600005,
        1.74600005,  1.75699997,  1.75699997,  1.75600004,  1.75399995,
        1.76699996,  1.76699996,  1.78400004,  1.78199995,  1.78100002,
        1.77999997,  1.79400003,  1.80700004,  1.80299997,  1.80299997,
        1.80499995,  1.81900001,  1.81700003,  1.82799995,  1.824     ,
        1.829     ,  1.83899999,  1.83800006,  1.85300004,  1.85500002,
        1.85000002,  1.86500001,  1.86399996,  1.87600005,  1.875     ,
        1.87899995,  1.88699996,  1.89699996,  1.903     ,  1.89900005,
        1.91299999,  1.91700006,  1.92299998,  1.92299998,  1.92499995,
        1.95899999,  1.949     ,  1.977     ,  1.96399999,  1.96500003,
        1.95099998,  1.93700004,  1.92900002,  1.92799997,  1.91299999,
        1.90499997,  1.903     ,  1.89400005,  1.87800002,  1.86600006,
        1.86899996,  1.85699999,  1.847     ,  1.84500003,  1.83500004,
        1.83000004,  1.82299995,  1.82099998,  1.80799997,  1.79799998,
        1.79700005,  1.78400004,  1.76999998,  1.76100004,  1.76100004,
        1.75899994,  1.74899995,  1.73800004,  1.73599994,  1.72099996,
        1.722     ,         inf,         inf,  1.44099998,  1.43200004,
        1.42999995,  1.42799997,  1.41400003,  1.41499996,  1.40600002,
        1.40100002,  1.40199995,  1.39400005,  1.38999999,  1.37800002,
        1.37800002,  1.36600006,  1.36600006,  1.35099995,  1.35699999,
        1.35399997,  1.34399998,  1.34000003,  1.32599998,  1.329     ,
        1.33099997,  1.31099999,  1.31500006,  1.31299996,  1.31400001,
        1.30499995,  1.30499995,  1.30299997,  1.30200005,  1.28699994,
        1.28900003,  1.27900004,  1.28299999,  1.28100002,  1.27999997,
        1.26900005,  1.26600003,  1.25999999,  1.25399995,  1.255     ,
        1.25399995,  1.25      ,  1.25699997,  1.25399995,  1.24399996,
        1.24300003,  1.23099995,  1.24000001,         inf,  1.15100002,
        1.13300002,  1.125     ,  1.13100004,         inf,  1.13100004,
        1.11699998,  1.11800003,  1.12100005,  1.12199998,  1.10699999,
        1.11099994,  1.10899997,  1.10699999,  1.097     ,  1.09500003,
        1.097     ,  1.097     ,  1.09500003,  1.09800005,  1.09500003,
        1.08500004,  1.08700001,  1.08099997,  1.08500004,  1.08299994,
        1.074     ,  1.07299995,  1.07599998,  1.06299996,  1.07000005,
        1.05700004,  1.06299996,  1.06299996,  1.06500006,  1.05999994,
        1.04799998,  1.05299997,  1.05299997,  1.04799998,  1.04900002,
        1.051     ,  1.051     ,  1.051     ,  1.04999995,  1.03499997,
        1.03999996,  1.03900003,  1.03799999,  1.028     ,  1.03299999,
        1.03699994,  1.02699995,  1.03199995,  1.02600002,  1.02699995,
        1.02900004,  1.02699995,  1.023     ,  1.02499998,  1.01699996,
        1.01300001,  1.01400006,  1.01699996,  1.01499999,  1.01499999,
        1.00399995,  0.99800003,  1.01499999,  1.00199997,  1.00399995,
        1.00100005,  1.00199997,  1.005     ,  1.00399995,  1.00199997,
        0.991     ,  1.00600004,  0.99400002,  0.991     ,  0.99400002,
        0.98699999,  0.995     ,  0.991     ,  0.991     ,  0.98100001,
        0.99299997,  0.98299998,  0.98000002,  0.98100001,  0.98100001,
        0.98299998,  0.98400003,  0.98199999,  0.98400003,  0.98500001,
        0.986     ,  0.986     ,  0.98400003,  0.972     ,  0.98500001,
        0.972     ,  0.97299999,  0.972     ,  0.97399998,  0.98500001,
        0.97500002,  0.97399998,  0.98299998,  0.97500002,  0.97899997,
               inf,  1.023     ,         inf,  1.08000004,         inf,
        1.14400005,  1.13900006,         inf,  1.222     ,  1.22500002,
        1.22399998,  1.23099995,  1.21300006,  1.22500002,  1.21300006,
        1.21300006,  1.21300006,  1.21200001,  1.21200001,  1.21500003,
        1.21200001,  1.21300006,  1.21800005,  1.21300006,  1.21300006,
        1.21399999,  1.21200001,  1.21500003,  1.21300006,  1.21300006,
        1.21200001,  1.21399999,  1.21500003,  1.21399999,  1.21399999,
        1.21300006,  1.21200001,  1.21399999,  1.21200001,  1.21200001]

ranges1 =[ 0.04851606,  0.14400939,  0.23707367,  0.33045234,  0.41635129,
        0.48981425,  0.44406407,  0.50625015,  0.56587653,  0.62264643,
        0.69263805,  0.74512521,  0.79257206,  0.83539654,  0.87338279,
        0.90604478,  0.93397282,  0.95742522,  0.97445885,  0.9859502 ,
        0.99278871,  0.9994261 ,  0.99934805,  0.99433002,  0.984492  ,
        0.96998821,  0.95088481,  0.9274373 ,  0.8997487 ,  0.87458029,
        0.8391186 ,  0.79228054,  0.7488536 ,  0.69614148,  0.64564879,
        0.58996556,  0.53470366,  0.47876219,  0.41918608,  0.35745897,
        0.29506412,  0.23613179,  0.16872702,  0.10142985,  0.03383415,
       -0.03381358, -0.10105739, -0.16784533, -0.23382789, -0.29879912,
       -0.36243419, -0.42464064, -0.48716086, -0.54543895, -0.60129726,
       -0.66207189, -0.71223457, -0.75902948, -0.80222761, -0.84161554,
       -0.87699651, -0.9081915 , -0.93250194, -0.95499957, -0.97343655,
       -0.98696012, -0.99570012, -0.99975318, -0.99889008, -0.99324888,
       -0.98281828, -0.96760263, -0.9476431 , -0.9229385 , -0.89716296,
       -0.8636747 , -0.82578914, -0.78363524, -0.73729353, -0.68679562,
       -0.63285227, -0.57548995, -0.51504382, -0.4518498 , -0.38624201,
       -0.31862736, -0.2493831 , -0.17888891, -0.1075814 , -0.04884264]

# 创建一个x轴的索引
#x = list(range(len(ranges)))
x1 = list(range(len(ranges1)))
# 绘制曲线图
#plt.plot(x, ranges, marker='o')
plt.plot(x1, ranges1, marker='o')
# 添加标题和标签
plt.title('Range Curve')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示图形
plt.show()