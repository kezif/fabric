# fabric
Program for extract data from 3d image of fabric.

Fabric samples that would be processed look something like this:
![fabric](https://i.imgur.com/1yxWBOp.png)

Point of data excraction is to slice data into 4+1 layers (+1 stands for shadow) 

You can (obviously) load scanned image and excract data and slice it right away 
![loaded image](https://i.imgur.com/Qfg7hcU.png)

And change exctaraction parametrs on the fly (boolean indexing speed 😎)
![gif parametr change](https://i.imgur.com/pylUvq2.gif)

Also my program have some preprocessing for handling missing data, converting to polar coordinates, centering scanned image and aggregating into consistend number of data points (from n = any to n = 130 for each layer)

## Modeling
After defining correct parametrs you can press save data and build models.
This formula is used for each layer: 

![r0+r1*(\dfrac{1+\sin(n*\theta+\Delta fi1)}{2})^{k1}+r2 * (\dfrac{1 + sin(2 * \theta + \Delta fi2)}{2})^{k2} ](https://render.githubusercontent.com/render/math?math=r0%2Br1*(%5Cdfrac%7B1%2B%5Csin(n*%5Ctheta%2B%5CDelta%20fi1)%7D%7B2%7D)%5E%7Bk1%7D%2Br2%20*%20(%5Cdfrac%7B1%20%2B%20sin(2%20*%20%5Ctheta%20%2B%20%5CDelta%20fi2)%7D%7B2%7D)%5E%7Bk2%7D%20)

For purpose of finding best parametrs for model i use least square method and scipy build in minimize function

```Here we can see a lot of explanations about fitting, shoosing initial parametrs, and constraining it```
