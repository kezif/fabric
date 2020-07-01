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


For purpose of finding best parametrs for model I use least square method and scipy build in minimize function. Also function parametrs have constraint ( for example fi2 can only be from -pi to pi). Because of that L-BFGS-B would be used 

### initial parametrs

If we look closely at data, we can find some paterns. It become more obvious in cartesian coordinates:
![catr cord data](https://i.imgur.com/TApvx2z.png)


So brobably r0 can't be less then smallest point, r1 be any between max and min point:
![r pars](https://i.imgur.com/wuDCwnI.png)


To find n - number of petals, I mapped each data point to iether 1 or -1  reduced and count total/2:
```python
wave_len = data[:,0]
middle = np.median(wave_len)  # find median

wave_len = wave_len - middle  # split values into lower and higher then med
wave_len = wave_len[wave_len != 0. ]  # remove 0 values
plt.plot(wave_len)
#scale = -1 * np.min(wave_len) if np.sign(wave_len) < 0 else np.max(wave_len)
plt.plot(np.sign(wave_len )* np.max(wave_len), c='r')
sign = np.sign(wave_len )  # map vales into -1 if lower or 1
w = sign 
w = w[1:][w[1:] != w[:-1]]  # remove values if there are repeated: 1,1,1,1,-1 -> 1,-1
int(len(w) // 2)
# 5
```

![n](https://i.imgur.com/0LbVh2s.png)

## Result
Fiting result with chosen initial parametrs for clice at height - 25

![res](https://i.imgur.com/tW5Faw6.png)
