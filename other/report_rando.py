import matplotlib.pyplot as plt
import numpy as np


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (5,5)
plt.axes().set_aspect('equal')
lim = 2.5
x = [1,-1,-1,1]
y =[1,1,-1,-1]
labels = ["00","01","11","10"]
plt.scatter(x,y,zorder=10,c='black')

x1, y1 = 0.7, 0.5
plt.plot((0,x1),(y1,y1),c='r',alpha=1,zorder=1)
plt.plot((x1,x1),(0,y1),c='r',alpha=1,zorder=1)
plt.scatter(x1,y1,color='black')

xg = np.linspace(-lim,lim,100,endpoint=True)
yg = gaussian(xg,1,0.5)
plt.plot(xg,yg + 1,alpha=0.3,c='black')
#xg_cut = np.array([a for a in xg if a > x1])
#plt.fill_between(xg_cut,gaussian(xg_cut,1,y1)+1,1,zorder=0.5,alpha=0.5,facecolor='b')
plt.plot((1,x1),(1,1),zorder=1,c='red',lw=1.5)

for i in range(len(x)):
    plt.annotate(labels[i],(x[i]+0.1,y[i]+0.05),fontsize=20,fontname='serif')

plt.annotate('$y_r$', xy=(0.25, 0.6),
             fontname = 'serif',
             fontsize = 20,
           )
plt.annotate('$y_i$', xy=(0.75, 0.15),
             fontname = 'serif',
             fontsize = 20,
           )
#plt.annotate("$P(y_r|Bit_2 = 0) = P(N_r = 1-y_r)$",(-1,2),font='serif',fontsize=15,color='black')
plt.annotate('$N_r$',(0.7,1.2),font='serif',fontsize=15,color='red')
plt.ylim((-lim,lim))
plt.xlim((-lim,lim))
plt.axhline(0, color='gray', linewidth=1)
plt.axvline(0, color='gray', linewidth=1)
plt.xticks(fontname = 'serif', fontsize=15)
plt.yticks(fontname = 'serif', fontsize=15)
plt.show()
