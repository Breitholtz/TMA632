import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_1D_A=pd.read_csv('sols_results_1d_newdata_1.csv')
data_1D_B=pd.read_csv('sols_results_1d_newdata_x*sinx.csv')
data_2D_A=pd.read_csv('sols_results_2d_newdata_(1,1).csv')
data_2D_B=pd.read_csv('sols_results_2d_newdata_(x*x*sin(y), y*y*cos(x)).csv')

data_1D_A["dt"]=[10/float(x) for x in data_1D_A["num_steps_time"].as_matrix()]
data_1D_B["dt"]=[10/float(x) for x in data_1D_B["num_steps_time"].as_matrix()]
data_2D_A["dt"]=[10/float(x) for x in data_2D_A["num_steps_time"].as_matrix()]
data_2D_B["dt"]=[10/float(x) for x in data_2D_B["num_steps_time"].as_matrix()]

data_1D_A["dx"]=[2*np.pi/float(x) for x in data_1D_A["mesh_size"].as_matrix()]
data_1D_B["dx"]=[2*np.pi/float(x) for x in data_1D_B["mesh_size"].as_matrix()]
data_2D_A["dx"]=[2*np.pi/float(x) for x in data_2D_A["mesh_size"].as_matrix()]
data_2D_B["dx"]=[2*np.pi/float(x) for x in data_2D_B["mesh_size"].as_matrix()]

data_2D_B=data_2D_B.query("l2_compare<=10")




"""
for a, b in data_1D_A.groupby("mesh_size"):
    num_steps=np.log2(b["dt"].as_matrix())
    l2_compare=np.log2(b["l2_compare"].as_matrix())

    plt.plot(num_steps,
             l2_compare,
             'ro',
             markerfacecolor="white")
for a, b in data_1D_B.groupby("mesh_size"):
    num_steps=np.log2(b["dt"].as_matrix())
    l2_compare=np.log2(b["l2_compare"].as_matrix())

    plt.plot(num_steps,
             l2_compare,
             'bo',
             markerfacecolor="white")

for a, b in data_2D_A.groupby("mesh_size"):
    num_steps=np.log2(b["dt"].as_matrix())
    l2_compare=np.log2(b["l2_compare"].as_matrix())

    plt.plot(num_steps,
             l2_compare,
             'ko',
             markerfacecolor="white")

for a, b in data_2D_B.groupby("mesh_size"):
    num_steps=np.log2(b["dt"].as_matrix())
    l2_compare=np.log2(b["l2_compare"].as_matrix())

    plt.plot(num_steps,
             l2_compare,
             'go',
             markerfacecolor="white")


"""
#plt.subplots()

dom=[]
y=[]
for a, b in data_1D_A.groupby("dt"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'ro',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'ro',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="red",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_1D_B.groupby("dt"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
#    plt.plot(np.log2(a),
#             mean_l2,
#             'bo',
#             markerfacecolor="black",
#             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'bo',
         markerfacecolor="black",
         markeredgewidth=1.5)

lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="blue",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_A.groupby("dt"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'yo',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'yo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="yellow",
         ls="--",
         label="S=%.02f"%lf[0])
dom=[]
y=[]
for a, b in data_2D_B.groupby("dt"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
#             mean_l2,
#             'go',
#             markerfacecolor="black",
#             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'go',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="green",
         ls="--",
         label="S=%.02f"%lf[0])

plt.title("(Loglog Plot) L2 Error; Difference between standard and first-order solution. \n"r"Red: B=$(1)$. Blue: B=$(xsinx)$." "\n"r"Yellow: B=$(1,1)$. Green: B=$(x^2siny,y^2cosx)$")
plt.ylabel("Mean of L2 Error over Different dx")
plt.xlabel("dt")
plt.legend(loc="best")

plt.subplots()

dom=[]
y=[]
for a, b in data_1D_A.groupby("dx"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'ro',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)

plt.plot(dom,
         y,
         'ro',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="red",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_1D_B.groupby("dx"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'bo',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'bo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="blue",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_A.groupby("dx"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'yo',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'yo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="yellow",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_B.groupby("dx"):
    mean_l2=np.log2(np.mean(b["l2_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)

    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'go',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)

plt.plot(dom,
         y,
         'go',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="green",
         ls="--",
         label="S=%.02f"%lf[0])


plt.title("(Loglog Plot) L2 Error; Difference between standard and first-order solution. \n"r"Red: B=$(1)$. Blue: B=$(xsinx)$." "\n"r"Yellow: B=$(1,1)$. Green: B=$(x^2siny,y^2cosx)$")
plt.ylabel("Mean of L2 Error over Different dt")
plt.xlabel("dx")
plt.legend(loc="best")


plt.subplots()


dom=[]
y=[]
for a, b in data_1D_A.groupby("dt"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'ro',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)

plt.plot(dom,
         y,
         'ro',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="red",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_1D_B.groupby("dt"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'bo',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'bo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="blue",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_A.groupby("dt"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)

    plt.plot(np.log2(a),
             mean_l2,
             'yo',
             markerfacecolor="black",
             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'yo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="yellow",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_B.groupby("dt"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)

    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'go',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'go',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="green",
         ls="--",
         label="S=%.02f"%lf[0])

plt.title("(Loglog Plot) H1 Error; Difference between standard and first-order solution. \n"r"Red: B=$(1)$. Blue: B=$(xsinx)$." "\n"r"Yellow: B=$(1,1)$. Green: B=$(x^2siny,y^2cosx)$")
plt.ylabel("Mean of H1 Error over Different dx")
plt.xlabel("dt")
plt.legend(loc="best")


plt.subplots()

dom=[]
y=[]
for a, b in data_1D_A.groupby("dx"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    #plt.plot(np.log2(a),
    #         mean_l2,
    #         'ro',
    #         markerfacecolor="black",
    #         markeredgewidth=1.5)
plt.plot(dom,
         y,
         'ro',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="red",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_1D_B.groupby("dx"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)
    plt.plot(np.log2(a),
             mean_l2,
             'bo',
             markerfacecolor="black",
             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'bo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="blue",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_A.groupby("dx"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)

    plt.plot(np.log2(a),
             mean_l2,
             'yo',
             markerfacecolor="black",
             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'yo',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="yellow",
         ls="--",
         label="S=%.02f"%lf[0])

dom=[]
y=[]
for a, b in data_2D_B.groupby("dx"):
    mean_l2=np.log2(np.mean(b["h1_compare"]))
    dom.append(np.log2(a))
    y.append(mean_l2)


    plt.plot(np.log2(a),
             mean_l2,
             'go',
             markerfacecolor="black",
             markeredgewidth=1.5)
plt.plot(dom,
         y,
         'go',
         markerfacecolor="black",
         markeredgewidth=1.5)
lf=np.polyfit(dom, y, 1)
p=np.poly1d(lf)
plt.plot(dom,
         p(dom),
         color="green",
         ls="--",
         label="S=%.02f"%lf[0])


plt.title("(Loglog Plot) H1 Error; Difference between standard and first-order solution. \n"r"Red: B=$(1)$. Blue: B=$(xsinx)$." "\n"r"Yellow: B=$(1,1)$. Green: B=$(x^2siny,y^2cosx)$")
plt.ylabel("Mean of H1 Error over Different dt")
plt.xlabel("dx")
plt.legend(loc="best")
plt.show()
