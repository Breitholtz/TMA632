import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


##
# Visualisera error då dx är konstant
# data är en dataframe enligt den struktur som finns i csv filerna
# error_dict är en dict som kolumntitel som key och en korresponderande
# titel som value.
# model_title är en titel för modellen, som skrivs ut i titlen av ploten
# error är det error som vill undersökas. t.ex. "l2_full_true"
def visualize_dt_errors(data, error_dict, model_title, error):

    fig, ax = plt.subplots(figsize=(8,6))
    for a, b in data.groupby("dx"):
        
        time_steps=np.log2(b["dt"].as_matrix())
        error_vals=np.log2(b[error].as_matrix())
        b.sort_values(by="dt", inplace=True)
        linear_fit=np.polyfit(time_steps, error_vals, 1)
        ax.plot(time_steps,
                error_vals,
                'o-',
                label="dx=%.02f. Slope=%.02f"%(a, linear_fit[0]))
    ax.set_title("(Loglog Plot) %s \n %s"%(error_dict[error], model_title))
    ax.set_xlabel("dt")
    ax.set_ylabel("%s"%error_dict[error])
    ax.legend(loc="best")

    
##
# Visualisera error då dt är konstant
# data är en dataframe enligt den struktur som finns i csv filerna
# error_dict är en dict som kolumntitel som key och en korresponderande
# titel som value.
# model_title är en titel för modellen, som skrivs ut i titlen av ploten
# error är det error som vill undersökas. t.ex. "l2_full_true"
def visualize_dx_errors(data, error_dict, model_title, error):

    fig, ax = plt.subplots(figsize=(8,6))
    for a, b in data.groupby("dt"):
        dmesh=np.log2(b["dx"].as_matrix())
        error_vals=np.log2(b[error].as_matrix())
        linear_fit=np.polyfit(dmesh, error_vals, 1)
        ax.plot(dmesh,
                error_vals,
                'o-',
                label="dt=%.04f. Slope=%.02f"%(a, linear_fit[0]))
    ax.set_title("(Loglog Plot) %s \n %s"%(error_dict[error], model_title))
    ax.set_xlabel("dmesh")
    ax.set_ylabel("%s"%error_dict[error])
    ax.legend(loc="best")
        
##
# Visualisera två olika kolumner i en scatter plot.

def scatter_errors(data, X_col, Y_col, dict_error_names, title, model_title):

    fig, ax = plt.subplots(figsize=(8,6))

    X=data[X_col].as_matrix()
    Y=data[Y_col].as_matrix()

    ax.plot(np.log(X), np.log(Y), 'ro', markerfacecolor="white")
    ax.set_title("%s \n %s"%(title, model_title))
    ax.set_xlabel(dict_error_names[X_col])
    ax.set_ylabel(dict_error_names[Y_col])
    ax.axis("equal")

##
# Visualisera en tabell
def visualize_table(data, row_labels, col_labels):
    # Add a table at the bottom of the axes

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    
    ax.table(cellText=data,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
    ax.set_title("f")

    fig.tight_layout()


##
# Ladda Data
data_1D_A=pd.read_csv('sols_results_1d_1.csv')
data_1D_B=pd.read_csv('sols_results_1d_x*sinx.csv')
data_2D_A=pd.read_csv('sols_results_2d_(1.1).csv')
data_2D_B=pd.read_csv('sols_results_2d_(x*x*sin(y), y*y*cos(x)).csv')
data_3D_A=pd.read_csv('sols_results_3d_(sinz,cosx,siny).csv')
data_3D_B=pd.read_csv('sols_results_3d_(1,1,1).csv')


# Add dt to Data Files
data_1D_A["dt"]=[10/float(x) for x in data_1D_A["num_steps_time"].as_matrix()]
data_1D_B["dt"]=[10/float(x) for x in data_1D_B["num_steps_time"].as_matrix()]
data_2D_A["dt"]=[10/float(x) for x in data_2D_A["num_steps_time"].as_matrix()]
data_2D_B["dt"]=[10/float(x) for x in data_2D_B["num_steps_time"].as_matrix()]
data_3D_A["dt"]=[10/float(x) for x in data_3D_A["num_steps_time"].as_matrix()]
data_3D_B["dt"]=[10/float(x) for x in data_3D_B["num_steps_time"].as_matrix()]
# Add dx

data_1D_A["dx"]=[2*np.pi/float(x) for x in data_1D_A["mesh_size"].as_matrix()]
data_1D_B["dx"]=[2*np.pi/float(x) for x in data_1D_B["mesh_size"].as_matrix()]
data_2D_A["dx"]=[2*np.pi/float(x) for x in data_2D_A["mesh_size"].as_matrix()]
data_2D_B["dx"]=[2*np.pi/float(x) for x in data_2D_B["mesh_size"].as_matrix()]
data_3D_A["dx"]=[2*np.pi/float(x) for x in data_3D_A["mesh_size"].as_matrix()]
data_3D_B["dx"]=[2*np.pi/float(x) for x in data_3D_B["mesh_size"].as_matrix()]

##
# Dict som relaterar en kolumntitel till en titel för metoden
dict_error_names={'l2_full_true':'L2 Error; Standard Method',
                  'h1_full_true':'H1 Error; Standard Method',
                  'l2_full_fine':'L2 Error wr. to Fine Solution; Standard',
                  'h1_full_fine':'H1 Error wr. to Fine Solution; Standard',
                  'l2_fos_true':'L2 Error; First-Order',
                  'h1_fos_true':'H1 Error; First-Order',
                  'l2_fos_fine':'L2 Error wr. to Fine Solution; First-Order',
                  'h1_fos_fine':'H1 Error wr. to Fine Solution; First Order'}



## Exempel på hur man kan visualizera error convergence.
# Ändra data till den modell som vill undersökas
# Ändra model_title till den title du vill ha i ploten
model_title=r"B=$(sin(z),cos(x),sin(y))$"
data=data_3D_B
for i in range(len(list_of_errors)):
    visualize_dx_errors(data,
                        dict_error_names,
                        model_title,
                        list_of_errors[i])
                        
plt.show()
