import pandas as pd
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy as sp
from qspmodel import *
import numpy as np

# Checking or creating necessary output folders
if not os.path.exists('Data/'):
    os.makedirs('Data/Dynamic/')
    os.makedirs('Data/GlobalSensitivity/')
else:
    if not os.path.exists('Data/Dynamic/'):
        os.makedirs('Data/Dynamic/')
    if not os.path.exists('Data/GlobalSensitivity/'):
        os.makedirs('Data/GlobalSensitivity/')

# some global parameters
lmod=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #indices of immune cells variables in cell data (Excluding Naive-cells from the microenvironment)
clusters=5 #number of clusters

T=3000
t=np.linspace(0, T, 30001)

nvar=Breast_QSP_Functions().nvar # number of variables
nparam=Breast_QSP_Functions().nparam # number of parameters
################################################################################
###########################Reading data#########################################
clustercells = pd.read_csv('input/input/Breast_Large_Tumor_cell_data.csv')
clustercells = clustercells.to_numpy()

meanvals = pd.read_csv('input/input/mean_variable.csv')
meanvals = meanvals.to_numpy()

#Getting the parameters
for cluster in range(clusters):
    QSP_=QSP.from_cell_data(clustercells[cluster])
    params=QSP_.p
    print(params)
################################################################################
###########################Solving ODE##########################################
# reading initial conditions
IC=pd.read_csv('input/input/breast_IC_ND.csv')  #non-dimensional IC
IC = IC.to_numpy()

for cluster in range(clusters):
     print('Starting computations for cluster '+str(cluster+1))
     filename='Cluster-'+str(cluster+1)+'-results-'

     QSP_=QSP.from_cell_data(clustercells[cluster])
     params=QSP_.p

     print(' Parameters set. Computing the solution')

     u, _ = QSP_.solve_ode(t, IC[cluster], 'given')
     u = clustercells[cluster]*u

     wr=np.empty((t.size, 18))
     wr[:,0]=t
     wr[:,1:]=u
     c=csv.writer(open('Data/Dynamic/'+filename+'dat.csv',"w"))
     c.writerow(['time']+QSP_.variable_names())
     c.writerows(wr)
     del c
################################################################################
###########################Plotting ODE#########################################
dynamic_all = []
for c in [1,2,3,4,5]:
    dat = pd.read_csv('Data/Dynamic/Cluster-'+str(c)+'-results-dat.csv')
    #Summing the cells present in the micro-environment
    dat['Total cells'] = dat[dat.columns[2]]+dat[dat.columns[3]]+dat[dat.columns[4]]+dat[dat.columns[5]]+dat[dat.columns[6]]+0.2*dat[dat.columns[7]]+dat[dat.columns[8]]+dat[dat.columns[9]]+dat[dat.columns[10]]+dat[dat.columns[11]]
    dat['Cluster'] = c
    dynamic_all.append(dat)

dynamic_all_df = pd.concat(dynamic_all, axis=0)

palette={'Cluster 1':'#3F9B0B', 'Cluster 2':'#FF796C', 'Cluster 3':'#0343DF','Cluster 4':'#000000', 'Cluster 5':'#D5B60A'}
dynamic_all_df['Cluster'] = dynamic_all_df['Cluster'].apply(lambda x: 'Cluster '+str(x))
custom_lines = [Line2D([0], [0], color='#3F9B0B', lw=3.5),
                Line2D([0], [0], color='#FF796C', lw=3.5),
                Line2D([0], [0], color='#0343DF', lw=3.5),
                Line2D([0], [0], color='#000000', lw=3.5),
                Line2D([0], [0], color='#D5B60A', lw=3.5)]
sns.set(font_scale=1.5)
sns.set_style("ticks")
fig, axs = plt.subplots(6, 3, sharey=False, figsize=(10.5,13))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
axs = axs.flatten()

for i, col in enumerate(dynamic_all_df.columns[1:-1]):
    axs[i]=sns.lineplot(data=dynamic_all_df, x='time', y=col, hue='Cluster', palette=palette, ax=axs[i], legend=False)
    axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axs[i].set_xlabel("time (days)",fontsize=14)
    axs[i].set_ylabel(str(col),fontsize=14)
    axs[i].margins(x=0)


axs[9].legend(custom_lines, ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'], bbox_to_anchor=(3.98, 0.5),loc='center left',fontsize='x-small')
plt.savefig('fig/dynamics.eps', format='eps')
plt.show()
################################################################################
#######################Sensitivity Analysis#####################################

print('Starting steady state global sensitivity analysis')

# Read the parameter perturbation grid
globalgridname='lhs38-5000'
paramscalinggrid = pd.read_csv('input/input/'+globalgridname+'.csv', header=None).to_numpy().T
lhsnum=paramscalinggrid.shape[0]

# modify from (0,1) range to fit our needs
paramscalinggrid[:,:37]=pow(10,4*paramscalinggrid[:,:37]-2) # scale to values between 0.01 and 100 (10^-2 to 10^2)
paramscalinggrid[:,37]=paramscalinggrid[:,37]/0.75

# Read the local parameter perturbation grid
# level 1 or 0 corresponds to no local perturbation
gridlevel=2
sensitivity_radius=1 # percentage for local perturbation
if gridlevel>1:
    localgridname='Local-level'+str(gridlevel)
    filename='grid75-level'+str(gridlevel)
    data = pd.read_csv('input/input/'+filename+'.csv', header=None).to_numpy()
    w=data[:,0]
    x=data[:,1:]
    del data, filename
else:
    localgridname='Local-nogrid'
    w=np.array([1])
    x=[0]

# coefficients for variable sensitivity
lambda0=np.zeros((nvar,2))
lambda0[8,0]=1 # just cancer



import time

start = time.time()
k_filter_all_clusters = []

for cluster in range(0,5):
    print('Starting computations for cluster '+str(cluster+1))

    filename='V75-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'

    # Calculating all the parameters and weights
    basecore=Breast_QSP_Functions()
    baseparams=op.fsolve((lambda par,frac: basecore.SS_system(par,frac)),
                            np.ones(nparam),args=(clustercells[cluster],))

    paramarray=np.empty((lhsnum,nparam))
    weights=np.empty(lhsnum)
    nonnegs=np.ones(lhsnum, dtype=bool)
    for k in range(lhsnum):
        qspcore=Breast_QSP_Functions(SSrestrictions=paramscalinggrid[k])
        paramarray[k]=op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),
                                np.ones(nparam),args=(clustercells[cluster],))
        if (paramarray[k]<0).any(): nonnegs[k]=False
        weights[k]=np.sqrt(((paramarray[k]-baseparams)**2).sum())

    weights/=weights[nonnegs].min()
    weights=np.exp(-weights)
    weights/=weights[nonnegs].sum()

    lambda0[1:11,1]=clustercells[cluster,lmod]/np.sum(clustercells[cluster,lmod]) # all cells (except Tn)
    dudp=np.zeros((nparam,2))

    k_filter = []

    for k in range(lhsnum):
        if nonnegs[k]:
            for l in range(w.size):
                QSP_=QSP(paramarray[k]*(1+(sensitivity_radius*1e-2)*x[l]))
                cancer_sensitivity = np.dot(np.abs(QSP_.Sensitivity()),lambda0)[:,0]
                # filter out singularities
                if any(x > 10000 for x in cancer_sensitivity):
                    k_filter.append(k)
                    k_filter_all_clusters.append(k)

    k_filter = set(k_filter)
    k_new = [k for k in range(lhsnum) if k not in k_filter]

    for k in k_new:
        if nonnegs[k]:
            for l in range(w.size):
                QSP_=QSP(paramarray[k]*(1+(sensitivity_radius*1e-2)*x[l]))
                dudp=dudp+weights[k]*w[l]*np.dot(QSP_.Sensitivity(),lambda0)

    print(' Writing to file')

    c=csv.writer(open('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv',"w"))
    c.writerows(dudp)
    del c

end = time.time()
print('Run time: ', end - start)
print('Global sensitivity analysis complete')



Par_list =['\lambda_{T_hH}','\lambda_{T_hD}','\lambda_{T_hIL_{12}}','\lambda_{T_hE}',
                        '\lambda_{T_cE}','\lambda_{T_cD}','\lambda_{T_cIL_{12}}',
                        '\lambda_{T_rD}','\lambda_{T_rE}',
                        '\lambda_{DC}','\lambda_{DH}','\lambda_{DE}',
                        '\lambda_{MIL_{10}}','\lambda_{MI_{\gamma}}','\lambda_{MIL_{12}}','\lambda_{MT_h}','\lambda_{ME}',
                        '\lambda_{C}','\lambda_{CIL_6}','\lambda_{CA}',
                        '\lambda_{A}',
                        '\lambda_{HD}','\lambda_{HN}','\lambda_{HM}','\lambda_{HT_c}','\lambda_{HC}',
                        '\lambda_{IL_{12}M}','\lambda_{IL_{12}D}','\lambda_{IL_{12}T_h}','\lambda_{IL_{12}T_c}',
                        '\lambda_{IL_{10}M}','\lambda_{IL_{10}D}','\lambda_{IL_{10}T_r}','\lambda_{IL_{10}T_h}','\lambda_{IL_{10}T_c}','\lambda_{IL_{10}C}',
                        '\lambda_{EA}','\lambda_{E}',
                        '\lambda_{I_{\gamma}T_c}','\lambda_{I_{\gamma}T_h}','\lambda_{I_{\gamma}D}',
                        '\lambda_{IL_6A}','\lambda_{IL_6M}','\lambda_{IL_6D}',
                        '\delta_{T_hT_r}','\delta_{T_hIL_{10}}','\delta_{T_h}',
                        '\delta_{T_cIL_{10}}','\delta_{T_CT_r}','\delta_{T_c}',
                        '\delta_{T_r}',
                        '\delta_{T_N}',
                        '\delta_{DC}','\delta_{D}',
                        '\delta_{D_N}',
                        '\delta_{M}',
                        '\delta_{M_N}',
                        '\delta_{CT_c}','\delta_{CI_{\gamma}}','\delta_{C}',
                        '\delta_{A}',
                        '\delta_{N}',
                        '\delta_{H}',
                        '\delta_{IL_{12}}',
                        '\delta_{IL_{10}}',
                        '\delta_{E}',
                        '\delta_{I_{\gamma}}',
                        '\delta_{IL_6}',
                        'A_{T_N}','A_{D_N}','A_{M}',
                        '\\alpha_{NC}','C_0','A_0','E_0']

par_list = ["$"+x+"$" for x in Par_list]

for cluster in range(clusters):
    filename='V75-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
    sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
    sensitivity_df.index = par_list
    sensitive_ids = np.abs(sensitivity_df)[0][abs(sensitivity_df)[0]>0].nlargest(n=20).index
    print('Cluster ', cluster+1)
    print('Sensitivities:\n', sensitivity_df.loc[sensitive_ids])
################################################################################
######################Plotting Sensitivities####################################
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('xtick',labelsize=13)
fig, axs = plt.subplots(5, 2, sharey=False, figsize=(10,9))
fig.subplots_adjust(wspace=0.75, hspace=0.75)
axs[0, 0].set_title('Sensitivity of Cancer')
axs[0, 1].set_title('Sensitivity of Total cells count')

for cluster in range(clusters):
   filename='V75-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
   sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
   sensitivity_df.index = par_list
   for i in range(2):
       sensitive_ids = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index  #filtering singularities and finfing max
       sensitivity_df[i][sensitive_ids[:6]].plot.bar(ax=axs[cluster, i], rot=0, width=0.7)
       axs[cluster, i].axhline()
       axs[cluster, i].set_ylabel('Cluster '+str(cluster+1))

plt.savefig('fig/sensitivity1.eps', format='eps',dpi=300)
fig, axs = plt.subplots(5, 2, sharey=False, figsize=(10,9))
fig.subplots_adjust(wspace=0.75, hspace=0.75)
axs[0, 0].set_title('Sensitivity of Cancer')
axs[0, 1].set_title('Sensitivity of Total cells count')

for cluster in range(clusters):
   filename='V75-'+globalgridname+'-'+localgridname+'-cluster-'+str(cluster+1)+'-results-'
   sensitivity_df = pd.read_csv('Data/GlobalSensitivity/'+filename+'sensitivity_steady.csv', header=None)
   sensitivity_df.index = par_list
   ids_to_remove = []
   for i in range(2):
       sensitive_ids1 = np.abs(sensitivity_df)[i][abs(sensitivity_df)[i]>0].nlargest(n=20).index
       sensitivity_df[i][sensitive_ids1[6:12]].plot.bar(ax=axs[cluster, i], rot=0, width=0.7)
       axs[cluster, i].axhline()
       axs[cluster, i].set_ylabel('Cluster '+str(cluster+1))
plt.savefig('fig/sensitivity2.eps', format='eps',dpi=300)
################################################################################
########################Varying parameters######################################
def plot_cancer_vary_assumption_perturb_sensitive_params(assumption_idx, assumption_scale, perturb_scale, T):
   import seaborn as sns
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D

   restriction_map = {37: 'deltaCIgamma-deltaCTc', 36:'deltaCIgamma-deltaCTc-deltaC', 35:'lambdaCA-lambdaCIL6', 13:'lambdaMIgamma',\
                       16:'lambdaME', 15:'lambdaMTh', 14:'lambdaMIL10-LambdaMIL12', 34:'lambdaE-lambdaEA',0:'lambdaThH-lambdaThD',1: 'lambdaThIL12-LambdaThD'}
   perturb_map = {0: 'no perturbation', -perturb_scale: '-'+str(perturb_scale*10)+'%', perturb_scale: '+'+str(perturb_scale*10)+'%'}
   palette = {0:'#3F9B0B', 1:'#FF796C', 2:'#0343DF',3:'#000000',4:'#D5B60A'}
   alphas = [0.15, 0.15, 0.15,0.15,0.15]
   restrictions = np.ones(38)

   sns.set(font_scale=1.5)
   sns.set_style("ticks")
   fig, axs = plt.subplots(1, 3, sharey=False, figsize=(10.5,13))
   fig.subplots_adjust(wspace=0.5)
   axs = axs.flatten()
   t = np.linspace(0, T, 10*T+1)
   custom_lines = [Line2D([0], [0], color='#3F9B0B', lw=3.5),
                   Line2D([0], [0], color='#FF796C', lw=3.5),
                   Line2D([0], [0], color='#0343DF', lw=3.5),
                   Line2D([0], [0], color='#000000', lw=3.5),
                   Line2D([0], [0], color='#D5B60A', lw=3.5)]

   for i, newscale in enumerate([1, 1/assumption_scale, assumption_scale]):
       for cluster in range(5):
           restrictions[assumption_idx] = newscale
           qspcore = Breast_QSP_Functions(SSrestrictions = restrictions)
           new_params = op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),
                                        np.ones(nparam),args=(clustercells[cluster],))
           QSP_ = QSP(new_params)
           u, _ = QSP_.solve_ode(t, IC[cluster], 'given')
           u = clustercells[cluster]*u
           umax = umin = u
           for param_id in sensitive_param_ids:
               for j in [-perturb_scale, perturb_scale]:
                   perturb_arr = np.zeros(len(new_params))
                   perturb_arr[param_id] = j
                   QSP_ = QSP(new_params*(1+(1e-2)*perturb_arr))
                   u_perturb, _ = QSP_.solve_ode(t, IC[cluster], 'given')
                   u_perturb = clustercells[cluster]*u_perturb
                   umax = np.maximum(u_perturb, umax)
                   umin = np.minimum(u_perturb, umin)

           axs[i].margins(x=0)
           axs[i].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
           axs[i].fill_between(t, umax[:,8], umin[:,8], facecolor=palette[cluster], alpha=alphas[cluster])
           axs[i].plot(t, u[:,8], color=palette[cluster])
       axs[i].set_xlabel('time (days)',fontsize=14)
       axs[i].set_ylabel('Cancer cells',fontsize=14)
   if assumption_idx==16:
       axs[0].legend(custom_lines,['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'], bbox_to_anchor=(3.98, 0.5),loc='center left')

   plt.show()




restriction_map = {37: 'deltaCIgamma-deltaCTc', 36:'deltaCIgamma-deltaC', 35:'lambdaCA-lambdaCIL6', 13:'lambdaMIgamma-lambdaMIL10',\
                   16:'lambdaME-lambdaMIL10', 15:'lambdaMTh-lambdaMIL10'}
keys= [37,36,35,13,16,15]
sensitive_param_ids = [57,58,59,17,19,18,60,20,55,13,16,15,56,70,61]

for i in range(len(keys)):
   print('Varying', restriction_map[keys[i]], 'assumption + perturb sensitive params by 10%')
   plot_cancer_vary_assumption_perturb_sensitive_params(assumption_idx=keys[i], assumption_scale=5, perturb_scale=1, T=3000)

################################################################################
#####################Exchange parameters plots##################################

def plot_cross_cluster_IC(cluster, T, title=None):
   import matplotlib.pyplot as plt
   from matplotlib.lines import Line2D
   import seaborn as sns

   QSP_=QSP.from_cell_data(clustercells[cluster])
   clusters = [c for c in range(5) if c != cluster]
   dynamic_all = []

   for c in clusters:
       dynamic_df = pd.DataFrame(columns=['time', 'Cluster']+QSP_.variable_names())
       t = np.linspace(0, T, 10*T+1)
       dynamic_df['time'] = t
       dynamic_df['Cluster'] = ['Cluster '+str(c+1)]*len(dynamic_df['time'])

       u, _ = QSP_.solve_ode(t, IC[c], 'given')
       u = clustercells[cluster]*u
       dynamic_df[QSP_.variable_names()] = u
       dynamic_df['Total cells'] = dynamic_df[dynamic_df.columns[2:-6]].sum(axis=1)
       dynamic_all.append(dynamic_df)

   dynamic_all_df = pd.concat(dynamic_all, axis=0)
   sns.set(font_scale=1.5)
   sns.set_style("ticks")
   fig, axs = plt.subplots(6, 3, sharey=False, figsize=(10.5,13))
   fig.subplots_adjust(wspace=0.5, hspace=0.5)
   axs = axs.flatten()
   palette={'Cluster 1':'#3F9B0B', 'Cluster 2':'#FF796C', 'Cluster 3':'#0343DF','Cluster 4':'#000000', 'Cluster 5':'#D5B60A'}

   custom_lines = [[Line2D([0], [0], color='#FF796C', lw=3.5), Line2D([0], [0], color='#0343DF', lw=3.5),Line2D([0], [0], color='#000000', lw=3.5),Line2D([0], [0], color='#D5B60A', lw=3.5)],
                   [Line2D([0], [0], color='#3F9B0B', lw=3.5), Line2D([0], [0], color='#0343DF', lw=3.5),Line2D([0], [0], color='#000000', lw=3.5),Line2D([0], [0], color='#D5B60A', lw=3.5)],
                   [Line2D([0], [0], color='#3F9B0B', lw=3.5), Line2D([0], [0], color='#FF796C', lw=3.5),Line2D([0], [0], color='#000000', lw=3.5),Line2D([0], [0], color='#D5B60A', lw=3.5)],
                   [Line2D([0], [0], color='#3F9B0B', lw=3.5), Line2D([0], [0], color='#FF796C', lw=3.5),Line2D([0], [0], color='#0343DF', lw=3.5),Line2D([0], [0], color='#D5B60A', lw=3.5)],
                   [Line2D([0], [0], color='#3F9B0B', lw=3.5), Line2D([0], [0], color='#FF796C', lw=3.5),Line2D([0], [0], color='#0343DF', lw=3.5),Line2D([0], [0], color='#000000', lw=3.5)]]

   cluster_names = [['Cluster 2', 'Cluster 3','Cluster 4','Cluster 5'],
                    ['Cluster 1', 'Cluster 3','Cluster 4','Cluster 5'],
                    ['Cluster 1', 'Cluster 2','Cluster 4','Cluster 5'],
                    ['Cluster 1', 'Cluster 2','Cluster 3','Cluster 5'],
                    ['Cluster 1', 'Cluster 2','Cluster 3','Cluster 4']]

   for i, col in enumerate(dynamic_all_df.columns[2:]):
       sns.lineplot(data=dynamic_all_df, x='time', y=col, hue='Cluster', palette=palette, ax=axs[i], legend=False)
       axs[i].margins(x=0)
       axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
       axs[i].set_xlabel("time (days)",fontsize=14)
       axs[i].set_ylabel(str(col),fontsize=14)

   axs[9].legend(custom_lines[cluster], cluster_names[cluster], bbox_to_anchor=(3.98, 0.5), loc='center left',fontsize='x-small')

   if title==None: title = str(cluster+1)
   plt.show()

usr_inpt5 = input("Do you want to plot the dynamics based on each cluster's parameters?(yes=1, no=0)")

if int(usr_inpt5)==1:
   for c in range(5):
       plot_cross_cluster_IC(c, 3000)

###############################################################################
#########################Varying IC plots######################################
cols = ['Tn','Th','Tc','Tr','Dn','D','Mn','M','C','N','A','H','IL12','IL10','E','Ig','IL6','Clusters']

df = pd.read_csv('input/input/Large_Tumor_Patients.csv')
df = df[cols + ['percent_tumor']]
df.head()

cutoffs = []
for cluster in range(5):
    dat = df[df.Clusters == cluster+1].drop('Clusters', axis=1)
    cutoff = dat['C'].quantile(q=0.4, interpolation='nearest')
    cutoffs.append(cutoff)

print(cutoffs)

def plot_varying_IC(df, cluster, T, title=None, cutoff=cutoffs[cluster], cols=cols):
    import matplotlib.pyplot as plt
    import seaborn as sns

    dat = df[(df.Clusters == cluster+1) & (df.C <= cutoff)][cols].drop('Clusters', axis=1)
    dynamic_all = []

    for patient in dat.index:
        QSP_=QSP.from_cell_data(clustercells[cluster])

        dynamic_df = pd.DataFrame(columns=['time', 'patient']+QSP_.variable_names())
        t = np.linspace(0, T, 10*T+1)
        dynamic_df['time'] = t
        dynamic_df['patient'] = [patient]*len(dynamic_df['time'])

        IC = np.array(dat.loc[patient])/clustercells[cluster]
        u, _ = QSP_.solve_ode(t, IC, 'given')
        u = clustercells[cluster]*u
        dynamic_df[QSP_.variable_names()] = u
        dynamic_df['Total cells'] = dynamic_df[dynamic_df.columns[2:-6]].sum(axis=1)

        dynamic_all.append(dynamic_df)

    dynamic_all_df = pd.concat(dynamic_all, axis=0)

    sns.set(font_scale=1.5)
    sns.set_style("ticks")
    fig, axs = plt.subplots(6, 3, sharey=False, figsize=(10.5,13))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    axs = axs.flatten()

    for i, col in enumerate(dynamic_all_df.columns[2:]):
        sns.lineplot(data=dynamic_all_df, x='time', y=col, hue='patient', ax=axs[i], legend=False)
        axs[i].margins(x=0)
        axs[i].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        axs[i].set_xlabel("time (days)",fontsize=14)
        axs[i].set_ylabel(str(col),fontsize=14)

    if title==None: title = str(cluster+1)
    plt.savefig('fig/varying_IC_'+str(cluster+1)+'.eps', format='eps',dpi=300)
    plt.show()



for c in range(5):
    print('starting cluster'+str(c+1))
    plot_varying_IC(df, c, 3000)
