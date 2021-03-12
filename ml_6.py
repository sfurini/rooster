import sys
import copy
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st
import scipy
import pickle

from mord import LogisticAT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

#from sklearn.model_selection import StratifiedKFold, KFold
#from sklearn.metrics import auc
from kneed import KneeLocator

filename_pheno = '../phenotype.csv'
filename_include = '../eurIDsPCA.txt'
filename_exclude = '../filtered.txt'
filename_al1 = '../data_al1_rare_1982.csv'
filename_al2 = '../data_al2_rare_1982.csv'
filename_gc1 = '../data_gc_unique_hetero_1982.csv'
filename_gc2 = '../data_gc_unique_homo_1982.csv'

train_size = 0.80
sex = int(sys.argv[1])
rep = int(sys.argv[2])

pdf = PdfPages('figures_{}_{}.pdf'.format(sex, rep))

def plot_dist_rare(df, filename_pdf):
    from kneed import KneeLocator
    numb = df.sum(axis = 0)
    y, x = np.histogram(numb, bins=100)
    cdf = np.cumsum(y)
    pl.figure()
    pl.plot(x[:-1], cdf/cdf[-1], 'o-')
    x_ = x[:-1]
    y_ = cdf/cdf[-1]
    kneedle = KneeLocator(x_.tolist(), y_.tolist(), S=1, curve="concave", direction="increasing")
    pl.plot([round(kneedle.knee, 3)]*2, [0, 1.1], 'k--')
    pl.xlabel('number of patients with mutated gene')
    pl.ylim(0, 1)
    pl.plot(round(kneedle.knee, 3), round(kneedle.knee_y, 3), 'ro')
    pl.savefig(filename_pdf)
    inds = numb < kneedle.knee
    print(numb)
    print(inds)
    print(kneedle.knee)
    print(np.sum(inds))
    return inds

def plot_olr(age, grading, adjusted_grading, pdf_name):
    df_task = pd.DataFrame({'age':age,'grading':grading, 'ordered_LR_prediction':adjusted_grading})
    pl.figure()
    df_plot_m = df_task
    pl.subplot(1,1,1)
    df_plot_m['Patient distribution'] = 'actual distribution'
    df_task_original_copy = copy.deepcopy(df_plot_m)
    df_task_original_copy['grading'] = df_plot_m['ordered_LR_prediction']
    df_task_original_copy['Patient distribution'] = 'predicted distribution'
    df_task_ = pd.concat([df_plot_m[['age', 'grading', 'Patient distribution']], df_task_original_copy[['age', 'grading', 'Patient distribution']]], 0)
    sns.violinplot(x='age', y='grading', data=df_task_, orient='h', inner=None, hue='Patient distribution', split=True,
                   order = [5, 4, 3, 2, 1, 0], color='gray', scale='count')
    pl.plot(df_plot_m['age'], 5-df_plot_m['grading'], 'or')
    pl.plot(df_plot_m['age'], 5-df_plot_m['ordered_LR_prediction'], 'ok', label='predicted grading (OLR)')
    ok = df_plot_m[(df_task['grading'] - df_task['ordered_LR_prediction'])<0]
    pl.plot(ok['age'], 5-ok['grading'], 'og')
    pl.xlabel('')
    pl.ylabel('grading')
    pl.legend()
    pl.savefig(pdf_name)
    pl.show()
    pl.close()

class MoreTrain(object):
    pass

class PostMendelianModel(LogisticRegression):
    def __init__(self, rares = None, inds = None, rare_w = None, lambda_0 = None, lambda_1 = None, lambda_2 = None, lambda_3 = None):
        self.rares = rares
        self.inds = inds
        self.rare_w = rare_w
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
    def get_params(self, deep = True):
        params = {'rares':self.rares}
        params['inds'] = self.inds
        params['rare_w'] = self.rare_w
        for i in range(len(self.inds)):
            params['lambda_{}'.format(i)] = getattr(self, 'lambda_{}'.format(i))
        return params
    def set_params(self, **kwargs):
        self.rares = kwargs.get('rares',self.rares)
        self.inds = kwargs.get('inds',self.inds)
        self.rare_w = kwargs.get('rare_w',self.rare_w)
        self.lambda_0 = kwargs['lambda_0']
        self.lambda_1 = kwargs['lambda_1']
        self.lambda_2 = kwargs['lambda_2']
        self.lambda_3 = kwargs['lambda_3']
        return self
    def fit(self, X, y):
        self.coef_ = np.empty(X.shape[1])
        for i, ind in enumerate(self.inds):
            X_ss = X[:,ind[0]:ind[1]]
            model = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=200, C=getattr(self, 'lambda_{}'.format(i)), class_weight='balanced')
            model.fit(X_ss, y)
            if self.rares[i]:
                self.coef_[ind[0]:ind[1]] = self.rare_w * model.coef_
            else:
                self.coef_[ind[0]:ind[1]] =  model.coef_
    def predict(self, X):
        s = np.dot(X,self.coef_.reshape(-1,1))
        return s
    def score(self, X, y):
        return roc_auc_score(y.astype(float), self.predict(X))

class PostMendelianModel1(PostMendelianModel):
    def set_params(self, **kwargs):
        self.rares = kwargs.get('rares',self.rares)
        self.inds = kwargs.get('inds',self.inds)
        self.rare_w = kwargs.get('rare_w',self.rare_w)
        return self
    
class AdjustGrading(object):
    """
    orl con bootstrap
    """
    def __init__(self):
        pass
    def fit(self):
        """
        trova le soglie usando bootsrtap
        """
        pass
    def predict(self):
        pass

#class Splitter(object):
#    def __init__(self):
#        pass
#    def split(self, X, y, sex):
#        pass
#    def get_n_splits(self, X, y, sex):
#        pass

pheno = pd.read_csv(filename_pheno, index_col = 0)
pheno = pheno[np.logical_or.reduce(( pheno['grading'] == 0, pheno['grading'] == 1, pheno['grading'] == 2, pheno['grading'] == 3, pheno['grading'] == 4, pheno['grading'] == 5 ))]
pheno = pheno[np.logical_or.reduce(( pheno['gender'] == 0, pheno['gender'] == 1 ))]
pheno = pheno[np.logical_not(pheno['age'].isnull())]
if filename_exclude is not None:
    with open(filename_exclude,'rt') as fin:
        samples_2_exclude = [l.strip()[:-5] for l in fin.readlines() if l[0] != '#']
    pheno = pheno[[sample not in samples_2_exclude for sample in pheno.index]]
if filename_include is not None:
    with open(filename_include,'rt') as fin:
        samples_2_include = [l.strip()[:-5] for l in fin.readlines() if l[0] != '#']
    pheno = pheno[[sample in samples_2_include for sample in pheno.index]]

samples = list(set(pheno.index[pheno['gender'].astype(int) == sex]) & set(pheno.index) & set([sample for sample in pd.read_csv(filename_al1, index_col = 0, usecols = [0,]).index  for filename in [filename_al1, filename_al2, filename_gc1, filename_gc2]]))
samples.sort()
n_samples = len(samples)
print('Number of samples: {}'.format(n_samples), flush = True)

df_al1 = pd.read_csv(filename_al1, index_col=0)#, usecols = range(10000))
df_al2 = pd.read_csv(filename_al2, index_col=0)#, usecols = range(10000))
df_gc1 = pd.read_csv(filename_gc1, index_col=0)#, usecols = range(10000))
df_gc2 = pd.read_csv(filename_gc2, index_col=0)#, usecols = range(10000))

pheno = pheno.loc[samples]
al1 = df_al1.loc[samples].values
al2 = df_al2.loc[samples].values
gc1 = df_gc1.loc[samples].values
gc2 = df_gc2.loc[samples].values
n_features_rep = [al1.shape[1], al2.shape[1], gc1.shape[1], gc2.shape[1]]
ends = list(np.cumsum(n_features_rep))
print('Number of features before filter: {}'.format(ends[-1]), flush = True)

age = pheno['age'].values
gender = pheno['gender'].values
grading = pheno['grading'].values.astype(int)
age_cat = pd.qcut(age, 3, labels=False)
grading_age_cat = ['{}_{}'.format(x,y) for x,y in zip(age_cat, grading)]

ind_train, ind_test = train_test_split(np.arange(n_samples).reshape((-1, 1)), train_size=train_size, stratify=grading_age_cat )
ind_train = ind_train.flatten()
ind_test = ind_test.flatten()

inds_al1 = plot_dist_rare(al1[ind_train,:], 'dist_{}_{}_{}.pdf'.format('al1', sex, rep))
inds_al2 = plot_dist_rare(al2[ind_train,:], 'dist_{}_{}_{}.pdf'.format('al2', sex, rep))
al1 = al1[:,inds_al1]
al2 = al2[:,inds_al2]
al1_columns = df_al1.columns[inds_al1]
al2_columns = df_al2.columns[inds_al2]
gc1_columns = df_gc1.columns
gc2_columns = df_gc2.columns

orl = LogisticAT(alpha=0)
orl.fit(age[ind_train].reshape((-1,1)), grading[ind_train])

X_train = np.concatenate( (al1[ind_train, :], al2[ind_train, :], gc1[ind_train, :], gc2[ind_train, :]), axis = 1 )
X_test = np.concatenate( (al1[ind_test, :], al2[ind_test, :], gc1[ind_test, :], gc2[ind_test, :]), axis = 1 )
y_train = ((grading[ind_train] - orl.predict(age[ind_train].reshape((-1, 1)))) > 0).astype(int)
y_test = ((grading[ind_test] - orl.predict(age[ind_test].reshape((-1, 1)))) > 0).astype(int)

plot_olr(age[ind_train], grading[ind_train], orl.predict(age[ind_train].reshape((-1, 1))), 'figure_{}_{}_olr_train.pdf'.format(sex,rep))
plot_olr(age[ind_test], grading[ind_test], orl.predict(age[ind_test].reshape((-1, 1))), 'figure_{}_{}_olr_test.pdf'.format(sex, rep))

n_features_rep = [al1.shape[1], al2.shape[1], gc1.shape[1], gc2.shape[1]]
ends = list(np.cumsum(n_features_rep))
starts = [0,] + ends[:-1]
inds = [[s,e] for s,e in zip(starts,ends)]
rares = [True, True, False, False ]
print('Number of features: {}'.format(ends[-1]), flush = True)

lambdas = {}
for i, ind in enumerate(inds):
    logreg = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=200, class_weight='balanced')
    gs = RandomizedSearchCV(logreg, {'C':st.loguniform(a = 0.01, b = 1)}, n_iter = 100, cv=10, scoring = 'roc_auc')
    gs.fit(X_train[:,ind[0]:ind[1]], y_train)
    C = gs.cv_results_['param_C'].data
    mean_score = gs.cv_results_['mean_test_score']
    std_score = gs.cv_results_['std_test_score']
    lambdas[i] = gs.best_params_['C']
    f = pl.figure()
    ax = f.add_subplot(111)
    ax.errorbar(C, mean_score, yerr = std_score)
    plt.xscale('log')
    pdf.savefig()
    plt.close()
pmm = PostMendelianModel1(rares = rares, inds = inds, lambda_0 = lambdas[0], lambda_1 = lambdas[1], lambda_2 = lambdas[2], lambda_3 = lambdas[3])
pmm_params = {'rare_w': scipy.stats.uniform(loc=0.1, scale=9.9)}
gs = RandomizedSearchCV(pmm, pmm_params, n_iter = 100, cv=10)
gs.fit(X_train, y_train)
print(gs.cv_results_)

#pmm_params = {'rare_w': scipy.stats.uniform(loc=0.1, scale=9.9)}
#for i_model, rare in enumerate(rares):
#    pmm_params['lambda_{}'.format(i_model)] = st.loguniform(a = 0.01, b = 1)
#pmm = PostMendelianModel(rares = rares, inds = inds)
#gs = RandomizedSearchCV(pmm, pmm_params, n_iter = 100, cv=10)
#gs.fit(X_train, y_train)
#print(gs.cv_results_)
 
rare_w_best = gs.best_params_['rare_w']
#lambda_0_best = gs.best_params_['lambda_0']
#lambda_1_best = gs.best_params_['lambda_1']
#lambda_2_best = gs.best_params_['lambda_2']
#lambda_3_best = gs.best_params_['lambda_3']
lambda_0_best = lambdas[0]
lambda_1_best = lambdas[1]
lambda_2_best = lambdas[2]
lambda_3_best = lambdas[3]
pmm = PostMendelianModel(rares=rares, inds=inds, rare_w=rare_w_best,
                        lambda_0=lambda_0_best, lambda_1=lambda_1_best, lambda_2=lambda_2_best, lambda_3=lambda_3_best)
pmm.fit(X_train, y_train)
print('Score train:', pmm.score(X_train, y_train))
print('Score test:', pmm.score(X_test, y_test))

for (e, cols), name_rep in zip(enumerate((al1_columns, al2_columns, gc1_columns, gc2_columns)), ('al1', 'al2', 'gc1', 'gc2')):
    coef_all = pmm.coef_[inds[e][0]:inds[e][1]]
    names = cols[coef_all != 0]
    coefs = coef_all[coef_all != 0]
    abs_coefs = np.abs(coefs)
    df = pd.DataFrame({'gene': names, 'importance': coefs, 'abs_i': abs_coefs})
    if df.shape[0] != 0:
        print(name_rep, df.shape[0])
        ax = df.sort_values(['abs_i'], ascending=False).iloc[:30].plot.bar(x='gene', y='importance', rot=270, title=name_rep)
        pl.savefig('{}_{}_{}.pdf'.format(name_rep, sex, rep))
        pl.show()

fout = open("res_{}_{}.obj".format(sex, rep),"wb")
pickle.dump(gs, fout)
pickle.dump(pmm, fout)

pdf.close()

# params = gs.cv_results_['param_lambda_0'].data
# scores = gs.cv_results_['mean_test_score']
# stds = gs.cv_results_['std_test_score']
# pl.figure('grid search average')
# pl.errorbar(1./params, scores, stds/2., marker='o', linestyle='-')
# pl.xscale('log')
# pl.ylabel('cross-validation score')
# pl.xlabel('LASSO parameter')
# pl.show()

#pmm.fit(X_train,y_train)
#print(pmm.score(X_train,y_train))
#print(pmm.score(X_test,y_test))

