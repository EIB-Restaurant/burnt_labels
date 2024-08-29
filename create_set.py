from datasets import Dataset
from datasets import DatasetDict
import pandas as pd
import numpy as np


class prob_object():
    def __init__(self, probs, labels, unq_arr, namedic, depth = [0]):
        self.ndic = namedic
        self.olabs = list(namedic.keys())
        self.depth = depth
        probs = probs / probs.sum(axis=1, keepdims=True)
        print(depth, probs.shape, labels.shape, unq_arr.shape)
        self.probs = probs
        self.labels = labels
        self.unq_arr = unq_arr
        self.lev_labels = self.labels[:, 0]
        self.lev_unq = np.arange(0,self.lev_labels.max()+1)
        # self.lev_unq = np.unique(self.lev_labels)

        self.lev_list = self._create_lev()
        self.lev_probs = self.ret_sum_prob()
        self.lev_preds = self.lev_probs.argmax(axis=1)
        self.lev_nums = [np.sum(self.lev_labels == u) for u in self.lev_unq]
        self.labs = [f'{n}-{lb}' for n, lb in zip(self.lev_nums, self.olabs)]
        self.sel_lev = [np.where((self.lev_labels == u) & (self.lev_preds == u))[0] for u in self.lev_unq]

        print(depth, len(self.labels), len(self.lev_list), [len(lev) for lev in self.lev_list], [len(lev) for lev in self.sel_lev], self.unq_arr.shape)
        self.children = []
        if len(self.lev_list) > 1 and self.unq_arr.shape[1] > 1:
            for ilev in range(len(self.lev_list)):
                if len(self.sel_lev[ilev]) > 0 and len(self.lev_list[ilev]) > 1:
                    self.children.append(prob_object(self.probs[self.sel_lev[ilev]][:,self.lev_list[ilev]], self.labels[self.sel_lev[ilev], 1:],
                                                  self.unq_arr[self.lev_list[ilev],1:], self.ndic[self.olabs[ilev]], depth=[*depth, ilev]))


    def _create_lev(self):
        msks = [(self.unq_arr[:,0] == u) for u in self.lev_unq]
        return [np.where(m)[0] for m in msks]
    
    def ret_sum_prob(self):
        return np.array([self.probs[:,lev].sum(axis=1) for lev in self.lev_list]).T
    
    def get_children(self):
        if len(self.children)>0:
            return self.children
        else:
            return None
        
    def plot_probs(self, figax = None, conf = '', color = 'cividis', norm = 'true', prec = '.2f'):
        lss = ['-', '--', '-.', ':']
        def text_up(x):
            x.set_text(x.get_text().split('-')[1])
            return x

        if figax is None:
            fig, ax = plt.subplots(figsize=(5,5))
        else:
            fig, ax = figax
        key = '-'.join(map(str,self.depth))
        if conf == 'mat':
            cm = confusion_matrix(self.lev_labels, self.lev_preds, labels=range(len(self.labs)), normalize=norm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labs)
            disp.plot(ax=ax, cmap=color, xticks_rotation=85, values_format=prec, colorbar=False, text_kw = {'fontsize': 10}) 
            # ax.set_xticklabels(ax.get_xticklabels(), fontsize=10)
            ax.set_xticklabels(map(text_up, ax.get_xticklabels()), fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        elif conf == 'prob':
            lab_arr = np.full_like(self.lev_probs, 0, dtype = int)
            lab_arr[np.arange(lab_arr.shape[0]),self.lev_labels] = 1
            for i in range(lab_arr.shape[1]):
                fpr, tpr, _ = roc_curve(lab_arr[:,i], self.lev_probs[:,i])
                ax.plot(fpr, tpr, label = self.labs[i], lw = 1.5+i/4, ls = lss[i%4], alpha = .05+.95**(i+1))
                # ax.plot(fpr, tpr, label = self.labs[i].split('-')[1], lw = 2)
            ax.plot([0,1], [0,1], 'k--', lw = 2)
            ax.legend(fontsize = 15)
            ax.set_xlim([-.05,1.05])
            ax.set_ylim([-.05,1.05])
            ax.set_xlabel('False Positive Rate', fontsize = 15)
            ax.set_ylabel('True Positive Rate', fontsize = 15)
            ax.tick_params(axis = 'both', direction = 'in', length = 7, which = 'major', width = 1.5)

        else:
            ax.plot(self.lev_preds, self.lev_labels, 'o', fillstyle='none', alpha = .1)
        ax.set_title(key)
        return {key: [fig, ax]}
    
    # def plot_roc(self, figax = None, conf = False, color = 'cividis', norm = 'true', prec = '.2f'):

    
    def plot_children(self, stop = 1, conf = '', color = 'cividis', norm = 'true', pself = False, prec = '.2f'):
        if pself:
            pl_dict = self.plot_probs(conf = conf, color = color, norm = norm, prec=prec)
        else:
            pl_dict = {}
        lch = len(self.children)
        if lch>0 and stop > 0:
            fig, axs = plt.subplots(1, lch, figsize=(7*lch, 5))
            for ich, child in enumerate(self.children):
                if lch > 1:
                    pl_dict.update(child.plot_probs([fig, axs[ich]], conf = conf, color = color, norm = norm, prec = prec))
                else:
                    pl_dict.update(child.plot_probs([fig, axs], conf = conf, color = color, norm = norm, prec = prec))

                pl_dict.update(child.plot_children(stop = stop - 1, conf = conf, color = color, norm = norm, prec = prec))
        return pl_dict
        


class create_dataset():
    def __init__(self, df, rs = 1):
        self.df = df
        self.rng = np.random.default_rng(seed=rs)

    def _get_df_spls(self, frcs):
        if np.sum(frcs) > 1:
            frcs = np.array(frcs) / np.sum(frcs)
        
        lendf = len(self.df)
        tot_frc = np.sum(frcs)
        n = int(lendf*tot_frc)
        n_spls = [int(lendf*frc) for frc in frcs]
        n_spls[-1] = n - np.sum(n_spls[:-1])
        i_spls = np.cumsum(n_spls)
        
        inds = self.rng.choice(np.arange(lendf), n, replace=False)
        sel_inds = np.split(inds, i_spls[:-1])
        # print(sel_inds)
        sel_msk = []
        for ind in sel_inds:
            msk = np.full(lendf, False)
            msk[ind] = True
            sel_msk.append(msk)
            
        return [self.df[ind] for ind in sel_msk]
    
    def get_train_val_test(self, frcs = [0.8, 0.1, 0.1]):
        if len(frcs) != 3:
            print(f'only using the first 3 fractions: {frcs[:3]}')

        df_dics = [f.to_dict('list') for f in self._get_df_spls(frcs[:3])]
        dtset = [Dataset.from_dict(dfd) for dfd in df_dics]
        return DatasetDict({k:v for k, v in zip(['train', 'test', 'validation'], dtset)})



class create_number_labs():
    def __init__(self, labs, col, base = None, base_lab = True) -> None:
        self.df = labs
        self._name_dict = create_number_labs.get_num_dic(self.df[col])
        self._num_dict = create_number_labs.get_depth(self._name_dict)
        self._depth = create_number_labs.get_max_depth(self._name_dict)
        self.label_array = np.array([self._gen_nums(x[col]) for _,x in self.df.iterrows()])
        self._max_level = np.max(self.label_array)
        if base is None:
            base = self._max_level

        for i in range(self._depth):
            self.df[f'lab_{i}'] = self.label_array[:,i]

        self.df.sort_values([f'lab_{i}' for i in range(self._depth)], inplace=True)


        if base_lab:
            self.base_rep = list(map(lambda x:create_number_labs.baseToNumber(x, base), self.label_array))
            self.df['base_label'] = self.base_rep
        else:
            self.u, indices = np.unique(self.label_array, return_inverse=True, axis = 0)
            self.labs_nums = np.arange(len(self.u))
            self.df['base_label'] = self.labs_nums[indices]





    @staticmethod
    def baseToNumber(lst, b):
        lst = lst[::-1]
        n = 0
        for i, d in enumerate(lst):
            n += d*b**i
        return n


    def _gen_nums(self, lb):
        spl = lb.split('/')
        arr = np.full(self._depth, 0)
        dic = self._name_dict
        # arr[0] = create_number_labs._get_index(self._name_dict, spl[0])
        for i in range(len(spl)):
            arr[i] = create_number_labs._get_index(dic, spl[i])
            dic = dic[spl[i]]
        return arr

    @staticmethod
    def _lab_spl(lab, i):
        spls = lab.split('/')
        if len(spls) > i:
            return spls[i]
        else:
            return '-'

    @staticmethod    
    def _get_index(dic, lab):
        # print(list(dic.keys()), lab)
        return list(dic.keys()).index(lab)       
    
    @staticmethod
    def get_max_depth(dic):
        if len(dic) == 0:
            return 0
        else:
            return 1 + max([create_number_labs.get_max_depth(dic[v]) for v in dic.keys()])
        
    @staticmethod
    def get_depth(dic, level = 0):
        if len(dic) == 0:
            return {}
        else:
            return {i:create_number_labs.get_depth(dic[v], level+1) for i, v in enumerate(dic.keys())}

    @staticmethod
    def get_num_dic(labs):
        lab_ids = {}
        spl_lbs = [l.split('/') for l in labs]
        l0s = [l[0] for l in spl_lbs]
        lrs = ['/'.join(l[1:]) if len(l)>1 else '-' for l in spl_lbs]
        tempdf = pd.DataFrame({'lrs':lrs})
        lsels = sorted(list(set(l0s)))
        if '-' in lsels:
            lab_ids['-'] = 0
            lsels.remove('-')

        for i, lab in enumerate(lsels):
            lab_ids[lab] = i

        id_labs = np.array(list(map(lambda x:lab_ids[x], l0s)))

        for i, lab in enumerate(lsels):
            lab_ids[lab] = create_number_labs.get_num_dic(tempdf.loc[id_labs == i, 'lrs'])

        if '-' in list(lab_ids.keys()):
            return {}
        else:
            return lab_ids


    
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def baseToNumber_norev(lst, b):
    n = 0
    for i, d in enumerate(lst):
        n += d*b**(len(lst)-i-1)
    return n


