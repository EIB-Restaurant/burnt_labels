from datasets import Dataset
from datasets import DatasetDict
import pandas as pd
import numpy as np




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


