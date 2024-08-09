import numpy as np
from bs4 import BeautifulSoup
import urllib
import time
import datetime
import os
# import pickle


class url_soup():
    def __init__(self, url) -> None:
        self.header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' 
                'AppleWebKit/537.11 (KHTML, like Gecko) '
                'Chrome/23.0.1271.64 Safari/537.11',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                'Accept-Encoding': 'none',
                'Accept-Language': 'en-US,en;q=0.8',
                'Connection': 'keep-alive'}
        
        self.url = url 
        self.req = urllib.request.Request(url=self.url, headers=self.header) 
        self.page = urllib.request.urlopen(self.req).read()
        self.soup = BeautifulSoup(self.page, 'html.parser')
        self.all_a = self.get_all_a()


    def get_record(self):
        records = []
        for v in self.soup.find_all('script'):
            try:
                v_dic = eval(v.text)
                if '@context' in v_dic:
                    if v_dic['@context'] == 'http://schema.org/':
                        records.append(v_dic)
            except:
                continue
        return records
    

    def get_all_a(self):
        return [f for f in self.soup.find_all('a') if 'href' in f.attrs]
 
    def get_a_hrefs(self, tp):
        records = []
        for a in self.all_a:
            if tp in a['href']:
                records.append(a['href'])
        return records
    
    def get_prod_dic(self, tp):
        records = {}
        for a in self.all_a:
            if tp in a['href']:
                records[a.text] = a['href']
        return records
       
    def get_cont_hrefs(self):
        return self.get_a_hrefs(self.url)
    
    def get_all_pages(self, now, n = 2):
        pg_cntr = 2
        prod_dic = self.get_prod_dic('/products/')
        pg_cnts = np.unique([int(f.split('?page=')[1]) for f in self.get_a_hrefs('?page=')])
        # print(f'starting pagenums: {pg_cnts}')
        while pg_cntr-1 <= n and pg_cntr in pg_cnts:
            print(f'{now()} crawling page {pg_cntr}')
            url_cntr = self.url + '?page=' + str(pg_cntr)
            soup_cntr = url_soup(url_cntr)
            prod_dic.update(soup_cntr.get_prod_dic('/products/'))
            pg_cnts = np.unique([int(f.split('?page=')[1]) for f in soup_cntr.get_a_hrefs('?page=')])
            pg_cntr += 1
            time.sleep(1)

        return prod_dic
    

class crawl_type():
    def __init__(self, url, pkl_fl, n = 2, nchild = 2, st_inp = None) -> None:
        self.now = datetime.datetime.now
        self.pkl = pkl_fl
        # print(f'{self.now()} creating a new crawl_type object with {url}, {n}, {nchild}')
        self.main = url_soup(url)
        self.url = url
        self.n = n
        self.nchild = nchild
        self.prod_dic = {}
        self.all_conts = self.main.get_cont_hrefs()
        self.all_conts.remove(url)
        # print(f'{self.now()} all_conts: {self.all_conts}')

        if os.path.exists(self.pkl):
            load_file = np.load(self.pkl, allow_pickle = True)
            self.prod_tree = load_file[1]
            st_i = load_file[0]
            if self.url != load_file[2]:
                print(f'{self.now()} the url in the pkl file does not match the current url')
                self.prod_tree = {}
                self.pkl = f'{self.pkl.split(".")[0]}-{self.now().strftime("%Y-%m-%d-%H-%M-%S")}.npy'
                st_i = 0
        else:
            self.prod_tree = {}
            st_i = 0

        url_res = self.main.get_cont_hrefs()
        while url in url_res:
            url_res.remove(url)

        en_i = len(url_res)

        print(f'{self.now()} there are {en_i} child pages in {url}')

        if not (st_inp is None):
            st_i = st_inp

        for iurl in range(st_i, en_i):
            url_run = url_res[iurl]
            self.prod_tree[url_run.split(url)[1]] = crawl_type._get_tree(url_run, self.now, f'b-{iurl}')
            np.save(self.pkl, np.array([iurl, self.prod_tree, self.url], dtype = object))

            time.sleep(1)

        # self._get_products()

    # @staticmethod
    # def _get_tree(url, now, depth = 2):
    #     url_out = {}
    #     url_res = url_soup(url).get_cont_hrefs()
    #     url_res.remove(url)
    #     if len(url_res) == 0:
    #         print(f'{now()} there are no child pages in {url}')
    #         return 'none'
    #     else:
    #         print(f'{now()} there are {len(url_res)} child pages in {url}')
    #         for pg_url in url_res[:depth]:
    #             print(f'{now()} crawling {pg_url}')
    #             url_out[pg_url] = crawl_type._get_tree(pg_url, now, depth)
    #             time.sleep(.5)
    #         return url_out
        
    @staticmethod
    def _get_tree(url, now, str_trc):
        url_out = {}
        # print(f'{now()} getting tree for {url}')
        url_res = url_soup(url).get_cont_hrefs()
        while url in url_res:
            url_res.remove(url)
            
        if len(url_res) == 0:
            # print(f'{now()} there are no child pages in {str_trc}')
            return 'none'
        else:
            print(f'{now()} there are {len(url_res)} child pages in {str_trc}')
            pg_cnts = list(np.arange(len(url_res)))
            # print(f'{now()} starting pagenums: {pg_cnts}')
            ipg = 0
            while ipg in pg_cnts:
                pg_url = url_res[ipg]
                # print(f'{now()} crawling {ipg}: {pg_url}')
                
                try:
                    url_out[pg_url.split(url)[1]] = crawl_type._get_tree(pg_url, now, f'{str_trc}-{ipg}')
                    pg_cnts.remove(ipg)
                    ipg += 1
                    err_cnt = 0

                except:
                    print(f'{now()} error in crawling {pg_url}-{err_cnt}')
                    if err_cnt > 3:
                        pg_cnts.remove(ipg)
                        ipg += 1
                        err_cnt = 0

                    err_cnt += 1

                time.sleep(1)
            return url_out

    def _get_products(self):
        if len(self.all_conts) == 0:
            print(f'{self.now()} there is only one page in {self.url}')
            self.prod_dic[self.url] = self.main.get_all_pages(self.now, self.n)
        elif len(self.all_conts) > 0:
            print(f'{self.now()} there are {len(self.all_conts)} pages in {self.url}')
            for f in self.all_conts[:self.nchild]:
                print(f'{self.now()} crawling {f}')
                self.prod_dic.update(self._crawl_child(f))
                time.sleep(.5)

    def get_prod_dic(self):
        return self.prod_dic

    def _crawl_child(self, f):
        child = crawl_type(f, self.n, self.nchild)
        return child.get_prod_dic()









    

    


        