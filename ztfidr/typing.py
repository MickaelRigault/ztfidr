""" Interact with the typingapp """

from . import io
import os
import pandas
import numpy as np


DB_PATH = os.path.join(io.IDR_PATH, "typingapp.db")

# typing app data
import sqlite3
con = sqlite3.connect(DB_PATH)


class _DBApp_():
    _DB_NAME = None

    def __init__(self):
        """ """
        self.load()
    
    @classmethod
    def data_from_db(cls, sql=None):
        """ """
        basequery = f"SELECT * FROM {cls._DB_NAME}"
        if sql is not None:
            basequery +=f" {sql}"
            
        return pandas.read_sql_query(basequery, con)
        
        
    def load(self):
        """ """
        self._data = self.data_from_db()
    
    # =============== #
    #   Properties    #
    # =============== #
    @property
    def data(self):
        """ """
        return self._data

# =================== #
#                     #
#  USERS              #
#                     #
# =================== #    
class _Users_( _DBApp_ ):
    _DB_NAME = "Users"
    
    def load(self):
        """ """
        self._data = self.data_from_db()[["id","username","name","email","date_added"]]

# =================== #
#                     #
#  CLASSIFICATIONS    #
#                     #
# =================== #
class Reports( _DBApp_ ):
    _DB_NAME = "Classifications"
    
    def load(self):
        """ """
        self._data = self.data_from_db(sql="where kind='report'")

# =================== #
#                     #
#  CLASSIFICATIONS    #
#                     #
# =================== #
class Classifications( _DBApp_ ):
    _DB_NAME = "Classifications"
    _NONIA_CASES = ["not ia", "ii","ib/c","gal","other"]
    
    def load(self):
        """ """
        self._data = self.data_from_db(sql="where kind='typing'")
        self._data["value"] = self._data["value"].str.strip().str.lower()

    def load_sample(self):
        """ """
        from . import get_sample
        self._sample = get_sample()
        
    # -------- #
    # GETTER   #
    # -------- #
    def get_data(self, incl_unclear=True, min_classification=None,
                     goodcoverage=None, redshift_range=None,
                     x1_range=None, c_range=None, **kwargs):
        """ 

        min_classification: [None or int]
            minimum number of typing for this target. 
            unclear are not counted.

        """
        data = self.data.copy()
        if not incl_unclear:
            data = data[data["value"] != "unclear"]
        if min_classification is not None and min_classification>0:
            ntyped = self.get_ntyped(incl_unclear=False)
            data = data[data["target_name"].isin(ntyped[ntyped>=min_classification].index)]
            
        # Sample
        _locals = locals()
        prop_sample = kwargs
        for k in ["goodcoverage", "redshift_range", "x1_range", "c_range"]:
            if _locals[k] is not None:
                prop_sample[k] = _locals[k]
                
        if len(prop_sample) >0:
            names_to_keep = self.sample.get_data(**prop_sample).index
            data = data[data["target_name"].isin(names_to_keep)]
            
        return data
        
        
            

    def get_ntyped(self, incl_unclear=False):
        """ series of the n"""
        return self.get_data(incl_unclear=incl_unclear).groupby("target_name").size().sort_values()
    
    def get_target_data(self, name):
        """ """
        return self.data.query(f"target_name == '{name}'")
    
    def get_of_types(self, types):
        """ get entry associated to the given type (lower case) """
        return self.data[self.data["value"].isin(np.atleast_1d(types))]
                         
    def get_nonia(self, incl_unclear=False):
        """ """
        list_to_get = self._NONIA_CASES.copy()
        if incl_unclear:
            list_to_get+="unclear"
            
        return self.data[self.data["value"].isin(list_to_get)]
        
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_classifications(self, ax=None, fig=None):
        """ """
        ndatas = self.data.groupby(["value"]).size().sort_values()
        
        
        if ax is None:
            if fig is None:
                import matplotlib.pyplot as mpl
                fig = mpl.figure(figsize=[7,4])
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        hbars = ax.barh(ndatas.index, ndatas, tick_label=ndatas.index)
        ax.tick_params("y", labelsize="small", color="C0", length=0)

        ax.bar_label(hbars, fmt='%d', color="C0", padding=1, 
                     fontsize="x-small")

        clearwhich = ["bottom","right","top", "left"] # "bottom"
        [ax.spines[which].set_visible(False) for which in clearwhich]
        ax.set_xticks([])
        return fig
        
    # ============== #
    #  Properties    #
    # ============== #
    @property
    def sample(self):
        """ """
        if not hasattr(self, "_sample"):
            self.load_sample()
            
        return self._sample
