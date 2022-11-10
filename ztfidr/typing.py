""" Interact with the typingapp """

from . import io
import os
import pandas
import numpy as np


DB_PATH = os.path.join(io.IDR_PATH, "typingapp.db")

# typing app data
import sqlite3
con = sqlite3.connect(DB_PATH)


SORT_FAVORED = ["not ia","ib/c","ii","gal", "other",
                "sn ia","ia-norm",
                "ia-91bg", "ia-91t", "ia-other"]

def get_typing(typings_values, 
               sort_favored=SORT_FAVORED):
    """ """
    typings, values = typings_values
    typings[values==values.max()]
    if len(typings)  == 1:
        return typings[0]
    
    ltypings = list(typings)
    ltypings.sort(key = lambda i: sort_favored.index(i))
    return ltypings[0]



def parse_classification(line, 
                         min_review=2,
                         min_autotyping=3, 
                         min_generic_typing=3,
                         unclear_limit=0.5,
                         to_unclear_limit=0.75,
                         min_classifications_to_unclear=6,
                         verbose=False):
    """ """
    classification = "confusing"
    if verbose:
        print(line)
        print(line["typing"])
        print("unclear" in line["typing"])
    # not enough review, pass
    if line.nreviews <= min_review: # not enough
        if verbose:
            print("not enough classification")
        classification = "None"
        
    # easy, all the same 
    elif line.ntypes == 1 and line.ntypings[0] >= min_autotyping:
        if verbose:
            print("all the same")
        classification =  line.typing[0]

    # special save for ia-norm cases.
        
    # enough for generic, let's see if it's sufficient.        
    elif line.nreviews >= min_generic_typing:
        if verbose:
            print("enough for generic review")
        # go back to the easy-case
        if "ia-norm" in line.typing and np.all([k in ["ia-norm", "sn ia", "unclear"] for k in line.typing]) \
          and line.ntypings[ list(line.typing).index("ia-norm") ] >= min_autotyping:
            classification = "ia-norm"
            
        # It looks like a Ia ir Ia-norm
        elif np.all([k in ["ia-norm", "sn ia"] for k in line.typing]):
            classification = "ia_or_ianorm"
            
        # Typing ok, Sub-tyiping not            
        elif np.all(['ia' in k and "not ia" not in k for k in line.typing]):
            classification = "ia"
            
        # all non-ia ?
        elif np.all([k in Classifications._NONIA_CASES for k in line.typing]):
            classification = "nonia"
        else:
            classification = "confusing"
            
    # saving unclears if possible
    # try to save remaining confusing cases
    if classification == "confusing" and "unclear" in line["typing"]:
        
        if verbose:
            print("trying to save confusing cases")
            print("unclear" in line["typing"])
            
        info = dict(zip(line["typing"],line["ntypings"]))
        n_unclears = info["unclear"] 
        n_not = np.sum([v for k,v in info.items() if k !="unclear"])
        n_classifications = n_unclears + n_not
        f_unclear = n_unclears / (n_classifications)
        f_not = 1-f_unclear
        if verbose:
            print(f"fraction of confusing: {f_unclear}" )
            
        # more than 4 classification agreed when unclear discarded.
        if f_unclear <= unclear_limit and n_not >=min_generic_typing:
            if verbose:
                print(f"saving this case" )

            new_line = clear_unclear(line.copy())
            classification = parse_classification(new_line)

        # At least 6 classification and more than 4 (if to_unclear_limit==0.75) are 'unclear'
        elif f_unclear >= to_unclear_limit and n_classifications >= min_classifications_to_unclear:
            if verbose:
                print(f"saving this case as 'unclear' ")
            classification = "unclear"
        elif n_not <= min_review:
            if verbose:
                print(f"saving this case as 'None' ")
            classification = "None"
            
        if verbose:
            print(f"cannot save this case ( {f_unclear} > {to_unclear_limit} & {n_classifications}>{n_classifications}")

    return classification


def clear_unclear(line):
    """ remove the 'unclear' component from the given typing row """
    typing = list(line.typing)
    if "unclear" not in typing:
        return line # nothing to do
    
    ntypings = list(line.ntypings)
    index_unclear = typing.index("unclear")
    typing.pop(index_unclear)
    ntypings.pop(index_unclear)
    return pandas.Series({"typing": typing,
                          "ntypings": ntypings,
                          "ntypes": len(ntypings),
                          "nreviews": np.sum(ntypings)})


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

    MASTER_USER = ["jesper", "Joel Johansson", "Kate Maguire",
                   "Mathew Smith", "Umut", "Georgios Dimitriadis"]


    CLASSIFIED_KEY = ["ia-norm", "ia-pec", "ia", "sn ia", "ia_or_ianorm"]
    IA_PEC = ["ia-91bg","ia-91t","ia-other"]
        
    def load(self):
        """ """
        self._data = self.data_from_db(sql="where kind='typing'")
        self._data["value"] = self._data["value"].str.strip().str.lower()

    def load_sample(self):
        """ """
        from . import get_sample
        self._sample = get_sample()

    def store_typing(self, **kwargs):
        """ """
        if "classification" not in self.data:
            self.load_classification(**kwargs)
            
        data = self.get_classification_df()
        classification = self.data.groupby("target_name").first()["classification"]
        data = data.join(classification)
        return data.to_csv( io.get_target_typing(False), sep=" ")
        

    def load_classification(self, min_review=2,
                                  min_autotyping=3,
                                  min_generic_typing=3,
                                # Master keys
                                  min_review_master=1,
                                  min_autotyping_master=2,
                                  min_generic_typing_master=3):
        """ """
        self._load_classification(min_review=min_review, min_autotyping=min_autotyping,
                                  min_generic_typing=min_generic_typing)
        # master
        master = self.get_masterclassification(False)
        master._load_classification(min_review=min_review_master, min_autotyping=min_autotyping_master,
                                  min_generic_typing=min_generic_typing_master)
        
        # master successful classifications
        mdata = master.data[~master.data["classification"].isin(["None","confusing"])
                            ].groupby("target_name").first()["classification"]
        mdata.name = "master_classification"


        # Arbiter
        report = Reports()
        dataarbiter = report.data[report.data["value"].str.startswith("arbiter")].copy()
        dataarbiter["arbiter_classification"] = dataarbiter["value"].str.replace("arbiter:","")
        a = dataarbiter["arbiter_classification"].str.replace("^ia","ia-", regex=True).replace("gal","nonia",regex=False)
        a[a=="ia-"] = "ia"
        dataarbiter["arbiter_classification"] = a.copy()
        # cleaning naming conventio
        # This has target_name as index
        adata = dataarbiter.groupby("target_name")["arbiter_classification"].first() # early issue, all agree, tested
        
        #
        # merging
        #
        data = self.data.set_index("target_name") # target_name as index
        datamerged = data.join(mdata).join(adata).reset_index()
        datamerged["auto_classification"] = datamerged["classification"]
        # merge master
        mclassified = datamerged[~datamerged["master_classification"].isna()]["master_classification"]
        datamerged.loc[mclassified.index, "classification"] = mclassified

        # merge arbiter
        aclassified = datamerged[~datamerged["arbiter_classification"].isna()]["arbiter_classification"]
        datamerged.loc[aclassified.index, "classification"] = aclassified
        
        self._data = datamerged
        
        
    def _load_classification(self, **kwargs):
        # Normal
        class_df = self.get_classification_df()
        class_df["classification"] = class_df.apply(parse_classification, axis=1,
                                                    **kwargs)

        # are you reloading ?
        if "classification" in self.data:
            _ = self.data.pop("classification")
            
        self._data = self.data.set_index("target_name").join(class_df["classification"]).reset_index()


        
    # -------- #
    # GETTER   #
    # -------- #
    def get_masterclassification(self, incl_unclear=True):
        """ """
        this = self.__class__()
        MASTER_ID = _Users_().data.set_index("name").loc[self.MASTER_USER]["id"].values
        this._data = self.data[self.data["user_id"].isin(MASTER_ID)]
        if not incl_unclear:
            this._data = this._data[~(this._data["value"] == "unclear")]
            
        return this

    def get_classification_stats(self, based_on="classification", **kwargs):
        """ Get the distribution of classifications made. 

        Parameters
        ----------
        based_on: str
            classification key to used (from self.data); could be:
            - classification
            - master_classification
            - auto_classification

        **kwargs goes to get_data()

        Returns
        -------
        pandas.Series
            groupby().size() serie.
        """
        if "classification" not in self.data:
            self.load_classification()

        data = self.get_data(**kwargs)
        return data.groupby("target_name").first().groupby(based_on).size()
    
    
    def get_data(self, incl_unclear=True, min_classification=None,
                     types=None, 
                     goodcoverage=None, redshift_range=None,
                     x1_range=None, c_range=None, **kwargs):
        """ 

        incl_unclear: [bool] -optional-
            should the unclear typing be included ?

        min_classification: [None or int] -optional-
            minimum number of typing for this target. 
            'unclear' are not counted as classifications
            (this uses self.get_ntyped()) 


        goodcoverage: [None or bool] -optional-
            # sample.get_data() options
            should we only use target with good lightcurve sampling ?
            - None: no selection
            - True: only good sampling targets
            - False: only bad sample targets
            
        redshift_range, x1_range, c_range: [None or [min, max] ] -optional-
            # sample.get_data() options
            range to limit redshift, x1 or c, respectively
        
        **kwargs goes to self.sample.get_data()

        Returns
        -------
        DataFrame
        """
        data = self.data.copy()
        if not incl_unclear:
            data = data[data["value"] != "unclear"]
            
        if min_classification is not None and min_classification>0:
            ntyped = self.get_ntyped(incl_unclear=False)
            data = data[data["target_name"].isin(ntyped[ntyped>=min_classification].index)]
            
        if types is not None:
            data = data[data["value"].isin(np.atleast_1d(types))]
            
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
        return 
                         
    def get_nonia(self, incl_unclear=False):
        """ """
        list_to_get = self._NONIA_CASES.copy()
        if incl_unclear:
            list_to_get+="unclear"
            
        return self.data[self.data["value"].isin(list_to_get)]

    def get_classification_df(self):
        """ """
        data = self.get_data(incl_unclear=True)
        typings = data.groupby(["target_name"])["value"].apply(np.unique, return_counts=True)
        types =  pandas.DataFrame(typings.to_list(), columns=["typing", "ntypings"],
                                index=typings.index)#.explode(["typing","ntypings"])
        types["ntypes"] = types["ntypings"].apply(len)
        types["nreviews"] = types["ntypings"].apply(np.sum)
        return types
    
    # -------- #
    # PLOTTER  #
    # -------- #
    def show_classification_stats(self, explode = 0.1,
                                  based_on = "classification", 
                                  redshift_range = None, dataprop = {},
                                  **kwargs):
        """ """
        import matplotlib.pyplot as plt
        
        prop_general = {**dict(wedgeprops=dict(width=1, edgecolor='0.7', linewidth=0.2)), 
                **kwargs}


        class_stat = self.get_classification_stats( based_on, redshift_range=redshift_range,
                                                    **dataprop)

        class_stat["ia-pec"] = class_stat[self.IA_PEC].sum()
        class_stat["classified"] = class_stat[self.CLASSIFIED_KEY].sum()
        class_stat["non ia"] = class_stat[[k for k in self._NONIA_CASES if k in class_stat]].sum()

        # showing plots


        fig = plt.figure(figsize=[9,4])
        ax_main = fig.add_axes([0.05,0.15,0.4,0.8])
        ax_sub = fig.add_axes([0.52,0.15,0.4,0.8])

        # -> Main plot
        to_show = ["None","confusing", "unclear", "classified", "non ia"]
        data_show = class_stat[to_show]
        if type(explode) == float:
            explode = np.full_like(data_show.values, explode, dtype="float")

        colors = ["w","0.8", plt.cm.bone(0.3),  plt.cm.bone(0.7),
                  "tab:brown"]
        prop = {**dict(colors=colors, explode=explode, autopct='%.0f%%', 
                       startangle=90), **prop_general}

        # PLOTTING
        _ = ax_main.pie(data_show.values, labels=data_show.index, **prop)

        ax_main.text(0.5, -0.15, f"{np.sum(data_show.values)} Supernovae", 
                     transform=ax_main.transAxes,
                     fontsize="large", ha="center", va="bottom")

        # -> sub plot
        to_show = self.CLASSIFIED_KEY
        data_show = class_stat[to_show]
        if type(explode) == float:
            explode = np.full_like(data_show.values, explode, dtype="float")

        colors = ["tab:blue", "tab:orange", "wheat", "lavender", "lightsteelblue"]
        prop = {**dict(colors=colors, explode=explode, autopct='%.0f%%', 
                       startangle=0), **prop_general}

        # PLOTTING
        _ = ax_sub.pie(data_show.values, labels=data_show.index, **prop)

        ax_sub.text(0.5, -0.15, f"{np.sum(data_show.values)} Supernovae", 
                    transform=ax_sub.transAxes,
                    fontsize="large", ha="center", va="bottom")

        return fig

    
    def show_classifications(self, ax=None, fig=None, 
                                 classifications=None, per_target=True):
        """ """
        if classifications is not None:
            ndatas = classifications.value_counts().sort_values()
        elif per_target:
            ndatas = self.get_classification().value_counts().sort_values()
        else:
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

    
    @property
    def types(self):
        """ list of all possible types"""
        return self.data["value"].unique()

    @property
    def ia_types(self):
        """ list of all possible types"""
        alltypes = self.types
        return alltypes[["ia" in _ and _ !="not ia" for _ in alltypes ]]
