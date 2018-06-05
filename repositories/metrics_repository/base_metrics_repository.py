import abc
import pandas as pd

from messages import error_messages

metric_names_list = [
    ("SUB_type", "LOC_type"),
    ("NCLOC_type", "LOCNAMM_type"),
    ("NOC_type", "NOCS_type"),
    ("CSOA_type", "NOM_type"),
    ("CSO_type", "NOMNAMM_type"),
    ("CSA_type", "NOA_type"),
    ("WMC_type", "WMC_type"),
    ("WMC_type", "WMCWAMM_type"),
    ("OCavg_type", "AMW_type"),
    ("OCavg_type", "AMWAMM_type"),
    ("WOC_type", "WOC_type"),
    ("TCC_type", "TCC_type"),
    ("LCOM_type", "LCOM5_type"),
    ("", "ATFD_type"),
    ("Dpt_type", "FDP_type"),
    ("RFC_type", "RFC_type"),
    ("CBO_type", "CBO_type"),
    ("QUERY_type", "CFNAMM_type"),
    ("CSA_type", "NOAM_type"),
    ("CSA_type", "NOPA_type"),
    ("DIT_type", "DIT_type"),
    ("INT_type", "NOI_type"),
    ("NOC_type", "NOC_type"),
    ("NOOC_type", "NMO_type"),
    ("NOIC_type", "NIM_type"),
    ("INT_type", "NOII_type"),
    ("LOC_method", "LOC_method"),
    ("v(G)_method", "CYCLO_method"),
    ("NEST_method", "MAXNESTING_method"),
    ("CALL_method", "CLNAMM_method"),
    ("NP_method", "NOP_method"),
    ("IMP_method", "NOAV_method"),
    ("", "ATLD_method"),
    ("STAT_method", "NOLV_method"),
    ("STAT_method", "FANOUT_method"),
    ("CALL_method", "ATFD_method"),
    ("CALLED_method", "CFNAMM_method"),
    ("", "CINT_method"),
    ("", "CDISP_method"),
    ("CALL_method", "CC_method"),
    ("CALL_method", "CM_method"),
    ("LAA_method", "LAA_method"),
    ("LOC", "LOC_package"),
    ("C(rec)", "NOCS_package"),
    ("METH(rec)", "NOM_package"),
    ("L(J)", "LOC_project"),
    ("LOCp", "NOPK_project"),
    ("C", "NOCS_project"),
    ("METH", "NOM_project"),
    ("type_type", "type"),
    ("method_method", "method")
]

class base_metrics_repository:
    def __init__(self, metaclass=abc.ABCMeta):
        pass

    @abc.abstractmethod
    def get_metrics_dataframe(self, prefix, dataset_id):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics_dataframe'))

    @staticmethod
    def get_transformed_dataset(old_df):
        new_df = pd.DataFrame()
        for old_metric, new_metric in metric_names_list:
            #cleaned_old_metric = old_metric.replace("_method", "").replace.replace("_type", "")
            if old_metric in old_df.columns:
                new_df.loc[:, new_metric] = old_df.loc[:, old_metric]
            else:
                new_df.loc[:, new_metric] = pd.Series()

        return new_df

