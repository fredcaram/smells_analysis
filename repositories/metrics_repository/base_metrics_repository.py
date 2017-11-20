import abc

from messages import error_messages

class base_metrics_repository:
    def __init__(self, metaclass=abc.ABCMeta):
        pass

    @abc.abstractmethod
    def get_metrics_dataframe(self, prefix, dataset_id):
        raise NotImplementedError(error_messages.NOT_IMPLEMENTED_ERROR_MESSAGE('get_metrics_dataframe'))

