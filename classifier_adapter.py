from pytorch_adapt.adapters.base_adapter import BaseGCAdapter
from pytorch_adapt.adapters.utils import with_opt
from pytorch_adapt.hooks import ClassifierHook

class ClassifierAdapter(BaseGCAdapter):
    """
    Wraps [AlignerPlusCHook][pytorch_adapt.hooks.AlignerPlusCHook].

    |Container|Required keys|
    |---|---|
    |models|```["G", "C"]```|
    |optimizers|```["G", "C"]```|
    """

    def init_hook(self, hook_kwargs):
        opts = with_opt(list(self.optimizers.keys()))
        self.hook = self.hook_cls(opts, **hook_kwargs)

    @property
    def hook_cls(self):
        return ClassifierHook
