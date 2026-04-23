"""
Test configuration for OPSS.

Stubs the opss top-level import chain (pipeline -> vision -> torch)
so that pure-numpy submodules (UKF3D, FeatureExtractor3D, config) can
be imported without torch being loadable.

The stubs are ONLY for the pipeline/vision import path that is
unrelated to the code under test.  All UKF/feature/config logic
runs against real numpy/scipy.
"""
import sys
import types

# Stub packages that opss/__init__.py eagerly imports but that are
# irrelevant to state-estimation tests and pull in torch.
_STUB_PACKAGES = [
    "torch", "torch.nn", "torch.optim", "torch.utils",
    "torch.utils.data",
    "opss.pipeline", "opss.pipeline.core",
    "opss.vision",
]

for _pkg in _STUB_PACKAGES:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = []
        _mod.__package__ = _pkg
        sys.modules[_pkg] = _mod

# Provide the names that opss/__init__.py expects from pipeline.core
_pipeline_core = sys.modules["opss.pipeline.core"]
_pipeline_core.OPSSPipeline = None
_pipeline_core.PipelineConfig = None
_pipeline_core.get_pipeline = None

# Provide minimal torch.nn stubs so nn_model.py can define classes
_torch = sys.modules["torch"]
_torch.no_grad = lambda: type("ctx", (), {
    "__enter__": lambda s: s, "__exit__": lambda *a: None
})()
_torch.FloatTensor = lambda x: None
_torch.save = lambda *a, **kw: None
_torch.load = lambda *a, **kw: {}
_torch.manual_seed = lambda s: None

class _FakeModule:
    """Minimal stand-in for torch.nn.Module so that subclasses can be defined."""
    def __init__(self, *a, **kw):
        pass
    def __init_subclass__(cls, **kw):
        pass
    def parameters(self):
        return []
    def state_dict(self):
        return {}
    def load_state_dict(self, d):
        pass
    def eval(self):
        return self
    def train(self, mode=True):
        return self

_nn = sys.modules["torch.nn"]
_nn.Module = _FakeModule
_nn.Sequential = lambda *a: None
_nn.Linear = lambda *a: None
_nn.Tanh = lambda: None
_nn.MSELoss = lambda: None
