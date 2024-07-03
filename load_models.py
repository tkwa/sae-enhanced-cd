# %%
import pickle
import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
from transformer_lens import HookedTransformer

import iit.model_pairs as mp


# %%

import circuits_benchmark.benchmark.cases.case_3 as case_3

task = case_3.Case3()
task_idx = task.get_index()
# %%

dir_name = f"/home/tkwa/src/InterpBench/{task_idx}"
cfg_dict = pickle.load(open(f"{dir_name}/ll_model_cfg.pkl", "rb"))
cfg = HookedTransformerConfig.from_dict(cfg_dict)
model = HookedTransformer(cfg)
weights = torch.load(f"{dir_name}/ll_model.pth")
model.load_state_dict(weights)

# %%

# turn off grads
model.eval()
model.requires_grad_(False)
torch.set_grad_enabled(False)
# %%
# load high level model
import importlib
from circuits_benchmark.utils.iit import make_iit_hl_model
import circuits_benchmark.utils.iit.correspondence as correspondence
# importlib.reload(circuits_benchmark)
# import iit.model_pairs as mp
# from iit.model_pairs import HLNode

def make_model_pair(benchmark_case):
    hl_model = benchmark_case.build_transformer_lens_model()
    hl_model = make_iit_hl_model(hl_model, eval_mode=True)
    tracr_output = benchmark_case.get_tracr_output()
    hl_ll_corr = correspondence.TracrCorrespondence.from_output(
            case=benchmark_case, tracr_output=tracr_output
        )
    model_pair = mp.StrictIITModelPair(hl_model, model, hl_ll_corr)
    return model_pair
# %%
import iit.model_pairs as mp
mp.HLNode
# %%
