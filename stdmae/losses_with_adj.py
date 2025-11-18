import torch
import torch.distributed as dist
from basicts.losses import masked_mae
from stdmae.stdmae_runner.mask_runner import CURRENT_MODEL_FOR_LOSS


def _maybe_all_reduce_tensor(tensor):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        tensor = torch.tensor(float(tensor))
    if dist.is_initialized():
        tensor = tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / float(dist.get_world_size())
    return tensor


def masked_mae_with_aux(pred, label, null_val: float = 0.0):
    loss_main = masked_mae(pred, label, null_val=null_val)
    aux_loss = 0.0
    lambda_a = 0.0
    mdl = CURRENT_MODEL_FOR_LOSS
    if mdl is not None:
        aux = getattr(mdl, "aux_loss", None)
        if hasattr(mdl, "lambda_a"):
            lambda_a = float(getattr(mdl, "lambda_a"))
        else:
            cfg = getattr(mdl, "cfg", None)
            if cfg is not None:
                lambda_a = float(cfg.get("MODEL", {}).get("PARAM", {}).get("lambda_a", 0.0))
        if aux is not None:
            if not torch.is_tensor(aux):
                aux = torch.tensor(float(aux), device=loss_main.device)
            else:
                aux = aux.to(loss_main.device)
            aux = _maybe_all_reduce_tensor(aux)
            return loss_main + lambda_a * aux
    return loss_main
