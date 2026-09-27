import torch


@torch.no_grad()
def slider_teachers(base, positive, negative, eta, normalize_to=None, directions=(1, -1)):
    direction = positive.float() - negative.float()
    teachers = [(base.float() + sign * eta * direction).detach() for sign in directions]
    if normalize_to is not None:
        reference_norm = normalize_to.float().flatten(1).norm(dim=1)
        teachers = [teacher * (reference_norm / teacher.flatten(1).norm(dim=1).clamp_min(1e-8)).reshape(
            -1, *([1] * (teacher.ndim - 1))) for teacher in teachers]
    return tuple(teachers)
