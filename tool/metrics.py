import torch
def relative_rmse_error_ornl(original, reconstructed, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original = torch.as_tensor(original, dtype=torch.float32, device=device)
    reconstructed = torch.as_tensor(reconstructed, dtype=torch.float32, device=device)

    rmse = torch.sqrt(torch.mean((original - reconstructed) ** 2))
    data_range = torch.max(original) - torch.min(original)

    relative_rmse = torch.where(
        data_range != 0, rmse / data_range, torch.tensor(0.0, device=device)
    )
    return relative_rmse
