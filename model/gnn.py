import torch
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, batch, device):
        super(GCN, self).__init__()
        self.batch = batch
        self.W1 = torch.nn.Parameter(
            torch.randn(self.batch, 23, 10).type(torch.FloatTensor).to(device),
            requires_grad=True,
        )
        self.W2 = torch.nn.Parameter(
            torch.randn(self.batch, 10, 10).type(torch.FloatTensor).to(device),
            requires_grad=True,
        )
        self.W3 = torch.nn.Parameter(
            torch.randn(23 * 10, 1).type(torch.FloatTensor).to(device),
            requires_grad=True,
        )

    def forward(self, x, A, D):
        # Symmetric Normalization
        activation = F.leaky_relu
        hidden_layer_1 = activation(D.bmm(A).bmm(D).bmm(x).bmm(self.W1))
        y_pred = activation(D.bmm(A).bmm(D).bmm(hidden_layer_1).bmm(self.W2))
        y_pred = y_pred.view(self.batch, 23 * 10)
        y_pred = activation(y_pred.mm(self.W3))
        return y_pred


if __name__ == "__main__":
    raise NotImplementedError
