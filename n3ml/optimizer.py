import torch

import n3ml.network


def y(t_j, t_i, d_k, tau):
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return t * torch.exp(1 - t / tau) / tau
    return 0


def dydt(t_j, t_i, d_k, tau):
    # w.r.t t_j
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return torch.exp(1 - t / tau) / tau - t * torch.exp(1 - t / tau) / (tau ** 2)
    return 0


def dydt2(t_j, t_i, d_k, tau):
    # w.r.t t_i
    if t_j >= 0 and t_i >= 0:
        t = t_j - t_i - d_k
        if t >= 0:
            return -torch.exp(1 - t / tau) / tau + t * torch.exp(1 - t / tau) / (tau ** 2)
    return 0


class Bohte:
    def __init__(self):
        pass

    def step(self, model, spiked_input, spiked_label, epoch):
        # print(model)

        # lr = 0.0001  #
        # lr = 0.01
        lr = 0.0075

        layer = []
        spike_time = [spiked_input]  # with input spike

        for l in model.layer.values():
            layer.append(l)
            spike_time.append(l.s)
        layer.reverse()
        spike_time.reverse()

        error = []

        for l in range(len(layer)):
            if l == 0:
                # print("last layer")
                dldx = torch.zeros(layer[l].out_neurons)
                for j in range(layer[l].out_neurons):
                    numer = (layer[l].s[j]-spiked_label[j])
                    denom = 0
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            denom += (layer[l].w[j, i, k]*dydt(spike_time[l][j], spike_time[l+1][i], layer[l].d[k], layer[l].tau_rc))
                    dldx[j] = -numer/(denom+1e-15)
                error.append(dldx)
                dxdw = torch.zeros_like(layer[l].w)
                for j in range(layer[l].out_neurons):
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            dxdw[j, i, k] = y(spike_time[l][j], spike_time[l+1][i], layer[l].d[k], layer[l].tau_rc)
                g = torch.zeros_like(layer[l].w)
                for j in range(layer[l].out_neurons):
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            g[j, i, k] = -lr * dxdw[j, i, k] * dldx[j]
                # print(g.detach().numpy())
                layer[l].w += g
                # print(layer[l].w.detach().numpy())
                layer[l].w[:] = torch.clamp(layer[l].w, 0.02, 1)  #
            else:
                # print("intermediate layer")
                dldx = torch.zeros(layer[l].out_neurons)
                for i in range(layer[l].out_neurons):
                    numer = 0
                    for j in range(layer[l-1].out_neurons):
                        sum = 0
                        for k in range(layer[l-1].delays):
                            sum += layer[l-1].w[j, i, k] * dydt2(spike_time[l-1][j], spike_time[l][i], layer[l-1].d[k], layer[l-1].tau_rc)
                        numer += error[-1][j] * sum
                    denom = 0
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            denom += layer[l].w[i, h, m] * dydt(spike_time[l][i], spike_time[l+1][h], layer[l].d[m], layer[l].tau_rc)
                    dldx[i] = -numer/(denom+1e-15)
                error.append(dldx)
                dxdw = torch.zeros_like(layer[l].w)
                for i in range(layer[l].out_neurons):
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            dxdw[i, h, m] = y(spike_time[l][i], spike_time[l+1][h], layer[l].d[m], layer[l].tau_rc)
                g = torch.zeros_like(layer[l].w)
                for i in range(layer[l].out_neurons):
                    for h in range(layer[l].in_neurons):
                        for m in range(layer[l].delays):
                            g[i, h, m] = -lr * dxdw[i, h, m] * dldx[i]
                # print(g)
                layer[l].w += g
                layer[l].w[:] = torch.clamp(layer[l].w, 0.02, 1)


class TavanaeiAndMaida:
    def __init__(self,
                 model: n3ml.network.Network,
                 lr: float = 0.0005) -> None:
        self.model = model
        self.lr = lr

    def step(self, spike_buffer, spiked_label, label):
        """"""
        """
            텐서로 변환된 spike_buffer[b]의 크기는 [epsilon, # neurons]와 같다.
        """
        buffer = {}
        for b in spike_buffer:
            buffer[b] = torch.stack(spike_buffer[b])

        if spiked_label[label] > 0.5:  # target neuron fires at that time
            in_grad = torch.zeros(self.model.fc2.out_neurons)
            for i in range(self.model.fc2.out_neurons):
                if i == label:
                    if torch.sum(buffer['fc2'][:, i]) < 1:
                        in_grad[i] = 1
                else:
                    if torch.sum(buffer['fc2'][:, i]) > 0:
                        in_grad[i] = -1

            # print(in_grad.numpy())

            # Compute propagated error in line 17-18
            e = torch.matmul(in_grad, self.model.fc2.w) * (torch.sum(buffer['fc1'], dim=0) > 0)

            # Update the weights in last layer in line 19
            updates = torch.ger(in_grad, torch.sum(buffer['fc1'], dim=0))
            # print(updates)
            self.model.fc2.w += updates * self.lr

            updates = torch.ger(e, torch.sum(buffer['inp'], dim=0))
            self.model.fc1.w += updates * self.lr
            # print(updates)
