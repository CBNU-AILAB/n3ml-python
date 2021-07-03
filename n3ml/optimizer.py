import torch


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

    def step(self, model, spiked_input, spiked_label):
        # print(model)

        # lr = 0.0001  #
        lr = 0.01

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
                    numer = (spiked_label[j]-layer[l].s[j])
                    denom = 0
                    for i in range(layer[l].in_neurons):
                        for k in range(layer[l].delays):
                            denom += (layer[l].w[j, i, k]*dydt(spike_time[l][j], spike_time[l+1][i], layer[l].d[k], layer[l].tau_rc))
                    dldx[j] = numer/(denom+1e-15)
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
                    dldx[i] = numer/denom
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
