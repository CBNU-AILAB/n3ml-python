import torch


# class PoissonEncoder:
#     # Poisson processor for encoding images
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         # images.size: [b, c, h, w]
#         # spiked_images.size: [t, b, c, h, w]
#         b, c, h, w = images.size()
#         r = images.unsqueeze(0).repeat(self.time_interval, 1, 1, 1, 1) / 32.0
#         p = torch.rand(self.time_interval, b, c, h, w)
#         return (p <= r).float()
#
#
# class PoissonEncoder2:
#     def __init__(self, time_interval):
#         self.time_interval = time_interval
#
#     def __call__(self, images):
#         rate = torch.zeros(size)
#         rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)


class Encoder:
    # language=rst
    """
    Base class for spike encodings transforms.

    Calls ``self.enc`` from the subclass and passes whatever arguments were provided.
    ``self.enc`` must be callable with ``torch.Tensor``, ``*args``, ``**kwargs``
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class PoissonEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable PoissonEncoder which encodes as defined in
        ``bindsnet.encoding.poisson`

        :param time: Length of Poisson spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = poisson


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be
    non-negative, and give the firing rate in Hz. Inter-spike intervals (ISIs) for
    non-negative data incremented by one to avoid zero intervals while maintaining ISI
    distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    return spikes.view(time, *shape)


if __name__ == '__main__':
    r = torch.rand(1, 2, 2, 3)
    p = torch.rand(4, 1, 2, 2, 3)

    rr = r.unsqueeze(0).repeat(4, 1, 1, 1, 1)
    print(rr.size())

    print(rr)
    print(p)
    print(p <= rr)

    encoder = PoissonEncoder()

    print(encoder(r, 4))
