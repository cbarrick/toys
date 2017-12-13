class MLP(N.Module):
    def __init__(self, *chans):
        super().__init__()
        layers = []
        n = len(chans)
        for i in range(n-1):
            full = N.Linear(chans[i], chans[i+1])
            relu = N.ReLU(inplace=True)
            layers += [full, relu]
        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
