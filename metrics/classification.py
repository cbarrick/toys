EPSILON = 1e-7


class Sum:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.val = 0

    def accumulate(self, batch):
        try:
            self.val += batch.sum(**self.kwargs)
        except (TypeError, AttributeError):
            self.val += batch

    def reduce(self):
        val = self.val
        self.val = 0
        return val


class Mean:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        try:
            n = len(batch)
            try:
                val = batch.mean(**self.kwargs)
            except (TypeError, AttributeError):
                # careful, ByteTensor will likely overflow
                # this is common with `==` comparisons
                val = batch.sum(**self.kwargs) / n
        except (TypeError, AttributeError):
            n = 1
            val = batch

        if self.n == 0:
            self.n = n
            self.val = val

        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

    def reduce(self):
        val = self.val
        self.n = 0
        self.val = 0
        return val


class Accuracy:
    def __init__(self, **kwargs):
        self.val = Mean(**kwargs)

    def accumulate(self, y, h):
        val = (y == h)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TruePositives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h == self.target) & (y == self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalsePositives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h == self.target) & (y != self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TrueNegatives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h != self.target) & (y != self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalseNegatives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h != self.target) & (y == self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class Precision:
    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        return tp / (tp + fp + EPSILON)


class Recall:
    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fn = self.fn.reduce()
        return tp / (tp + fn + EPSILON)


class FScore:
    def __init__(self, beta=1, target=1, **kwargs):
        self.beta = beta
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        fn = self.fn.reduce()
        beta2 = self.beta ** 2
        tp2 = (1 + beta2) * tp
        fn2 = beta2 * fn
        return tp2 / (tp2 + fn2 + fp + EPSILON)
