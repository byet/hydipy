from scipy.stats import norm, expon, beta, uniform


class NormalDistribution:
    def __init__(self, mean=0.0, sd=1.0):
        self.mean = mean
        self.sd = sd

        if isinstance(mean, str):
            if isinstance(sd, str):
                self.cdf = self._cdf_mean_sd_unk
                self.ppf = self._ppf_mean_sd_unk
            else:
                self.cdf = self._cdf_mean_unk
                self.ppf = self._ppf_mean_unk
        elif isinstance(sd, str):
            self.cdf = self._cdf_sd_unk
            self.ppf = self._ppf_sd_unk
        else:
            self.cdf = self._cdf_kn
            self.ppf = self._ppf_kn

    def _cdf_kn(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.sd)

    def _cdf_mean_unk(self, x, mu):
        return norm.cdf(x, loc=mu, scale=self.sd)

    def _cdf_sd_unk(self, x, sigma):
        return norm.cdf(x, loc=self.mean, scale=sigma)

    def _cdf_mean_sd_unk(self, x, mu, sigma):
        return norm.cdf(x, loc=mu, scale=sigma)

    def _ppf_kn(self, x):
        return norm.ppf(x, loc=self.mean, scale=self.sd)

    def _ppf_mean_unk(self, x, mu):
        return norm.ppf(x, loc=mu, scale=self.sd)

    def _ppf_sd_unk(self, x, sigma):
        return norm.ppf(x, loc=self.mean, scale=sigma)

    def _ppf_mean_sd_unk(self, x, mu, sigma):
        return norm.ppf(x, loc=mu, scale=sigma)


class UniformDistribution:
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

        if isinstance(a, str):
            if isinstance(b, str):
                self.cdf = self._cdf_a_b_unk
                self.ppf = self._ppf_a_b_unk
            else:
                self.cdf = self._cdf_a_unk
                self.ppf = self._ppf_a_unk
        elif isinstance(b, str):
            self.cdf = self._cdf_b_unk
            self.ppf = self._ppf_b_unk
        else:
            self.cdf = self._cdf_kn
            self.ppf = self._ppf_kn

    def _cdf_kn(self, x):
        scale = self.b - self.a
        return uniform.cdf(x, loc=self.a, scale=scale)

    def _cdf_a_unk(self, x, a):
        scale = self.b - a
        return uniform.cdf(x, loc=a, scale=scale)

    def _cdf_b_unk(self, x, b):
        scale = b - self.a
        return uniform.cdf(x, loc=self.a, scale=scale)

    def _cdf_a_b_unk(self, x, a, b):
        scale = b - a
        return uniform.cdf(x, loc=a, scale=scale)

    def _ppf_kn(self, x):
        scale = self.b - self.a
        return uniform.ppf(x, loc=self.a, scale=scale)

    def _ppf_a_unk(self, x, a):
        scale = self.b - a
        return uniform.ppf(x, loc=a, scale=scale)

    def _ppf_b_unk(self, x, b):
        scale = b - self.a
        return uniform.ppf(x, loc=self.a, scale=scale)

    def _ppf_a_b_unk(self, x, a, b):
        scale = b - a
        return uniform.ppf(x, loc=a, scale=scale)


class ExponentialDistribution:
    def __init__(self, rate=1):
        self.rate = rate

        if isinstance(rate, str):
            self.cdf = self._cdf_rate_unk
            self.ppf = self._ppf_rate_unk
            self.scale = f"1 / {rate}"
        else:
            self.cdf = self._cdf_kn
            self.ppf = self._ppf_kn
            self.scale = 1 / rate

    def _cdf_kn(self, x):
        return expon.cdf(x, scale=self.scale)

    def _cdf_rate_unk(self, x, lam):
        return expon.cdf(x, scale=(1 / lam))

    def _ppf_kn(self, x):
        return expon.ppf(x, scale=self.scale)

    def _ppf_rate_unk(self, x, lam):
        return expon.ppf(x, scale=(1 / lam))


class BetaDistribution:
    def __init__(self, alpha=0.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

        if isinstance(alpha, str):
            if isinstance(beta, str):
                self.cdf = self._cdf_a_b_unk
                self.ppf = self._ppf_a_b_unk
            else:
                self.cdf = self._cdf_a_unk
                self.ppf = self._ppf_a_unk
        elif isinstance(beta, str):
            self.cdf = self._cdf_b_unk
            self.ppf = self._ppf_b_unk
        else:
            self.cdf = self._cdf_kn
            self.ppf = self._ppf_kn

    def _cdf_kn(self, x):
        return beta.cdf(x, a=self.alpha, b=self.beta)

    def _cdf_a_unk(self, x, alpha):
        return beta.cdf(x, a=alpha, b=self.beta)

    def _cdf_b_unk(self, x, beta):
        return beta.cdf(x, a=self.alpha, b=beta)

    def _cdf_a_b_unk(self, x, alpha, beta):
        return beta.cdf(x, a=alpha, b=beta)

    def _ppf_kn(self, x):
        return beta.ppf(x, a=self.alpha, b=self.beta)

    def _ppf_a_unk(self, x, alpha):
        return beta.ppf(x, a=alpha, b=self.beta)

    def _ppf_b_unk(self, x, beta):
        return beta.ppf(x, a=self.alpha, b=beta)

    def _ppf_a_b_unk(self, x, alpha, beta):
        return beta.ppf(x, a=alpha, b=beta)
