from scipy.stats import norm, expon


class NormalDistribution:
    def __init__(self, mean=0., sd=1.):
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
        return expon.cdf(x, scale=(1/lam))

    def _ppf_kn(self, x):
        return expon.ppf(x, scale=self.scale)

    def _ppf_rate_unk(self, x, lam):
        return expon.ppf(x, scale=(1/lam))
