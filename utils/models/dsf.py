import torch
import torch.nn.functional as F

def write_log(log_file, text, mode='a'):
    with open(log_file, mode) as f:
        f.write(text)

class vMF():
    def __init__(self, X, Y, Z, args):
        """
        Compute parameters of von Mises-Fisher distributions.

        Args:
            X: Query features, shape (B, D).
            Y: Key features, shape (B, D).
            Z: Queue features, shape (K, D) or (D, K). If not None, it is transposed inside (self.Z = Z.T).
                - For SimCLR-style: Z is None.
                - For MoCo-style:   Z is the negative queue.
        """

        self.X = X
        self.Y = Y

        if Z is not None: self.Z = Z.T

        self.bs = args.batch_size
        self.dim = args.moco_dim
        self.v = (self.dim/2)-1
        self.args = args
        self.ns_X, self.ns_Y = args.ns, args.ns

        self.get_parameters()


    def get_parameters(self):
        "Get basic parameters of von Mises-Fisher Distribution"
        B, D = self.X.shape
        assert D == self.dim, f"Dim mismatch: X has {D}, args.moco_dim={self.dim}"
        eps = 1e-8

        # X
        # (B*ns_X, D) -> (B, ns_X, D) -> mean over ns_X
        self.X_bar = torch.mean(self.X.view(-1, self.ns_X, self.dim), dim=1)  # (B, D)
        self.R_bar_X = torch.norm(self.X_bar, dim=1) * self.args.R_bar_coeff  # (B,)
        self.mu_X = F.normalize(self.X_bar, dim=1)
        denom_X = torch.clamp(1.0 - self.R_bar_X**2, min=eps)
        self.kappa_X = self.R_bar_X * (self.dim - self.R_bar_X**2) / denom_X  # (B,)

        # Y
        self.Y_bar = torch.mean(self.Y.view(-1, self.ns_Y, self.dim), dim=1)
        self.R_bar_Y = torch.norm(self.Y_bar, dim=1) * self.args.R_bar_coeff
        self.mu_Y = F.normalize(self.Y_bar, dim=1)
        denom_Y = torch.clamp(1.0 - self.R_bar_Y**2, min=eps)
        self.kappa_Y = self.R_bar_Y * (self.dim - self.R_bar_Y**2) / denom_Y

    def MBF_uniform_expansion(self, dim, kappa, m=3):
        """
        Maclaurin-Bessel (uniform) asymptotic expansion helper for I_v(kappa).
        Returns: eta, p, z, coeff, sums  (all tensors on kappa.device)
        See https://dlmf.nist.gov/10.41 for more details.
        """

        # Here, dim actually corresponds to v (order of the Bessel function)
        v = torch.tensor(dim, device=kappa.device, dtype=kappa.dtype)
        z = kappa / v

        def _U_series_term(p, m=0):
            if m==0 : return torch.ones_like(p)
            elif m== 1 : return 1/24 * (3*p - 5 * p**3)
            elif m== 2 : return 1/1152 * (81*p**2 - 462*p**4 + 385*p**6)
            elif m== 3 : return 1/414720 * (30375*p**3 - 369603*p**5 + 765765*p**7 - 425425*p**9)
            else : raise Exception("MBF_uniform_expansion has out of range for m")
            # for m=4,5,6 it is referenced at DLMF (https://dlmf.nist.gov/10.41)

        def _eta_uniform(z): return torch.sqrt(1+z**2) + torch.log(z / (1 + torch.sqrt(1+z**2)))
        def _p_uniform(z): return torch.pow(1+z**2, -0.5)
        
        eta = _eta_uniform(z)
        p = _p_uniform(z)

        coeff = torch.exp(v*eta) / ( torch.sqrt(2*torch.pi*v) * torch.pow(1+z**2, 0.25) ) # 'coeff' is currently unused but kept for potential future extensions.
        sums = 0
        for i in range(m+1):
            sums += ( _U_series_term(p=p, m=i) / v**i )

        return eta, p, z, coeff, sums


class DSF_KLD(vMF):
    """
    Compute the Kullback-Leibler divergence between von Mises-Fisher (vMF) distributions
    in a MoCo v2-style contrastive learning setting.

    This class assumes a query-key-queue formulation where X and Y form positive pairs
    and Z represents the negative feature queue.
    """
    def __init__(self, X, Y, Z, K, args):
        super().__init__(X, Y, Z, args)

        self.K = K

        if self.args.kappa_normalization:
            self.kappa_X /= self.dim
            self.kappa_Y /= self.dim

        # mu_Z
        self.mu_Z = self.Z

        # kappa_Z
        kappa_X, kappa_Y = self.kappa_X.clone().detach(), self.kappa_Y.clone().detach()
        mean_kappa = torch.mean(torch.concat([kappa_X, kappa_Y]))
        self.kappa_Z = torch.full((self.K,), mean_kappa).cuda()

    def APK(self):
        if self.args.ns > 1 :
            return self.R_bar_X
        else :
            return torch.ones(self.args.batch_size).cuda()
    
    def p1(self):
        first_term = self.v * torch.log(self.kappa_X / self.kappa_Y)
        return first_term

    def p2(self):
        # log(Iv(k1)/Iv(k2))
        eta_X, _, z_X, _, sums_X = self.MBF_uniform_expansion(self.v, self.kappa_X)
        eta_Y, _, z_Y, _, sums_Y = self.MBF_uniform_expansion(self.v, self.kappa_Y)

        second_term = self.v*(eta_Y - eta_X) + 0.25*torch.log((1 + z_X**2) / (1 + z_Y**2)) + torch.log(sums_Y / sums_X)
        return second_term

    def p3(self):
        # Ap(k0)*(k0 - k1 mu1^Tmu0)
        third_term = self.APK() * (self.kappa_X - self.kappa_Y * torch.einsum('nc,nc->n', [self.mu_X, self.mu_Y]))
        return third_term
    
    def n1(self):
        first_term = self.v * torch.log( torch.outer(self.kappa_X, self.kappa_Z.pow(-1)) )
        return first_term
    
    def n2(self):
        eta_X, _, z_X, _, sums_X = self.MBF_uniform_expansion(self.v, self.kappa_X) # B
        eta_Z, _, z_Z, _, sums_Z = self.MBF_uniform_expansion(self.v, self.kappa_Z) # 4096

        # into (B, 4096)
        eta_X, z_X, sums_X = eta_X.unsqueeze(-1).expand(self.bs, self.K), z_X.unsqueeze(-1).expand(self.bs, self.K), sums_X.unsqueeze(-1).expand(self.bs, self.K)
        eta_Z, z_Z, sums_Z = eta_Z.unsqueeze(0).expand(self.bs, self.K), z_Z.unsqueeze(0).expand(self.bs, self.K), sums_Z.unsqueeze(0).expand(self.bs, self.K)

        second_term = self.v*(eta_Z - eta_X) + 0.25*torch.log((1 + z_X**2) / (1 + z_Z**2)) + torch.log(sums_Z / sums_X)

        return second_term
    
    def n3(self):
        temp = self.kappa_X.unsqueeze(-1).expand(self.bs, self.K) - self.kappa_Z.unsqueeze(0).expand(self.bs, self.K) * torch.matmul(self.mu_X, self.mu_Z.T)
        third_term = self.APK().unsqueeze(-1) * temp
        return third_term

    def get_logits(self):
        l_pos = ( self.p1() + self.p2() + self.p3() ).unsqueeze(-1)
        l_neg = self.n1() + self.n2() + self.n3()
        return torch.cat([l_pos, l_neg], dim=1)
    

class DSF_KLD_MoCov3(vMF):
    """
    Calculate Kullback-Leibler divergence between vMF distributions in a SimCLR-style (MoCo v3-style) setup.

    Args:
        X: Query features, shape (N, D).
        Y: Key features, shape (M, D).
            Typically, M = N * (number of GPUs).
        args: Hyperparameters including ns, kappa_normalization, etc.

    Returns:
        get_logits(): three tensors of shape (N, M) corresponding to the components of the DSF.
    """

    def __init__(self, X, Y, args):
        super().__init__(X, Y, None, args)

        if self.args.kappa_normalization:
            self.kappa_X /= self.dim
            self.kappa_Y /= self.dim

    def APK(self):
        return self.R_bar_X
    
    def get_logits(self):
        """
        Compute three components of the divergence-based similarity between all pairs.

        Returns:
            first_term:  Tensor of shape (N, M)
            second_term: Tensor of shape (N, M)
            third_term:  Tensor of shape (N, M)
        """

        N, M = self.mu_X.size()[0], self.mu_Y.size()[0]

        # first term
        first_term = self.v * torch.log(self.kappa_X.unsqueeze(1) / self.kappa_Y.unsqueeze(0))

        # second term
        eta_X, _, z_X, _, sums_X = self.MBF_uniform_expansion(self.v, self.kappa_X)
        eta_Y, _, z_Y, _, sums_Y = self.MBF_uniform_expansion(self.v, self.kappa_Y)

        eta_X, z_X, sums_X = eta_X.unsqueeze(-1).expand(N, M), z_X.unsqueeze(-1).expand(N, M), sums_X.unsqueeze(-1).expand(N, M)
        eta_Y, z_Y, sums_Y = eta_Y.unsqueeze(0).expand(N, M),  z_Y.unsqueeze(0).expand(N, M),  sums_Y.unsqueeze(0).expand(N, M)

        second_term = self.v * (eta_Y - eta_X) + 0.25*torch.log((1 + z_X**2) / (1 + z_Y**2)) - torch.log(sums_Y / sums_X)

        temp = self.kappa_X.unsqueeze(-1).expand(N, M) - self.kappa_Y.unsqueeze(0).expand(N, M) * torch.matmul(self.mu_X, self.mu_Y.T)
        third_term = self.APK().unsqueeze(-1) * temp

        return first_term, second_term, third_term