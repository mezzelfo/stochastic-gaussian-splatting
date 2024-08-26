import torch

class VariationalInferenceMultivariateNormal(torch.nn.Module):
    def __init__(self, mu_prior, L_prior) -> None:
        super().__init__()

        self.mu_prior = mu_prior.clone().detach()
        self.L_prior = L_prior.clone().detach()
        self.sigma_prior = torch.einsum('...ij,...kj->...ik', self.L_prior, self.L_prior)
        self.sigma_prior = self.sigma_prior + 1e-9 * torch.eye(self.sigma_prior.shape[-1], device=self.sigma_prior.device)

        self.sigma_prior_logdet = torch.logdet(self.sigma_prior)
        self.sigma_prior_inv = self.sigma_prior.inverse()

        self.mu_posterior = torch.nn.Parameter(self.mu_prior.clone().detach())
        self.L_posterior = torch.nn.Parameter(self.L_prior.clone().detach())

    def forward(self):
        eps = torch.randn_like(self.mu_posterior)
        z = self.mu_posterior + torch.einsum('...ij,...j->...i', self.L_posterior, eps)
        return z
    
    def KullbackLeibler_divergence(self):
        #KL(posterior || prior)
        sigma_posterior = torch.einsum('...ij,...kj->...ik', self.L_posterior, self.L_posterior)
        logdet_ratio = self.sigma_prior_logdet - torch.logdet(sigma_posterior)
        mean_diff = self.mu_posterior - self.mu_prior
        quadratic = torch.einsum('...i,...ij,...j->...', mean_diff, self.sigma_prior_inv, mean_diff)
        trace_term = torch.einsum('...ij,...ji->...', self.sigma_prior_inv, sigma_posterior)
        return 0.5 * (logdet_ratio - 3.0 + quadratic + trace_term)