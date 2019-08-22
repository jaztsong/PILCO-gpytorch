import math
import torch
import gpytorch

n = 30
train_x = torch.zeros(int(pow(n, 2)), 2)
sin_y = torch.zeros(int(pow(n, 2)))
cos_y = torch.zeros(int(pow(n, 2)))
add_y = torch.zeros(int(pow(n, 2)))
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
        # True function is sin(2*pi*x1*x2) with Gaussian noise
        cos_y[i * n + j] = math.cos((i + j) * (2 * math.pi)) + 0.5 * torch.rand(1, 1, 1)
        sin_y[i * n + j] = math.sin((i + j) * (2 * math.pi)) + 0.5 * torch.rand(1, 1, 1)
        add_y[i * n + j] = (i + j) + 0.8 * torch.rand(1, 1, 1)

train_x = train_x.repeat(3,1,1)
# 3-dimension output
train_y = torch.stack((sin_y, cos_y, add_y)).squeeze(-1)


# # Training data is 11 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100).view(1, -1, 1).repeat(3, 1, 1)
# # True function is sin(2*pi*x) with Gaussian noise
# sin_y = torch.sin(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
# sin_y_short = torch.sin(train_x[0] * (math.pi)) + 0.5 * torch.rand(1, 100, 1)
# cos_y = torch.cos(train_x[0] * (2 * math.pi)) + 0.5 * torch.rand(1, 100, 1)
# train_y = torch.cat((sin_y, sin_y_short, cos_y)).squeeze(-1)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=3)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_size=3), batch_size=3
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=3)
model = ExactGPModel(train_x, train_y, likelihood)
device = torch.device("cuda")
likelihood.to(device)
model.to(device)
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x.to(device))
    # Calc loss and backprop gradients
    loss = -mll(output, train_y.to(device)).sum()
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()



# Set into eval mode
model.eval()
likelihood.eval()
n = 20
test_x = torch.zeros(int(pow(n, 2)), 2)
for i in range(n):
    for j in range(n):
        test_x[i * n + j][0] = float(i) / (n-1)
        test_x[i * n + j][1] = float(j) / (n-1)
# Make predictions
with torch.no_grad():
    test_x = test_x.repeat(3, 1, 1)
    observed_pred = likelihood(model(test_x.to(device)))
    # Get mean
    mean = observed_pred.mean
    # Get lower and upper confidence bounds
    lower, upper = observed_pred.confidence_region()
