// implementation of the model f(Q) = Q + α*sqrt(Q) to be constrained using SS and SnIa
// uses SS + SnIa because SS itself are not enough to constrain this model
// neglects the effect of radiation
// for more information, see arXiv:2203.13788

functions{
  real E(real z, array[] real theta) {
    real Om = theta[2];
    return ( Om*(1+z)^3 + (1-Om) )^(0.5);
  }

  real integrand(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    return 1/E(x, theta);
  }
}

data {
  // SS observations
  int<lower=0> N1;
  array[N1] real redshift;
  array[N1] real dLobs;
  array[N1] real error;

  // SnIa observations
  int<lower=0> N2;
  array[N2] real zcmb;
  array[N2] real mb;
  array[N2] real dmb;
}

transformed data {
  array[0] real x_r;  // create null data values to match the signature required by integrate_1d
  array[0] int x_i;   // docs: https://mc-stan.org/docs/functions-reference/functions-1d-integrator.html
}

parameters {
  real h;
  real Om;
  real<lower=-2*6^0.5> alpha;  // avoid known singularity
}

transformed parameters {
  // parameter array
  array[3] real theta = {h, Om, alpha};

  // SS: compute the luminosity distance predicted by this model
  array[N1] real dL;
  real correction;
  for (i in 1:N1) {
    correction = ( (2*6^0.5 + alpha) / (2*6^0.5 + alpha/E(redshift[i], theta)) )^0.5;
    dL[i] = correction * (1.0 + redshift[i]) * (2.9979/h) * integrate_1d(integrand, 0, redshift[i], theta, x_r, x_i);
  }

  // SnIa: compute the magnitude for the marginalized likelihood
  real Delta;
  real A = 0;
  real B = 0;
  real C = 0;
  for (i in 1:N2) {
    Delta = mb[i] - 5.0*log10((1.0+zcmb[i]) * integrate_1d(integrand, 0, zcmb[i], theta, x_r, x_i));
    A += (Delta/dmb[i])^2;
    B += Delta/dmb[i]^2;
    C += dmb[i]^(-2);
  }
}

model {
  // priors
  h ~ normal(0.7, 10);
  Om ~ normal(0.284, 10);
  alpha ~ normal(0, 5);

  // likelihood for the SS
  dLobs ~ normal(dL, error);

  // likelihood for the SnIa
  // increment the log likelihood (aka target) manually
  target += -0.5*(A - B^2/C);
}
