// implementation of the model f(Q) = Qe^(λQ₀/Q) to be constrained using SnIa
// neglects the effects of radiation
// due to the usage of the DAE solver, redshifts must be provided by asceding order
// for more information, see arXiv:2306.10176

functions {
  vector residual(real z, vector state, vector deriv, array[] real theta) {
    real Om = theta[1];
    real lambda  = theta[2];

    real E = state[1];
    real lhs = (E^2 - 2*lambda)*exp(lambda/E^2);  // left hand side of the equation for E
    real rhs = Om*(1+z)^3;                   // right hand side of the equation for E

    // compute the residuals
    vector[2] res;
    res[1] = rhs - lhs;       // E satisfies the algebraic relation, meaning that res[1] should be 0
    res[2] = 1/E - deriv[2];  // the derivative of S₂ is in fact 1/E, this should also be 0

    return res;
  }
}

data {
  int<lower=0> N1;
  array[N1] real zcmb;
  array[N1] real mb;
  array[N1] real dmb;
}

parameters {
  real<lower=0,upper=1> Om;
}

transformed parameters {
  // quantity derived from Om
  real lambda = 0.5 + lambert_w0( -Om/(2*exp(0.5)) );

  // pseudo-parameter array
  array[2] real theta = {Om, lambda};

  // compute dE/dz(z = 0), knowing that E(0) = 1
  real deriv_0 = 1/(2*exp(lambda)) * 3*Om/(1 - lambda + 2*lambda^2);

  array[N1] vector[2] S;
  S = dae(residual, [1, 0.0]', [deriv_0, 1]', 0.0, zcmb, theta);

  // compute Δ and use it to compute A, B and C for the marginalized likelihood for the SNIa
  real Delta;
  real A = 0;
  real B = 0;
  real C = 0;
  for (i in 1:N1) {
    Delta = mb[i] - 5.0*log10((1.0+zcmb[i]) * S[i,2]);
    A += (Delta/dmb[i])^2;
    B += Delta/dmb[i]^2;
    C += dmb[i]^(-2);
  }
}

model {
  // priors
  Om ~ normal(0.3, 10);

  // increment the log likelihood (aka target) by a given ammount
  target += -0.5*(A - B^2/C);
}
