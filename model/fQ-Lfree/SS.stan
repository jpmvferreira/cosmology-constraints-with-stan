// implementation of the model f(Q) = Qe^(λQ₀/Q) to be constrained using SS
// neglects the effects of radiation
// due to the usage of the DAE solver, redshifts must be provided by asceding order
// for more information, see arXiv:2306.10176

functions {
  // define the residual to be minimized by the differential-algebraic equation (DAE) solver
  // doc for the DAE: https://mc-stan.org/docs/functions-reference/functions-dae-solver.html
  vector residual(real z, vector state, vector deriv, array[] real theta) {
    // fetch relevant parameters and derived parameters
    real Om = theta[2];
    real lambda = theta[3];

    real E = state[1];
    real lhs = (E^2 - 2*lambda)*exp(lambda/E^2);  // lfs of first Friedmann equation
    real rhs = Om*(1+z)^3;                        // rhs of first Friedmann equation

    // compute the residuals
    vector[2] res;
    res[1] = rhs - lhs;       // E satisfies this algebraic relation, meaning that res[1] should be 0
    res[2] = 1/E - deriv[2];  // the derivative of S₂ is 1/E, this should also be 0

    return res;
  }
}

data {
  int<lower=0> N1;
  array[N1] real redshift;
  array[N1] real dLobs;
  array[N1] real error;
}

parameters {
  real<lower=0> h;
  real<upper=1.21306131942526684721> Om;  // constrain such that Ωm ≤ 2e^(-0.5)
}

transformed parameters {
  // constants
  real k = 2.99790041;  // c/H0 in Gpc

  // compute λ
  real lambda;
  lambda = 0.5 + lambert_w0( -Om/(2*exp(0.5)) );

  // compute dE/dz(z=0), knowing that E(z=0) = 1
  real deriv_0;
  deriv_0 = 1/(2*exp(lambda)) * 3*Om/(1 - lambda + 2*lambda^2);

  // first entry is S₁(z), corresponding to E(z)
  // second entry is S₂(z), corresponding to the integral of 1/E(x) from x=0 to x=z
  array[N1] vector[2] S;

  // call the differential algebraic equation solver
  S = dae(residual, [1, 0.0]', [deriv_0, 1]', 0.0, redshift, {h, Om, lambda});

  // compute theoretical luminosity distance for gravitational waves for each measured redshift
  real E;              // E(z)
  real intofEinv;      // the integral of 1/E(x) from x=0 to x=z
  real correction;     // correction from modification to gravity
  array[N1] real dGW;  // luminosity distance for gravitational waves
  for (i in 1:N1) {
    E = S[i,1];
    intofEinv = S[i,2];
    correction = ((1-lambda)/(1-lambda/E^2))^0.5 * exp((lambda/2) * (1 - 1/E^2));
    dGW[i] = correction * (1.0+redshift[i]) * (k/h) * intofEinv;
  }
}

model {
  // priors
  h ~ normal(0.68, 10);
  Om ~ normal(0.353, 10);

  // likelihood
  dLobs ~ normal(dGW, error);
}
