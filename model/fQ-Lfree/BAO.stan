// implementation of the model f(Q) = Qe^(λQ₀/Q) to be constrained using BAO
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

  // compute r_s
  real rs(array[] real theta) {
    real h = theta[1];
    real Om = theta[2];

    real wm = Om*h^2;
    real wb = 0.02226;      // baryonic density
    real wn = 0.0107*0.06;  // neutrinos sum m = 0.06 eV
    return 55.154 * exp(-72.3*(wn+0.0006)^2) / (wm^0.25351*wb^0.12807);
  }
}

data {
  // number of eevents, redshift, d_v and corresponding error
  int<lower=0> N1;
  array[N1] real z;
  array[N1] real dvobs;
  array[N1] real error;
}

parameters {
  real<lower=0> h;
  real<upper=1.21306131942526684721> Om;  // constrain such that Ωm ≤ 2e^(-0.5)
}

transformed parameters {
  // constants
  real rf = 147.78;
  real c = 299792.458;  // speed of light in km/s

  // compute λ
  real lambda;
  lambda = 0.5 + lambert_w0( -Om/(2*exp(0.5)) );

  // pseudo-parameter array
  array[3] real theta = {h, Om, lambda};

  // compute dE/dz(z=0), knowing that E(z=0) = 1
  real deriv_0;
  deriv_0 = 1/(2*exp(lambda)) * 3*Om/(1 - lambda + 2*lambda^2);

  // first entry is S₁(z), corresponding to E(z)
  // second entry is S₂(z), corresponding to the integral of 1/E(x) from x=0 to x=z
  array[N1] vector[2] S;

  // call the differential algebraic equation solver
  S = dae(residual, [1, 0.0]', [deriv_0, 1]', 0.0, z, theta);

  // compute theoretical and observed BAO's
  real E;              // E(z)
  real intofEinv;      // the integral of 1/E(x) from x=0 to x=z
  real Dv;
  array[N1] real dv;
  for (i in 1:N1) {
    E = S[i,1];
    intofEinv = S[i,2];
    Dv = c/(100*h) * intofEinv^(2./3) * (z[i]/E)^(1./3);
    dv[i] = rf/rs(theta) * Dv;
  }
}

model {
  // priors
  h ~ normal(0.7, 10);
  Om ~ normal(0.3, 10);

  // likelihood
  dv ~ normal(dvobs, error);
}
