// implementation of the model f(Q) = Qe^(λQ₀/Q) to be constrained using the CMB, via shift parameters
// uses E(z) approximated at high redshifts to compute some quantities to avoid more complicated numerical routines
// for more information, see arXiv:2306.10176

functions {
  // system to solve using the DAE
  // notation: y₁ ≡ 1/E(z), y₂ ≡ I(z) ≡ integral of 1/E(x) from 0 to z
  vector daesystem(real z, vector y, vector dy, array[] real theta) {
    real Ob = theta[2];
    real Oc = theta[3];
    real Or = theta[4];
    real lambda = theta[5];
    real zstar = theta[6];

    real E = y[1];
    real dI = dy[2];

    // change of variable
    real x = zstar*z;

    // compute the residuals
    vector[2] res;

    real lhs = (E^2 - 2*lambda)*exp(lambda/E^2);               // lhs of modified 1st friedmann equation
    real rhs = (Ob + Oc)*(1+x)^3 + Or*(1+x)^4;  // rhs of modified 1st friedmann equation
    res[1] = rhs - lhs;

    res[2] = zstar/E - dI;

    return res;
  }

  // return f(z)
  real f(real z, array[] real theta) {
   real h = theta[1];
   real Ob = theta[2];

   real Tcmb = 2.7255;  // Zhai2018
   real Rb = 31500*Ob*(h^2)*((2.7/Tcmb)^4);

   return (3*(1+Rb/(1+z)))^0.5;
  }

  // E(z) approximated for high redshifts
  real Ehighredshifts(real z, array[] real theta) {
    real Ob = theta[2];
    real Oc = theta[3];
    real Or = theta[4];
    real lambda = theta[5];

    real fz = (Ob + Oc)*(1+z)^3 + Or*(1+z)^4;
    real E = sqrt( (fz + lambda + sqrt(fz^2 + 2*fz*lambda + 9*lambda^2)) / 2);

    return E;
  }

  // return 1/[E(z)*f(z)]
  real integrand(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    return 1/(Ehighredshifts(x, theta)*f(x, theta));
  }
}

data {
  real R_exp;
  real R_error;
  real wb_exp;
  real wb_error;
  real la_exp;
  real la_error;
}

transformed data {
  array[0] real x_r;  // dummy variables required because of integrate_1d signature
  array[0] int x_i;   // see: https://mc-stan.org/docs/functions-reference/functions-1d-integrator.html

  // inverse of the covariance matrix
  // the covariance matrix was taken from the Planck 2018 TT+lowE release
  matrix[3,3] Cinv;
  Cinv[1,1] = 70.41272413  ; Cinv[1,2] = -472.33176518  ; Cinv[1,3] = 4150.71131541;
  Cinv[2,1] = -472.33176518; Cinv[2,2] = 31027.46200904 ; Cinv[2,3] = 477813.59257032;
  Cinv[3,1] = 4150.71131541; Cinv[3,2] = 477813.59257032; Cinv[3,3] = 30083780.34236922;
}

parameters {
  // lower bound for physical reasons
  real<lower=0> h;

  // no bounds to allow for extra freedom
  real Ob;
  real Oc;
}

transformed parameters {
  // Ωr is a derived parameter from ωr
  // Ωr ≤ 2e^(-0.5) - Ωm, otherwise Lambert W function has no solutions
  // no lower bound because it's expected to be close to 0
  real wr = 4.15*10^(-5);
  real<upper=1.21306131942526684721-Ob-Oc> Or = wr/h^2;

  // compute lambda
  real lambda = 0.5 + lambert_w0( -(Ob + Oc + Or)/(2*exp(0.5)) );

  // compute zstar
  real g1 = 0.0783*(Ob*h^2)^(-0.238) / (1 + 39.5*(Ob*h^2)^0.763);
  real g2 = 0.560 / (1 + 21.1*(Ob*h^2)^1.81);
  real zstar = 1048 * (1 + 0.00124*(Ob*h^2)^(-0.738)) * (1+g1*((Ob + Oc)*h^2)^g2);

  // (pseudo-)parameter array
  array[6] real faketheta = {h, Ob, Oc, Or, lambda, zstar};

  // initial conditions
  // the value of E(z), I(z) and first derivatives at z = 0
  real E = 1.0;
  real dE = 1/(2*exp(lambda)) * (3*(Ob + Oc) + 4*Or)/(1 - lambda + 2*lambda^2);
  real I = 0.0;
  real dI = 1.0;

  // call the differential algebraic equation solver
  // because of the change of variable, we want the solution at time = 1, which corresponds to z = z*
  array[1] vector[2] S;
  S = dae(daesystem, [E, I]', [dE, dI]', 0.0, {1}, faketheta);

  // compute the integral of 1/(E(x)*f(x)) from zstar to ∞, using E(z) approximated at high redshifts
  real sol = integrate_1d(integrand, zstar, positive_infinity(), faketheta, x_r, x_i);

  // compute the shift parameters
  real la = pi()*S[1][2]/sol;
  real R = sqrt(Ob + Oc)*S[1][2];
  real wb = Ob*h^2;

  // difference vector between the model predictions and the observations
  vector[3] x = [la - la_exp, R - R_exp, wb - wb_exp]';
}

model {
  // priors
  h ~ normal(0.7, 3);
  Ob ~ normal(0.05, 3);
  Oc ~ normal(0.25, 3);

  // likelihood
  target += -x'*Cinv*x;
}
