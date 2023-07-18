functions {
  // return E(z)
  real E(real z, array[] real theta) {
    real Om = theta[2];
    return sqrt(Om*(1+z)^3 + (1-Om));
  }

  // define 1/E(z) with the correct function signature to provide to 'integrate_1d'
  real Einverse(real z, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    return 1/E(z, theta);
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
  int<lower=0> N1;
  array[N1] real z;
  array[N1] real dvobs;
  array[N1] real error;
}

transformed data {
  // null arrays that are required for the integrate_1d function signature
  array[0] real x_r;
  array[0] int x_i;
}

parameters {
  real h;  // defined as H₀ = 100h km s⁻¹ Mpc⁻¹
  real Om;
}

transformed parameters {
  // constants
  real rf = 147.78;
  real c = 299792.458;  // speed of light in km/s

  // (pseudo-)parameter array
  array[2] real theta = {h, Om};

  // compute theoretical and observed BAO's
  real Dv;
  array[N1] real dv;
  for (i in 1:N1) {
    Dv = c/(100*h) * integrate_1d(Einverse, 0, z[i], theta, x_r, x_i)^(2./3) * (z[i]/E(z[i], theta))^(1./3);
    dv[i] = rf/rs(theta) * Dv;
  }
}

model {
  // priors
  h ~ normal(0.7, 3);
  Om ~ normal(0.3, 3);

  // likelihood
  dv ~ normal(dvobs, error);
}
