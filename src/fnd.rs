use crate::{Grain, EulerAngles, UnitQuat, PolyGraph};
use rand::prelude::*;
use std::f64::consts::*;

const TWOPI: f64 = 6.283185307179586;
const INV_TWOPI: f64 = 1.0 / TWOPI;

fn delta_to_alpha(delta: f64) -> f64 {
    // max abs residual is around 3.38e-13 in case 100 approx points
    let (mut alpha, coefs) = if delta < 0.5 {
        (2.0448233504165804e-14, [
            1.7876780408089668,
            1.3957255727288031e-8,
            -1.1493775357161522,
            0.00003227140841001601,
            1.5734615835533645,
            0.009684809587727385,
            -2.6610699137381117,
            0.6615305028878575,
            1.2019327104448414,
            12.353681097537635,
            -41.01828197417929,
            58.220347090514046,
            -45.426005060530905,
            19.04083597215755,
            -3.3205182452784934,
        ])
    } else {
        (0.29654882195121857, [
            0.9596827151142042,
            -1.6942085152712263,
            11.14216293100142,
            -39.382022778552745,
            97.48178619662198,
            -161.35199891129258,
            152.9014328909631,
            4.841500855128998,
            -272.4729314032373,
            476.0752528483592,
            -472.81927326175037,
            304.03431682631447,
            -125.55179337654157,
            30.39869523141203,
            -3.2883547434257294,
        ])
    };

    let mut x = delta;
    for c in coefs {
        alpha += c * x;
        x *= delta;
    }
    alpha.clamp(0.0, FRAC_PI_2 - f64::EPSILON)
}

fn alpha_to_delta(alpha: f64) -> f64 {
    const FRAC_6_PI: f64 = 1.909859317102744;
    FRAC_6_PI * if alpha < FRAC_PI_4 { 
        alpha - (alpha.sin() * FRAC_1_SQRT_2).asin()
    } else { 
        let (ca, sa) = (alpha.cos(), alpha.sin());
        -FRAC_PI_3 + alpha + (ca / (1.0 + sa * sa).sqrt()).atan()
    }
}

fn lower_bnd_fund_cos_beta(alpha: f64) -> f64 {
    let fa = if alpha < FRAC_PI_4 { alpha.cos() } else { alpha.sin() };
    fa / (1.0 + fa * fa).sqrt()
}

fn cos_beta_length(alpha: f64) -> f64 {
    1.0 - lower_bnd_fund_cos_beta(alpha)
}

fn cos_beta_to_lambda(cos_beta: f64, alpha: f64) -> f64 {
    (1.0 - cos_beta) / cos_beta_length(alpha)
}

fn lambda_to_cos_beta(lambda: f64, alpha: f64) -> f64 {
    1.0 - cos_beta_length(alpha) * lambda
}

fn gamma_to_omega(gamma: f64) -> f64 {
    gamma * INV_TWOPI
}

fn omega_to_gamma(omega: f64) -> f64 {
    omega * TWOPI
}

fn random_alpha(rng: &mut impl Rng) -> f64 {
    let delta = FundAngles::random_angle(rng);
    delta_to_alpha(delta)
}

fn random_cos_beta(alpha: f64, rng: &mut impl Rng) -> f64 {
    let low = lower_bnd_fund_cos_beta(alpha);
    rng.gen_range(low..1.0)
}

fn random_gamma(rng: &mut impl Rng) -> f64 {
    rng.gen_range(0.0..PI*2.0)
}

pub fn random_euler_angles(rng: &mut impl Rng) -> EulerAngles {
    let alpha = random_alpha(rng);
    let cos_beta = random_cos_beta(alpha, rng);
    let gamma = random_gamma(rng);
    EulerAngles{ alpha, cos_beta, gamma }
}

pub fn euler_angles_inside(angs: EulerAngles) -> bool {
    let (a, cb) = (angs.alpha, angs.cos_beta);
    if a < FRAC_PI_4 {
        let ca = a.cos();
        cb >= ca / (1.0 + ca*ca).sqrt()
    } else if a < FRAC_PI_2 - f64::EPSILON {
        let sa = a.sin();
        cb >= sa / (1.0 + sa*sa).sqrt()
    } else {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundAngles {
    pub delta: f64,  // [0;1)
    pub lambda: f64, // [0;1)
    pub omega: f64,  // [0;1)
}

impl FundAngles {
    pub fn identity() -> Self {
        Self {
            delta: 0.0,
            lambda: 0.0,
            omega: 0.0,
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            delta: Self::random_angle(rng),
            lambda: Self::random_angle(rng),
            omega: Self::random_angle(rng),
        }
    }

    pub fn random_angle(rng: &mut impl Rng) -> f64 {
        rng.gen_range(0.0..1.0)
    }
}

impl From<EulerAngles> for FundAngles {
    fn from(angs: EulerAngles) -> Self {
        Self {
            delta: alpha_to_delta(angs.alpha),
            lambda: cos_beta_to_lambda(angs.cos_beta, angs.alpha),
            omega: gamma_to_omega(angs.gamma),
        }
    }
}

impl From<FundAngles> for EulerAngles {
    fn from(angs: FundAngles) -> Self {
        let alpha = delta_to_alpha(angs.delta);
        let cos_beta = lambda_to_cos_beta(angs.lambda, alpha);
        let gamma = omega_to_gamma(angs.omega);
        Self { alpha, cos_beta, gamma }
    }
}

impl From<UnitQuat> for FundAngles {
    fn from(o: UnitQuat) -> Self {
        EulerAngles::from(o).into()
    }
}

impl From<FundAngles> for UnitQuat {
    fn from(angs: FundAngles) -> Self {
        EulerAngles::from(angs).into()
    }
}

pub type Vec3<T> = Vec<Vec<Vec<T>>>;
pub type CellIdxs = (usize, usize, usize); 

#[derive(Debug, Clone)]
pub struct FundGrid {
    pub cells: Vec3<f64>,
    pub segms: (usize, usize, usize),
    pub dvol: f64,
}

impl FundGrid {
    pub fn new(delta_segms: usize, lambda_segms: usize, omega_segms: usize) -> Self {
        let cells = vec![vec![vec![0.0; omega_segms]; lambda_segms]; delta_segms];
        let dvol = 1.0 / (delta_segms * lambda_segms * omega_segms) as f64;
        Self{ cells, segms: (delta_segms, lambda_segms, omega_segms), dvol }
    }

    pub fn with_target_num_cells(n: usize) -> Self {
        let fos = (n as f64 / (4.0 * PI * PI)).powf(1.0 / 3.0);
        let os = fos.round() as usize;
        let ds = (fos * 2.0 * PI).round() as usize;
        let ls = ds;
        Self::new(ds, ls, os)
    }

    pub fn num_cells(&self) -> usize {
        self.segms.0 * self.segms.1 * self.segms.2
    }

    pub fn add(&mut self, g: &Grain) {
        let idxs = self.idxs(g.orientation.fund);
        *self.at_mut(idxs) += g.volume;
    }

    pub fn add_from_iter<'a>(&mut self, g_iter: impl Iterator<Item=&'a Grain>) {
        for g in g_iter {
            self.add(g);
        }
    }

    pub fn clear(&mut self) {
        for cell in self.cells.iter_mut().flatten().flatten() {
            *cell = 0.0;
        }
    }

    pub fn delta_idx(&self, angle: f64) -> usize {
        (angle * self.segms.0 as f64)
            .clamp(0.5, self.segms.0 as f64 - 0.5) as usize
    }
    
    pub fn lambda_idx(&self, angle: f64) -> usize {
        (angle * self.segms.1 as f64)
            .clamp(0.5, self.segms.1 as f64 - 0.5) as usize
    }

    pub fn omega_idx(&self, angle: f64) -> usize {
        (angle * self.segms.2 as f64)
            .clamp(0.5, self.segms.2 as f64 - 0.5) as usize
    }

    pub fn idxs(&self, angles: FundAngles) -> CellIdxs {
        (
            self.delta_idx(angles.delta), 
            self.lambda_idx(angles.lambda), 
            self.omega_idx(angles.omega)
        )
    }

    pub fn at(&self, idxs: CellIdxs) -> f64 {
        self.cells[idxs.0][idxs.1][idxs.2]
    }

    pub fn at_mut(&mut self, idxs: (usize, usize, usize)) -> &mut f64 {
        &mut self.cells[idxs.0][idxs.1][idxs.2]
    }

    pub fn normalize_grain_volumes(&self, g: &mut PolyGraph) {
        let inv_vol = 1.0 / (g.node_weights().map(|x| x.volume).sum::<f64>() * self.dvol);
        for Grain{ volume, .. } in g.node_weights_mut() {
            *volume *= inv_vol;
        }
    }
}