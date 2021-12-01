use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rand::prelude::*;
use rand::distributions::Uniform as RandUniform;
use rand_pcg::Pcg64;
use nalgebra as na;
use na::{Quaternion, UnitQuaternion, Vector3};
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::f64::consts::*;
use std::fs::File;
use std::io::Write;
use std::panic;
use std::time::Instant;

type UnitQuat = UnitQuaternion<f64>;
use fnd::FundAngles;

use crate::ori_opt::{OptResult, Rotator};
#[derive(Clone, Copy, Debug)]
pub struct GrainOrientation {
    pub quat: UnitQuat, 
    pub fund: FundAngles,
}

impl GrainOrientation {
    pub fn new(quat: UnitQuat, fund: FundAngles) -> Self {
        Self{ quat, fund }
    }

    pub fn identity() -> Self {
        Self{ 
            quat: UnitQuat::identity(), 
            fund: FundAngles::identity(), 
        }
    }

    pub fn random(rng: &mut impl Rng) -> Self {
        let fund = FundAngles::random(rng);
        let quat = fund.into();
        Self{ quat, fund }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Grain {
    pub orientation: GrainOrientation,
    pub volume: f64,
}

impl Grain {
    pub fn new(orientation: GrainOrientation, volume: f64) -> Self {
        Self{ orientation, volume }
    }

    pub fn identity_oriented(volume: f64) -> Self {
        Self{ 
            orientation: GrainOrientation::identity(),
            volume,
        }
    }

    pub fn randomly_oriented(volume: f64, rng: &mut impl Rng) -> Self {
        Self{ orientation: GrainOrientation::random(rng), volume }
    }
}

type PolyGraph = UnGraph<Grain, AngleArea>;

#[derive(Clone, Copy, Debug)]
pub struct AngleArea {
    pub angle: f64,
    pub area: f64,
}

fn build_graph(bnds: Vec<(f64, u32, u32)>, volumes: Vec<f64>) -> PolyGraph {    
    let num_nodes = volumes.len();
    let num_edges = bnds.len();
    
    let mut g = PolyGraph::with_capacity(num_nodes, num_edges);
    for i in 0..num_nodes {
        g.add_node(Grain::identity_oriented(volumes[i]));
    }
    for e in bnds {
        g.add_edge(e.1.into(), e.2.into(), AngleArea{ angle: f64::NAN, area: e.0 });
    }
    
    g
}

fn parse_bnds(path: &str) -> Vec<(f64, u32, u32)> {
    std::fs::read_to_string(path).unwrap()
        .lines()
        .map(|x| x.split_whitespace()
                  .collect::<Vec<_>>())
        .filter(|x| x.len() == 3)
        .map(|x| (
            x[0].parse::<f64>().unwrap(), 
            x[1].parse::<u32>().unwrap() - 1, 
            x[2].parse::<u32>().unwrap() - 1,
        ))
        .collect()
}

fn parse_volumes(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path).unwrap()
        .lines()
        .map(|x| x.trim().parse::<f64>().unwrap())
        .collect()
}

fn count_volumes_from_bnds(bnds: &[(f64, u32, u32)]) -> usize {
    bnds.iter().map(|x| x.1.max(x.2) as usize).max().unwrap() + 1
}

fn parse_graph(bnds_path: &str, volumes_path: &str) -> PolyGraph {
    let bnds = parse_bnds(bnds_path);
    let volumes = parse_volumes(volumes_path);
    build_graph(bnds, volumes)
}

fn write_orientations_quat(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    for w in g.node_weights() {
        let q = &w.orientation.quat;
        writeln!(&mut file, "{} {} {} {}", q.w, q.i, q.j, q.k).unwrap();
    }
}

fn write_orientations_mtex_euler(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    for w in g.node_weights() {
        // inversed quaternion is used because 
        // mtex defines orientations in a slightly different way 
        // than they have been defined by Bunge.
        // see more in MTEX article 'MTEX vs. Bunge Convention'
        let angs = EulerAngles::from(w.orientation.quat.inverse());
        let tmp = w.orientation.fund.into();
        if !fnd::euler_angles_inside(tmp) {
            println!("da fock: {:?}", tmp);
        }
        writeln!(&mut file, "{} {} {}", angs.alpha, angs.cos_beta.acos(), angs.gamma).unwrap();
    }
}

fn write_random_orientations_euler(num_oris: usize, path: &str, rng: &mut impl Rng) {
    let mut file = File::create(path).unwrap();
    for angs in (0..num_oris).map(|_| EulerAngles::random(rng)) {
        writeln!(&mut file, "{} {} {}", angs.gamma, angs.cos_beta.acos(), angs.alpha).unwrap();
    }
}

fn set_random_orientations(g: &mut PolyGraph, rng: &mut impl Rng) {
    for w in g.node_weights_mut() {
        w.orientation = GrainOrientation::random(rng);
    }
}

fn cube_rotational_symmetry() -> Vec<UnitQuat> {
    let dirs = vec![
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(-1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
        Vector3::new(0.0, 0.0, -1.0),
    ];

    let mut xy_ids = Vec::with_capacity(24);
    for i in 0..6 {
        let neg = i + 1 - 2 * (i % 2);
        for j in 0..6 {
            if j != i && j != neg {
                xy_ids.push((i, j));
            }
        }
    }

    xy_ids.into_iter()
        .map(|(x_id, y_id)| {
            let x = dirs[x_id];
            let y = dirs[y_id];
            let z = x.cross(&y);
            UnitQuat::from_basis_unchecked(&[x, y, z])
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Histogram {
    pub beg: f64,
    pub end: f64,
    pub heights: Vec<f64>,
    pub bar_len: f64,
    find_bar_coef: f64,
}

impl Histogram {
    pub fn new(beg: f64, end: f64, bars: usize) -> Self {
        let hs = vec![0.0; bars];
        Histogram { 
            beg, end, heights: hs, 
            bar_len: (end - beg) / bars as f64,
            find_bar_coef: bars as f64 / (end - beg),
        }
    }

    pub fn add(&mut self, aa: AngleArea) {
        let idx = self.idx(aa.angle);
        self.heights[idx] += aa.area;
    }

    pub fn add_from_slice(&mut self, aa_slice: &[AngleArea]) {
        for &aa in aa_slice {
            self.add(aa);
        }
    }

    pub fn bars(&self) -> usize {
        self.heights.len()
    }

    pub fn total_height(&self) -> f64 {
        self.heights.iter().sum()
    }

    pub fn area(&self) -> f64 {
        self.total_height() * self.bar_len
    }

    pub fn idx(&self, angle: f64) -> usize {
        let i = ((angle - self.beg) * self.find_bar_coef) as usize;
        i.min(self.bars() - 1)
    }

    pub fn ratiolize_mut(&mut self) -> &mut Self {
        let inv_total = 1.0 / self.total_height();
        for h in self.heights.iter_mut() {
            *h *= inv_total;
        }
        self
    }

    pub fn ratiolize(&self) -> Self {
        let mut hist = self.clone();
        hist.ratiolize_mut();
        hist
    }

    pub fn normalize_mut(&mut self) -> &mut Self {
        let inv_area = 1.0 / self.area();
        for h in self.heights.iter_mut() {
            *h *= inv_area;
        }
        self
    }

    pub fn normalize(&self) -> Self {
        let mut hist = self.clone();
        hist.normalize_mut();
        hist
    }

    pub fn normalize_grain_boundary_area(&self, g: &mut PolyGraph) {
        let inv_area = 1.0 / (g.edge_weights().map(|x| x.area).sum::<f64>() * self.bar_len);
        for AngleArea{ area, .. } in g.edge_weights_mut() {
            *area *= inv_area;
        }
    }

    fn update_with_edge_new_angle(&mut self, new_aa: AngleArea, prev_angle: f64) {
        let hpos = self.idx(new_aa.angle);
        let prev_hpos = self.idx(prev_angle);
        if hpos != prev_hpos {
            self.heights[prev_hpos] -= new_aa.area;
            self.heights[hpos] += new_aa.area;
        }
    }

    pub fn update_with_grain_new_angles(
        &mut self, g: &PolyGraph, 
        n: NodeIndex, prev_angles: &Vec<f64>
    ) -> Histogram {
        let prev_hist = self.clone();
        for (e, &pa) in g.edges(n).zip(prev_angles) {
            self.update_with_edge_new_angle(*e.weight(), pa);
        }

        prev_hist
    }

    pub fn update_with_grain_new_angles_noret(
        &mut self, g: &PolyGraph, 
        n: NodeIndex, prev_angles: &Vec<f64>
    ) {
        for (e, &pa) in g.edges(n).zip(prev_angles) {
            self.update_with_edge_new_angle(*e.weight(), pa);
        }
    }

    pub fn update_with_2grains_new_angles(
        &mut self, g: &PolyGraph, 
        n1: NodeIndex, n2: NodeIndex, 
        prev_angles1: &Vec<f64>, prev_angles2: &Vec<f64>,
    ) -> Histogram {
        let prev_hist = self.clone();
        for (e, &pa) in g.edges(n1).zip(prev_angles1) {
            self.update_with_edge_new_angle(*e.weight(), pa);
        }
        for (e, &pa) in g.edges(n2).zip(prev_angles2) {
            // when using petgraph v0.6.0 source is always n2 even when grapth is undirected
            if e.target() == n1 {
                continue;
            }
            // more implementation stable version, doesn't require source to be always n2
            // if e.source() == n2 && e.target() == n1 ||
            //    e.source() == n1 && e.target() == n2 {
            //     continue;
            // }
            self.update_with_edge_new_angle(*e.weight(), pa);
        }

        prev_hist
    }

    pub fn pairs(&self) -> impl Iterator<Item=(f64, f64)> {
        let d = self.bar_len;
        let first = self.beg + d * 0.5;
        self.heights.clone().into_iter()
            .enumerate()
            .map(move |(i, h)| (first + i as f64 * d, h))
    }
}

pub mod fnd {
    use crate::{Grain, EulerAngles, UnitQuat, PolyGraph};
    use rand::prelude::*;
    use std::f64::consts::*;

    const TWOPI: f64 = 6.283185307179586;
    const INV_TWOPI: f64 = 1.0 / TWOPI;

    // fn delta_to_alpha_rough(delta: f64) -> f64 {
    //     let coefs = [
    //         1.0616904724669767, 
    //         0.5304936828043207, 
    //         -1.6125716869506046, 
    //         1.3248915389845322, 
    //         -0.33738085998446576,
    //     ];
    //     let mut alpha = 0.002259301224771941;
    //     let mut x = delta;
    //     for c in coefs {
    //         alpha += c * x;
    //         x *= delta;
    //     }
    //     alpha.clamp(0.0, FRAC_PI_2 - f64::EPSILON)
    // }

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
        pub segms: usize,
        pub dvol: f64,
        //...
    }

    impl FundGrid {
        pub fn new(each_angle_segms: usize) -> Self {
            let segms = each_angle_segms;
            let cells = vec![vec![vec![0.0; segms]; segms]; segms];
            let dvol = 1.0 / segms.pow(3) as f64;
            Self{ cells, segms, dvol }
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

        pub fn idx(&self, angle: f64) -> usize {
            (angle * self.segms as f64)
                .clamp(0.5, self.segms as f64 - 0.5) as usize
        }

        pub fn idxs(&self, angles: FundAngles) -> CellIdxs {
            (
                self.idx(angles.delta), 
                self.idx(angles.lambda), 
                self.idx(angles.omega)
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

        //...
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerAngles {
    pub alpha: f64,    // [0;2*Pi)
    pub cos_beta: f64, // [0;  Pi)
    pub gamma: f64,    // [0;2*Pi)
}

impl EulerAngles {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self {
            alpha: rng.gen_range(0.0..PI*2.0),
            cos_beta: rng.gen_range(-1.0..1.0),
            gamma: rng.gen_range(0.0..PI*2.0),
        }
    }
}

impl From<UnitQuat> for EulerAngles {
    fn from(o: UnitQuat) -> Self {
        let cos_beta = o.w*o.w - o.i*o.i - o.j*o.j + o.k*o.k;
        if cos_beta.abs() >= 1.0 - f64::EPSILON {
            let om11 = o.w*o.w + o.i*o.i - o.j*o.j - o.k*o.k;
            let hom21 = o.i * o.j + o.w * o.k;
            let g = if hom21 < 0.0 { 2.0 * PI - om11.acos() } else { om11.acos() };
            let alpha = if g >= 2.0 * PI { g - 2.0 * PI } else { g };
            Self{ alpha, cos_beta, gamma: 0.0 }
        } else {
            let mut alpha = (o.w * o.j + o.i * o.k).atan2(o.w * o.i - o.j * o.k);
            if alpha <= -f64::EPSILON {
                alpha += 2.0 * PI;
            }
            let mut gamma = (o.i * o.k - o.w * o.j).atan2(o.w * o.i + o.j * o.k);
            if gamma <= -f64::EPSILON {
                gamma += 2.0 * PI;
            }
            Self{ alpha, cos_beta, gamma }
        }
    }
}

impl From<EulerAngles> for UnitQuat {
    fn from(angs: EulerAngles) -> Self {
        let factor_plus_b = (0.5 + 0.5 * angs.cos_beta).sqrt();
        let factor_minus_b = (0.5 - 0.5 * angs.cos_beta).sqrt();
        let half_sum_a_g = (angs.alpha + angs.gamma) * 0.5;
        let half_diff_a_g = (angs.alpha - angs.gamma) * 0.5;
        let q = Quaternion::new(
            factor_plus_b * half_sum_a_g.cos(), 
            factor_minus_b * half_diff_a_g.cos(), 
            factor_minus_b * half_diff_a_g.sin(), 
            factor_plus_b * half_sum_a_g.sin(), 
        );
        UnitQuat::new_unchecked(q)
    }
}

mod mis_opt {
    use crate::*;

    fn misorientation_angle(
        o1: UnitQuat, o2: UnitQuat, 
        syms: &Vec<UnitQuat>
    ) -> f64 {
        let r = o1.rotation_to(&o2);
        syms.iter()
            .map(|s| (s.scalar() * r.scalar() - s.imag().dot(&r.imag())).abs())
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            .acos() * 2.0
    
        //  simplified Ostapovich version
        // let r = o1.rotation_to(&o2);
        // syms.iter()
        //     .map(|s| (s.scalar() * r.scalar() - s.imag().dot(&r.imag())).abs().acos())
        //     .min_by(|x, y| x.partial_cmp(y).unwrap())
        //     .unwrap() * 2.0
    
        //  most simple and inefficient
        // syms.iter()
        //     .map(|s| o1.angle_to(&(s * o2)))
        //     .min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
    
        //  using nalgebra funcs
        // let inv_r = o1.rotation_to(&o2).inverse();
        // syms.iter()
        //     .map(|s| inv_r.angle_to(s))
        //     .min_by(|x, y| x.partial_cmp(y).unwrap())
        //     .unwrap()
    }
    
    fn update_angle(
        g: &mut PolyGraph, e: EdgeIndex, syms: &Vec<UnitQuat>
    ) -> f64 {
        let (n1, n2) = g.edge_endpoints(e).unwrap();
        let (o1, o2) = (g[n1].orientation.quat, g[n2].orientation.quat);
        let prev_angle = g[e].angle;
        g[e].angle = misorientation_angle(o1, o2, syms);
        prev_angle
    }
    
    fn update_grain_angles(
        g: &mut PolyGraph, n: NodeIndex, syms: &Vec<UnitQuat>
    ) -> Vec<f64> {
        let edges: Vec<_> = g.edges(n).map(|e| e.id()).collect();
        let mut prev_angles = Vec::with_capacity(edges.len());
        for e in edges {
            prev_angles.push(update_angle(g, e, syms));
        }
        prev_angles
    }
    
    fn update_grain_angles_noret(
        g: &mut PolyGraph, n: NodeIndex, syms: &Vec<UnitQuat>
    ) {
        let edges: Vec<_> = g.edges(n).map(|e| e.id()).collect();
        for e in edges {
            update_angle(g, e, syms);
        }
    }
    
    fn restore_grain_angles(g: &mut PolyGraph, n: NodeIndex, prev_angles: Vec<f64>) {
        let edges: Vec<_> = g.edges(n).map(|e| e.id()).collect();
        for (&e, a) in edges.iter().zip(prev_angles) {
            g[e].angle = a;
        }
    }
    
    pub fn update_angles(g: &mut PolyGraph, syms: &Vec<UnitQuat>) {
        for e in g.edge_indices() {
            update_angle(g, e, syms);
        }
    }
    
    pub fn angle_area_vec(g: &PolyGraph) -> Vec<AngleArea> {
        g.edge_weights().map(|&e| e).collect()
    }
    
    pub fn max_gap(sorted_pairs: &Vec<AngleArea>) -> f64 {
        let mut max_gap = 0.0;
        let mut prev_angle = sorted_pairs.first().unwrap().angle;
        for &AngleArea{ angle, .. } in sorted_pairs.iter().skip(1) {
            let gap = angle - prev_angle;
            max_gap = gap.max(max_gap);
            prev_angle = angle;
        }
    
        max_gap
    }

    pub fn diff_norm(hist: &Histogram, f: impl Fn(f64) -> f64) -> f64 {
        hist.pairs()
            .map(|(a, d)| {
                let fa = f(a);
                ((fa - d) / (fa + d)).powi(2)
            })
            .sum::<f64>().sqrt()
    
            // .map(|(a, d)| {
            //     let fa = f(a);
            //     ((fa - d) / (1.0 + fa + d)).powi(2)
            // })
            // .sum::<f64>().sqrt()
    
            // .map(|(a, d)| {
            //     let fa = f(a);
            //     ((fa - d) / (1.0 + fa + d)).abs()
            //     // ((fa - d) / (fa + d)).abs()
            //     // (fa - d).abs()
            // })
            // .max_by(|&x, &y| x.partial_cmp(&y).unwrap())
            // .unwrap()
    }
    
    pub fn swap_ori(g: &mut PolyGraph, n1: NodeIndex, n2: NodeIndex) {
        let gn1_ori = g[n1].orientation;
        g[n1].orientation = g[n2].orientation;
        g[n2].orientation = gn1_ori;
    }
    
    pub fn iterate_swaps(
        g: &mut PolyGraph, hist: &mut Histogram, syms: &Vec<UnitQuat>,
        rng: &mut impl Rng, f: impl Fn(f64) -> f64
    ) -> Option<f64> {
    
        let distr = RandUniform::new(0, g.node_count() as u32);
        let n1: NodeIndex = rng.sample(distr).into();
        let n2: NodeIndex = loop {
            let n: NodeIndex = rng.sample(distr).into();
            if n != n1 {
                break n;
            }
        };
        
        swap_ori(g, n1, n2);
        let prev_angles1 = update_grain_angles(g, n1, syms);
        let prev_angles2 = update_grain_angles(g, n2, syms);
        let prev_hist = hist.update_with_2grains_new_angles(
            g, n1, n2, &prev_angles1, &prev_angles2
        );
    
        let prev_dnorm = diff_norm(&prev_hist, |x| f(x));
        let dnorm = diff_norm(hist, f);
        if dnorm < prev_dnorm {
            Some(dnorm)
        } else {
            *hist = prev_hist;
            swap_ori(g, n1, n2);
            restore_grain_angles(g, n1, prev_angles1);
            restore_grain_angles(g, n2, prev_angles2);
            None
        }
    }
    
    pub fn iterate_rotations(
        g: &mut PolyGraph, hist: &mut Histogram, syms: &Vec<UnitQuat>,
        rng: &mut impl Rng, f: impl Fn(f64) -> f64
    ) -> Option<f64> {
    
        const MAX_ROTS: usize = 4;
    
        let distr = RandUniform::new(0, g.node_count() as u32);
        let n: NodeIndex = rng.sample(distr).into();
        
        let prev_ori = g[n].orientation;
        g[n].orientation = GrainOrientation::random(rng);
        let prev_angles = update_grain_angles(g, n, syms);
        let prev_hist = hist.update_with_grain_new_angles(g, n, &prev_angles);
    
        let prev_dnorm = diff_norm(&prev_hist, |x| f(x));
        let dnorm = diff_norm(hist, |x| f(x));
        if dnorm < prev_dnorm {
            Some(dnorm)
        } else {
            for _ in 0..MAX_ROTS - 1 {
                g[n].orientation = GrainOrientation::random(rng);
                let prev_angles = update_grain_angles(g, n, syms);
                hist.update_with_grain_new_angles_noret(g, n, &prev_angles);
                let dnorm = diff_norm(hist, |x| f(x));
                if dnorm < prev_dnorm {
                    return Some(dnorm);
                }
            }
    
            *hist = prev_hist;
            g[n].orientation = prev_ori;
            restore_grain_angles(g, n, prev_angles);
            None
        }
    }
}

mod ori_opt {
    use crate::*;

    pub fn texture_sum(grid: &fnd::FundGrid) -> f64 {
        grid.cells.iter().flatten().flatten()
            .map(|&x| x * x)
            .sum()
    }

    pub fn texture_index(grid: &fnd::FundGrid) -> f64 {
        texture_sum(grid) * grid.dvol
    }

    // returns previus and current orientation cell idxs and cell heights
    // fn rotate_randomly(
    //     g: &mut PolyGraph, grid: &mut fnd::FundGrid, 
    //     n: NodeIndex, texture_sum: &mut f64, rng: &mut impl Rng
    // ) -> ((fnd::CellIdxs, f64), (fnd::CellIdxs, f64)) {
    //     let vol = g[n].volume;

    //     let (prev_idxs, prev_h1) = {
    //         let idxs = grid.idxs(g[n].orientation.fund);
    //         let prev_h = grid.at(idxs);
    //         *grid.at_mut(idxs) -= vol;
    //         (idxs, prev_h)
    //     };
    //     let (cur_idxs, prev_h2) = {
    //         g[n].orientation = GrainOrientation::random(rng);
    //         let mut idxs = grid.idxs(g[n].orientation.fund);
    //         while idxs == prev_idxs {
    //             g[n].orientation = GrainOrientation::random(rng);
    //             idxs = grid.idxs(g[n].orientation.fund);
    //         }
    //         let prev_h = grid.at(idxs);
    //         *grid.at_mut(idxs) += vol;
    //         (idxs, prev_h)
    //     };
    //     *texture_sum += 2.0 * vol * ((prev_h2 - prev_h1) + vol);
    //     ((prev_idxs, prev_h1), (cur_idxs, prev_h2))
    // }
    
    // pub fn iterate_rotations_cubic_isotropic(
    //     g: &mut PolyGraph, grid: &mut fnd::FundGrid, 
    //     texture_sum: &mut f64, rng: &mut impl Rng,
    // ) -> Option<f64> {

    //     let distr = RandUniform::new(0, g.node_count() as u32);
    //     let n: NodeIndex = rng.sample(distr).into();
        
    //     let prev_ori = g[n].orientation;
    //     let prev_texsum = *texture_sum;
    //     let (prev, cur) = rotate_randomly(g, grid, n, texture_sum, rng);
    //     if *texture_sum < prev_texsum {
    //         Some(*texture_sum * grid.dvol)
    //     } else {
    //         g[n].orientation = prev_ori;
    //         // restoration order matters in case 
    //         // the new orientation is in the same cell as the previous one
    //         *grid.at_mut(cur.0) = cur.1;
    //         *grid.at_mut(prev.0) = prev.1;
    //         *texture_sum = prev_texsum;
    //         None
    //     }
    // }

    #[derive(Debug, Clone, Copy)]
    struct CellBackup {
        idxs: fnd::CellIdxs,
        height: f64,
    }

    #[derive(Debug, Clone)]
    pub struct RotatorBackup {
        grain_idx: NodeIndex, 
        texture_sum: f64,
        prev_ori: GrainOrientation,
        prev_cell_bu: CellBackup,
        cur_cell_bu: CellBackup,
    }

    pub enum OptResult {
        MoreOptimal(f64),
        SameOrLessOptimal(f64),
    }

    #[derive(Debug, Clone)]
    pub struct Rotator {
        backup: Option<RotatorBackup>,
        texture_sum: f64,
    }

    impl Rotator {
        pub fn new(grid: &fnd::FundGrid) -> Self {
            Self{ backup: None, texture_sum: texture_sum(grid) }
        }

        pub fn texture_index(&self, grid: &fnd::FundGrid) -> f64 {
            self.texture_sum * grid.dvol
        }

        pub fn rotate_grain_ori(
            &mut self, grain_idx: NodeIndex, g: &mut PolyGraph, 
            grid: &mut fnd::FundGrid, rng: &mut impl Rng,
        ) -> OptResult {
            
            let vol = g[grain_idx].volume;
            let prev_texsum = self.texture_sum;

            let prev_ori = g[grain_idx].orientation;
            let prev_idxs = grid.idxs(prev_ori.fund);
            let prev_h1 = grid.at(prev_idxs);
            *grid.at_mut(prev_idxs) -= vol;
            let prev_cell_bu = CellBackup{ idxs: prev_idxs, height: prev_h1 };

            let mut cur_ori = GrainOrientation::random(rng);
            let mut cur_idxs = grid.idxs(cur_ori.fund);
            while cur_idxs == prev_idxs {
                cur_ori = GrainOrientation::random(rng);
                cur_idxs = grid.idxs(cur_ori.fund);
            }
            g[grain_idx].orientation = cur_ori;
            let prev_h2 = grid.at(cur_idxs);
            *grid.at_mut(cur_idxs) += vol;
            let cur_cell_bu = CellBackup{ idxs: cur_idxs, height: prev_h2 };

            self.texture_sum += 2.0 * vol * ((prev_h2 - prev_h1) + vol);
            
            let backup = RotatorBackup{ 
                grain_idx, 
                texture_sum: prev_texsum,
                prev_ori, 
                prev_cell_bu, 
                cur_cell_bu,
            };
            self.backup = Some(backup);

            let texidx = self.texture_sum * grid.dvol;
            if self.texture_sum < prev_texsum {
                OptResult::MoreOptimal(texidx)
            } else {
                OptResult::SameOrLessOptimal(texidx)
            }
        }

        pub fn rotate_random_grain_ori(
            &mut self, g: &mut PolyGraph, grid: &mut fnd::FundGrid, 
            rng: &mut impl Rng,
        ) -> OptResult {

            let distr = RandUniform::new(0, g.node_count() as u32);
            let grain_idx: NodeIndex = rng.sample(distr).into();
            self.rotate_grain_ori(grain_idx, g, grid, rng)
        }

        pub fn restore_previous(&mut self, g: &mut PolyGraph, grid: &mut fnd::FundGrid) {
            let RotatorBackup{ 
                grain_idx, 
                prev_ori, 
                prev_cell_bu, 
                cur_cell_bu, 
                texture_sum 
            } = self.backup.take().unwrap();

            g[grain_idx].orientation = prev_ori;
            // restoration order matters in case 
            // the new orientation is in the same cell as the previous one
            *grid.at_mut(cur_cell_bu.idxs) = cur_cell_bu.height;
            *grid.at_mut(prev_cell_bu.idxs) = prev_cell_bu.height;
            self.texture_sum = texture_sum;
        }
    }

    
}

fn rotate_to_fund_domain(o: UnitQuat, syms: &Vec<UnitQuat>) -> UnitQuat {
    for s in syms {
        let q = s * o;
        let angs = EulerAngles::from(q);
        if fnd::euler_angles_inside(angs) {
            return q;
        }
    }
    panic!("failed to rotate to fundamental domain")
}

fn main1() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();

    let (alpha, cos_beta, gamma) = (FRAC_PI_2, 1.0 - 1e-16, FRAC_PI_3 * 2.0);
    let angs = EulerAngles{ alpha, cos_beta, gamma };
    let q = UnitQuat::from(angs);
    let back_angs = EulerAngles::from(q);
    dbg!(angs);
    dbg!(back_angs);
    println!("alpha+gamma: {}", alpha + gamma);
}

fn main3() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();
    for basis in &syms {
        println!("{:?}", basis);
    }
    println!("syms {}", syms.len());
    mis_opt::update_angles(&mut g, &syms);

    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = Histogram::new(hist_beg, hist_end, 30);
    hist.normalize_grain_boundary_area(&mut g);
    println!(
        "grains boundary area mul by bar len {}", 
        g.edge_weights().map(|e| e.area).sum::<f64>() * hist.bar_len
    );
    let aa = mis_opt::angle_area_vec(&g);
    hist.add_from_slice(&aa);
    // let mut file = File::create("hist.txt").unwrap();
    // for (angle, height) in hist.pairs() {
    //     println!("{} {}", angle.to_degrees(), height);
    //     writeln!(&mut file, "{}\t{}", angle.to_degrees(), height).unwrap();
    // }

    let uni = StatUniform::new(hist_beg, hist_end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();

    let lognorn_stop = 0.0856;
    
    // let now = Instant::now();
    // for i in 0..3_000_000 {
    //     if let Some(dnorm) = iterate_swaps(
    //         &mut g, &mut hist, &syms, &mut rng, |x| lognorm.pdf(x)
    //     ) {
    //         // println!("iter {}, norm {}", i, dnorm);
    //         if i >= 1_000_000 {
    //             println!("iter {}, norm {}", i, dnorm);
    //             break
    //         }
    //         // if dnorm < lognorn_stop {
    //         //     break
    //         // }
    //     }
    // }
    // println!(
    //     "swaps alg time:        {}, norm {}", 
    //     now.elapsed().as_secs_f64(), diff_norm(&mut hist, |x| lognorm.pdf(x))
    // );

    let now = Instant::now();
    for i in 0..3_000_000 {
        if let Some(dnorm) = mis_opt::iterate_rotations(
            &mut g, &mut hist, &syms, &mut rng, |x| lognorm.pdf(x)
        ) {
            // println!("iter {}, norm {}", i, dnorm);
            if i >= 140_000 {
                println!("iter {}, norm {}", i, dnorm);
                break
            }
            // if dnorm < lognorn_stop {
            //     break
            // }
            // if dnorm < 0.79861 { // uniform test
            //     break
            // }
        }
    }
    println!(
        "rotations alg time: {}, norm {}", 
        now.elapsed().as_secs_f64(), mis_opt::diff_norm(&mut hist, |x| lognorm.pdf(x))
    );

    let mut file = File::create("hist.txt").unwrap();
    for (angle, height) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), height.to_radians()).unwrap();
    }
    // for angle in (0..30).map(|i| (i as f64 + 0.5) * (hist_end - hist_beg) / 30.0) {
    //     writeln!(&mut file, "{}\t{}", angle.to_degrees(), lognorm.pdf(angle).to_radians()).unwrap();
    // }

    write_orientations_mtex_euler(&g, "orientations-euler.out");
}

fn grids_diff_norm(g1: &fnd::FundGrid, g2: &fnd::FundGrid) -> f64 {
    g1.cells.iter().flatten().flatten()
        .zip(g2.cells.iter().flatten().flatten())
        .map(|(x, y)| (x - y))
        .sum()
        // .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
}

fn main() {
    let bnds = parse_bnds("bnds-10k.stface");
    let num_vols = count_volumes_from_bnds(&bnds);
    let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let segms = 21;
    let mut grid = fnd::FundGrid::new(segms);
    grid.normalize_grain_volumes(&mut g);
    grid.add_from_iter(g.node_weights());

    let minmax = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap(), 
        *g.cells.iter().flatten().flatten().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
    );

    println!("dvol: {}", grid.dvol);
    println!("min max f: {:?}", minmax(&grid));
    
    let now = Instant::now();
    let mut rotator = ori_opt::Rotator::new(&grid);
    println!("starting texture index: {}", rotator.texture_index(&grid));
    for i in 0..10_000_000 {
        if let OptResult::MoreOptimal(texidx) = rotator.rotate_random_grain_ori(
            &mut g, &mut grid, &mut rng
        ) {
            // println!("iter {}, texture index {}", i, texidx);
        } else {
            rotator.restore_previous(&mut g, &mut grid);
        }
    }
    println!(
        "rotations alg time: {} s, texture index {}", 
        now.elapsed().as_secs_f64(), rotator.texture_index(&grid)
    );

    println!("min max f: {:?}", minmax(&grid));

    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_euler(g.node_count(), "orientations-euler.out", &mut rng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{AbsDiffEq, RelativeEq};

    fn test_rotate_to_fund_domain(o: UnitQuat, syms: &Vec<UnitQuat>) -> UnitQuat {
        let mut res = None;
        for s in syms {
            let q = s * o;
            let angs = EulerAngles::from(q);
            if fnd::euler_angles_inside(angs) {
                if res.is_none() {
                    res = Some(q);
                } else {
                    panic!(
                        "multiple orientations in fundamental domain: {:?} and {:?}", 
                        EulerAngles::from(res.unwrap()), angs
                    )
                }
            }
        }
        if let Some(q) = res {
            q
        } else {
            panic!(
                "failed to rotate to fundamental domain: {:?}, {:?}", 
                o, EulerAngles::from(o)
            )
        }
    }

    #[test]
    fn test_fund_domain() {    
        let mut rng = Pcg64::seed_from_u64(0);
        let syms = cube_rotational_symmetry();
        for &basis in &syms {
            println!("{:?}", EulerAngles::from(basis));
        }
    
        for _ in 0..1_000_000 {
            test_rotate_to_fund_domain(
                fnd::random_euler_angles(&mut rng).into(), 
                &syms
            );
        }
    }

    impl AbsDiffEq for EulerAngles {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            f64::abs_diff_eq(&self.alpha, &other.alpha, epsilon) &&
            f64::abs_diff_eq(&self.cos_beta, &other.cos_beta, epsilon) &&
            f64::abs_diff_eq(&self.gamma, &other.gamma, epsilon)
        }
    }

    impl RelativeEq for EulerAngles {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self, other: &Self, 
            epsilon: Self::Epsilon, max_relative: Self::Epsilon
        ) -> bool {
            f64::relative_eq(&self.alpha, &other.alpha, epsilon, max_relative) &&
            f64::relative_eq(&self.cos_beta, &other.cos_beta, epsilon, max_relative) &&
            f64::relative_eq(&self.gamma, &other.gamma, epsilon, max_relative)
        }
    }

    #[test]
    fn test_quaternion_and_euler_angles_conversion() {
        use approx::assert_relative_eq;

        let mut rng = Pcg64::seed_from_u64(0);
        for _ in 0..1_000_000 {
            let angs = EulerAngles::random(&mut rng);
            let q = UnitQuat::from(angs);
            let back = EulerAngles::from(q);
            assert_relative_eq!(angs, back, max_relative = f32::EPSILON as f64);
        }
        for _ in 0..1_000 {
            let mut angs = EulerAngles::random(&mut rng);
            angs.cos_beta = 1.0;
            let mut correct_back_alpha = angs.alpha + angs.gamma;
            if correct_back_alpha >= 2.0 * PI {
                correct_back_alpha -= 2.0 * PI;
            }
            let q = UnitQuat::from(angs);
            let back = EulerAngles::from(q);
            let correct_back = EulerAngles{ alpha: correct_back_alpha, cos_beta: 1.0, gamma: 0.0 };
            assert_relative_eq!(correct_back, back, max_relative = f32::EPSILON as f64);
        }
    }

    impl AbsDiffEq for FundAngles {
        type Epsilon = <f64 as AbsDiffEq>::Epsilon;

        fn default_epsilon() -> Self::Epsilon {
            f64::default_epsilon()
        }

        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            f64::abs_diff_eq(&self.delta, &other.delta, epsilon) &&
            f64::abs_diff_eq(&self.lambda, &other.lambda, epsilon) &&
            f64::abs_diff_eq(&self.omega, &other.omega, epsilon)
        }
    }

    impl RelativeEq for FundAngles {
        fn default_max_relative() -> Self::Epsilon {
            f64::default_max_relative()
        }

        fn relative_eq(
            &self, other: &Self, 
            epsilon: Self::Epsilon, max_relative: Self::Epsilon
        ) -> bool {
            f64::relative_eq(&self.delta, &other.delta, epsilon, max_relative) &&
            f64::relative_eq(&self.lambda, &other.lambda, epsilon, max_relative) &&
            f64::relative_eq(&self.omega, &other.omega, epsilon, max_relative)
        }
    }

    #[test]
    fn test_fund_angles_and_euler_angles_conversion() {
        use approx::assert_relative_eq;

        let mut rng = Pcg64::seed_from_u64(0);
        for _ in 0..1_000_000 {
            let angs = FundAngles::random(&mut rng);
            let eul = EulerAngles::from(angs);
            let back = FundAngles::from(eul);
            assert_relative_eq!(angs, back, max_relative = f32::EPSILON as f64 * 10.0);
        }
    }
}
