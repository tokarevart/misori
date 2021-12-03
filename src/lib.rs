use nalgebra as na;
use na::{Quaternion, UnitQuaternion, Vector3};
use rand::distributions::Uniform as RandUniform;
use std::f64::consts::*;
use std::fs::File;
use std::io::Write;

pub use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
pub use petgraph::visit::EdgeRef;
pub use rand::prelude::*;
pub use fnd::FundAngles;

pub type UnitQuat = UnitQuaternion<f64>;

pub mod fnd;
pub mod mis_opt;
pub mod ori_opt;
#[cfg(test)]
mod tests;

pub fn random_grain(g: &PolyGraph, rng: &mut impl Rng) -> NodeIndex {
    let distr = RandUniform::new(0, g.node_count() as u32);
    rng.sample(distr).into()
}

pub fn random_grains2(g: &PolyGraph, rng: &mut impl Rng) -> (NodeIndex, NodeIndex) {
    let distr = RandUniform::new(0, g.node_count() as u32);
    let grain1_idx: NodeIndex = rng.sample(distr).into();
    let grain2_idx: NodeIndex = loop {
        let n: NodeIndex = rng.sample(distr).into();
        if n != grain1_idx {
            break n;
        }
    };
    (grain1_idx, grain2_idx)
}

pub enum RotationMode {
    Start,
    Continue{
        prev_ori: GrainOrientation
    }
}

pub enum RotationOptResult {
    MoreOptimal{
        criterion: f64,
        prev_ori: Option<GrainOrientation>,
    },
    SameOrLessOptimal{
        criterion: f64,
        prev_ori: Option<GrainOrientation>,
    },
}

pub enum SwapOptResult {
    MoreOptimal(f64),
    SameOrLessOptimal(f64),
}

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

pub type PolyGraph = UnGraph<Grain, AngleArea>;

#[derive(Clone, Copy, Debug)]
pub struct AngleArea {
    pub angle: f64,
    pub area: f64,
}

pub fn build_graph(bnds: Vec<(f64, u32, u32)>, volumes: Vec<f64>) -> PolyGraph {    
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

pub fn parse_bnds(path: &str) -> Vec<(f64, u32, u32)> {
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

pub fn parse_volumes(path: &str) -> Vec<f64> {
    std::fs::read_to_string(path).unwrap()
        .lines()
        .map(|x| x.trim().parse::<f64>().unwrap())
        .collect()
}

pub fn count_volumes_from_bnds(bnds: &[(f64, u32, u32)]) -> usize {
    bnds.iter().map(|x| x.1.max(x.2) as usize).max().unwrap() + 1
}

pub fn parse_graph(bnds_path: &str, volumes_path: &str) -> PolyGraph {
    let bnds = parse_bnds(bnds_path);
    let volumes = parse_volumes(volumes_path);
    build_graph(bnds, volumes)
}

pub fn write_orientations_quat(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    let total_vol: f64 = g.node_weights().map(|w| w.volume).sum();
    let inv_avg_vol = g.node_count() as f64 / total_vol;
    for w in g.node_weights() {
        let q = &w.orientation.quat;
        writeln!(&mut file, "{} {} {} {} {}", 
            q.w, q.i, q.j, q.k, w.volume * inv_avg_vol
        ).unwrap();
    }
}

pub fn write_orientations_mtex_euler(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    let total_vol: f64 = g.node_weights().map(|w| w.volume).sum();
    let inv_avg_vol = g.node_count() as f64 / total_vol;
    for w in g.node_weights() {
        // inversed quaternion is used because 
        // MTEX defines orientations in a slightly different way 
        // than they have been defined by Bunge.
        // see more in the MTEX article 'MTEX vs. Bunge Convention'
        let angs = EulerAngles::from(w.orientation.quat.inverse());
        writeln!(&mut file, "{} {} {} {}", 
            angs.alpha, angs.cos_beta.acos(), angs.gamma, w.volume * inv_avg_vol
        ).unwrap();
    }
}

pub fn write_random_orientations_mtex_euler(
    g: &PolyGraph, path: &str, rng: &mut impl Rng
) {
    let mut file = File::create(path).unwrap();
    let total_vol: f64 = g.node_weights().map(|w| w.volume).sum();
    let inv_avg_vol = g.node_count() as f64 / total_vol;
    for w in g.node_weights() {
        let q = UnitQuat::from(EulerAngles::random(rng));
        let angs = EulerAngles::from(q.inverse());
        writeln!(&mut file, "{} {} {} {}", 
            angs.alpha, angs.cos_beta.acos(), angs.gamma, w.volume * inv_avg_vol
        ).unwrap();
    }
}

pub fn set_random_orientations(g: &mut PolyGraph, rng: &mut impl Rng) {
    for w in g.node_weights_mut() {
        w.orientation = GrainOrientation::random(rng);
    }
}

pub fn cube_rotational_symmetry() -> Vec<UnitQuat> {
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
