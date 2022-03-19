use nalgebra as na;
use na::{Quaternion, UnitQuaternion, UnitVector3, Vector3};
use rand::distributions::Uniform as RandUniform;
use std::cell::RefCell;
use std::f64::consts::*;
use std::fs::File;
use std::io::Write;
use std::ops::{Deref, DerefMut};

pub use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
pub use petgraph::visit::EdgeRef;
pub use rand::prelude::*;
use rand_pcg::Pcg64;
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

fn writeln_fund_angles_mtex_euler(angs: FundAngles, vol: f64, file: &mut File) {
    // inversed quaternion is used because 
    // MTEX defines orientations in a slightly different way 
    // than they have been defined by Bunge.
    // see more in the MTEX article 'MTEX vs. Bunge Convention'
    let angs = EulerAngles::from(UnitQuat::from(angs).inverse());
    writeln!(file, "{} {} {} {}", 
        angs.alpha, angs.cos_beta.acos(), angs.gamma, vol
    ).unwrap();
}

fn writeln_euler_angles_mtex_euler(angs: EulerAngles, vol: f64, file: &mut File) {
    // inversed quaternion is used because 
    // MTEX defines orientations in a slightly different way 
    // than they have been defined by Bunge.
    // see more in the MTEX article 'MTEX vs. Bunge Convention'
    let angs = EulerAngles::from(UnitQuat::from(angs).inverse());
    writeln!(file, "{} {} {} {}", 
        angs.alpha, angs.cos_beta.acos(), angs.gamma, vol
    ).unwrap();
}

fn writeln_quat_angles_mtex_euler(q: UnitQuat, vol: f64, file: &mut File) {
    // inversed quaternion is used because 
    // MTEX defines orientations in a slightly different way 
    // than they have been defined by Bunge.
    // see more in the MTEX article 'MTEX vs. Bunge Convention'
    let angs = EulerAngles::from(q.inverse());
    writeln!(file, "{} {} {} {}", 
        angs.alpha, angs.cos_beta.acos(), angs.gamma, vol
    ).unwrap();
}

pub fn write_orientations_mtex_euler(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    let total_vol: f64 = g.node_weights().map(|w| w.volume).sum();
    let inv_avg_vol = g.node_count() as f64 / total_vol;
    for w in g.node_weights() {
        writeln_quat_angles_mtex_euler(
            w.orientation.quat, 
            w.volume * inv_avg_vol, 
            &mut file
        );
    }
}

pub fn write_random_orientations_in_rodrigues_sphere_mtex_euler(
    num_oris: usize, center: Vector3<f64>, radius: f64, path: &str
) {
    let mut file = File::create(path).unwrap();
    let mut rng = Pcg64::seed_from_u64(0);
    for _ in 0..num_oris {
        let rod = RodriguesVector::random_in_sphere(center, radius, &mut rng);
        let q = UnitQuat::from(rod);
        writeln_euler_angles_mtex_euler(
            EulerAngles::from(q), 
            1.0, 
            &mut file
        );
    }
}

// write as if grains of a certain polycrystal have random orientations
pub fn write_random_orientations_mtex_euler(
    g: &PolyGraph, path: &str, rng: &mut impl Rng
) {
    let mut file = File::create(path).unwrap();
    let total_vol: f64 = g.node_weights().map(|w| w.volume).sum();
    let inv_avg_vol = g.node_count() as f64 / total_vol;
    for w in g.node_weights() {
        writeln_euler_angles_mtex_euler(
            EulerAngles::random(rng), 
            w.volume * inv_avg_vol, 
            &mut file
        );
    }
}

pub fn write_cells_center_orientations_mtex_euler(discr: usize, path: &str) {
    let (discr_d, discr_l, discr_o) = (
        ((discr + discr) as f64 * PI) as usize, 
        ((discr + discr) as f64 * PI) as usize, 
        discr
    );
    let (dang_d, dang_l, dang_o) = (
        1.0 / discr_d as f64, 
        1.0 / discr_l as f64, 
        1.0 / discr_o as f64
    );

    let mut file = File::create(path).unwrap();
    // for v in (0..3)
    //         .map(|_| (0..discr).map(|i| (i as f64 + 0.5) * dang))
    //         .multi_cartesian_product() {
    //     let fangs = FundAngles{ delta: v[0], lambda: v[1], omega: v[2] };
    //     writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
    // }
    let rng = RefCell::new(Pcg64::seed_from_u64(0));
    let threshold = 0.1;
    for delta in (0..discr_d)
        .map(|i| (i as f64 + 0.5) * dang_d)
        // .filter(|&x| x < threshold)
        // .take_while(|&x| x < threshold + dang_d)
    {
        for lambda in (0..discr_l)
            .map(|i| (i as f64 + 0.5) * dang_l)
            // .filter(|&x| x < threshold)
            // .take_while(|&x| x < threshold + dang_l)
        {
            for omega in (0..discr_o)
                .map(|i| (i as f64 + 0.5) * dang_o) 
                // .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_o) 
                // .filter(|&x| x < threshold)
                // .take_while(|&x| x < threshold + dang_o)
            {
                let fangs = FundAngles{ delta, lambda, omega };
                writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
                // let ea = EulerAngles{ alpha: delta * 2.0 * PI, cos_beta: lambda * 2.0 - 1.0, gamma: omega * 2.0 * PI };
                // writeln_euler_angles_mtex_euler(ea, 1.0, &mut file);
            }
        }
    }
}

pub fn write_cells_random_orientations_mtex_euler(discr: usize, path: &str) {
    let (discr_d, discr_l, discr_o) = (
        ((discr + discr) as f64 * PI).round() as usize, 
        ((discr + discr) as f64 * PI).round() as usize, 
        discr
    );
    dbg!(discr_d, discr_l, discr_o);
    let (dang_d, dang_l, dang_o) = (
        1.0 / discr_d as f64, 
        1.0 / discr_l as f64, 
        1.0 / discr_o as f64
    );

    let mut file = File::create(path).unwrap();
    let rng = RefCell::new(Pcg64::seed_from_u64(2));
    // for v in (0..3)
    //         .map(|_| (0..discr).map(
    //             |i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang
    //         ))
    //         .multi_cartesian_product() {
    //     let fangs = FundAngles{ delta: v[0], lambda: v[1], omega: v[2] };
    //     writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
    // }
    let threshold = 0.02;
    for delta in (0..discr_d)
        .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_d)
        // .filter(|&x| x < threshold)
        // .take_while(|&x| x < threshold + dang_d)
    {
        for lambda in (0..discr_l)
            .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_l)
            // .filter(|&x| x < threshold)
            // .take_while(|&x| x < threshold + dang_l)
        {
            for omega in (0..discr_o)
                .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_o) 
                // .filter(|&x| x < threshold)
                // .take_while(|&x| x < threshold + dang_o)
            {
                let fangs = FundAngles{ delta, lambda, omega };
                writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
                // let ea = EulerAngles{ alpha: delta * 2.0 * PI, cos_beta: lambda * 2.0 - 1.0, gamma: omega * 2.0 * PI };
                // writeln_euler_angles_mtex_euler(ea, 1.0, &mut file);
            }
        }
    }
}

pub fn write_cell_random_orientations_mtex_euler(discr: usize, path: &str) {
    let (discr_d, discr_l, discr_o) = (
        ((discr + discr) as f64 * PI).round() as usize, 
        ((discr + discr) as f64 * PI).round() as usize, 
        discr
    );
    dbg!(discr_d, discr_l, discr_o);
    let (dang_d, dang_l, dang_o) = (
        1.0 / discr_d as f64, 
        1.0 / discr_l as f64, 
        1.0 / discr_o as f64
    );

    let mut file = File::create(path).unwrap();
    let rng = RefCell::new(Pcg64::seed_from_u64(2));
    // for v in (0..3)
    //         .map(|_| (0..discr).map(
    //             |i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang
    //         ))
    //         .multi_cartesian_product() {
    //     let fangs = FundAngles{ delta: v[0], lambda: v[1], omega: v[2] };
    //     writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
    // }
    let threshold = 0.02;
    let cell_d = 0;
    let cell_l = 0;
    let cell_o = 0;
    for delta in std::iter::repeat(cell_d)
        .take(20)
        .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_d)
        // .filter(|&x| x < threshold)
        // .take_while(|&x| x < threshold + dang_d)
    {
        for lambda in std::iter::repeat(cell_l)
            .take(20)
            .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_l)
            // .filter(|&x| x < threshold)
            // .take_while(|&x| x < threshold + dang_l)
        {
            for omega in std::iter::repeat(cell_o)
                .take(20)
                .map(|i| (i as f64 + 0.5 + rng.borrow_mut().gen_range(-0.5..= 0.5)) * dang_o) 
                // .filter(|&x| x < threshold)
                // .take_while(|&x| x < threshold + dang_o)
            {
                let fangs = FundAngles{ delta, lambda, omega };
                writeln_fund_angles_mtex_euler(fangs, 1.0, &mut file);
                // let ea = EulerAngles{ alpha: delta * 2.0 * PI, cos_beta: lambda * 2.0 - 1.0, gamma: omega * 2.0 * PI };
                // writeln_euler_angles_mtex_euler(ea, 1.0, &mut file);
            }
        }
    }
}

pub fn set_random_orientations(g: &mut PolyGraph, rng: &mut impl Rng) {
    for w in g.node_weights_mut() {
        w.orientation = GrainOrientation::random(rng);
    }
}

// pub fn set_random_orientations(g: &mut PolyGraph, rng: &mut impl Rng) {
//     let mut tmp = GrainOrientation::random(rng);
//     while tmp.fund.lambda <= f32::EPSILON as f64 {
//         tmp = GrainOrientation::random(rng);
//     }
//     let mut i = 0;
//     for w in g.node_weights_mut() {
//         if i % 1000 == 0 {
//             tmp = GrainOrientation::random(rng);
//             while tmp.fund.lambda <= f32::EPSILON as f64 {
//                 tmp = GrainOrientation::random(rng);
//             }
//         }
//         i += 1;
//         w.orientation = tmp;
//     }
// }

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RodriguesVector(Vector3<f64>);

impl RodriguesVector {
    pub fn axis(&self) -> UnitVector3<f64> {
        UnitVector3::new_normalize(self.0)
    }

    pub fn angle(&self) -> f64 {
        self.norm().atan() * 2.0
    }

    pub fn from_axis_angle(axis: UnitVector3<f64>, angle: f64) -> Self {
        Self(axis.into_inner() * (angle * 0.5).tan())
    }

    pub fn random_in_sphere(center: Vector3<f64>, radius: f64, rng: &mut impl Rng) -> Self {
        let mut rod = Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0));
        while rod.norm_squared() >= 1.0 {
            rod = Vector3::from_fn(|_, _| rng.gen_range(-1.0..1.0));
        }
        Self(rod * radius + center)
    }
}

impl Deref for RodriguesVector {
    type Target = Vector3<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RodriguesVector {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<UnitQuat> for RodriguesVector {
    fn from(o: UnitQuat) -> Self {
        Self(o.imag() / o.scalar())
    }
}

impl From<RodriguesVector> for UnitQuat {
    fn from(r: RodriguesVector) -> Self {
        let sqtan = r.norm_squared();
        let sqcos = 1.0 / (1.0 + sqtan);
        let sqsin = 1.0 - sqcos;

        let q = Quaternion::from_parts(
            sqcos.sqrt(), r.normalize() * sqsin.sqrt()
        );
        UnitQuat::new_unchecked(q)
    }
}
