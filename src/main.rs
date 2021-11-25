use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rand::prelude::*;
use rand::distributions::Uniform as RandUniform;
use rand_pcg::Pcg64;
use nalgebra as na;
use na::{Matrix3, Quaternion, Rotation3, SimdPartialOrd, UnitQuaternion, Vector3};
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::f64::consts::*;
use std::fs::File;
use std::io::Write;
use std::panic;
use std::time::Instant;

type Orientation = UnitQuaternion<f64>;
type PolyGraph = UnGraph<Orientation, AngleArea>;

#[derive(Clone, Copy, Debug)]
struct AngleArea {
    pub angle: f64,
    pub area: f64,
}

fn parse_graph(path: &str) -> PolyGraph {
    let tuples = std::fs::read_to_string(path).unwrap()
        .lines()
        .map(|x| x.split_whitespace()
                  .collect::<Vec<_>>())
        .filter(|x| x.len() == 3)
        .map(|x| (
            x[0].parse::<f64>().unwrap(), 
            x[1].parse::<u32>().unwrap() - 1, 
            x[2].parse::<u32>().unwrap() - 1,
        ))
        .collect::<Vec<(f64, u32, u32)>>();
    
    let num_nodes = tuples.iter().map(|x| x.1.max(x.2) as usize).max().unwrap();
    let num_edges = tuples.len();
    
    let mut g = PolyGraph::with_capacity(num_nodes, num_edges);
    for _ in 0..num_nodes+1 {
        g.add_node(Orientation::identity());
    }
    for e in tuples {
        g.add_edge(e.1.into(), e.2.into(), AngleArea{ angle: f64::NAN, area: e.0 });
    }
    
    g
}

fn write_orientations(g: &PolyGraph, path: &str) {
    let mut file = File::create(path).unwrap();
    for q in g.node_weights() {
        writeln!(&mut file, "{} {} {} {}", q.w, q.i, q.j, q.k).unwrap();
    }
}

fn set_random_orientations(g: &mut PolyGraph, rng: &mut impl Rng) {
    for w in g.node_weights_mut() {
        *w = EulerAngles::random(rng).into();
    }
}

fn cube_rotational_symmetry() -> Vec<Orientation> {
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
            Orientation::from_basis_unchecked(&[x, y, z])
        })
        .collect()
}

fn misorientation_angle(
    o1: Orientation, o2: Orientation, 
    syms: &Vec<Orientation>
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
    g: &mut PolyGraph, e: EdgeIndex, syms: &Vec<Orientation>
) -> f64 {
    let (n1, n2) = g.edge_endpoints(e).unwrap();
    let (o1, o2) = (g[n1], g[n2]);
    let prev_angle = g[e].angle;
    g[e].angle = misorientation_angle(o1, o2, syms);
    prev_angle
}

fn update_grain_angles(
    g: &mut PolyGraph, n: NodeIndex, syms: &Vec<Orientation>
) -> Vec<f64> {
    let edges: Vec<_> = g.edges(n).map(|e| e.id()).collect();
    let mut prev_angles = Vec::with_capacity(edges.len());
    for e in edges {
        prev_angles.push(update_angle(g, e, syms));
    }
    prev_angles
}

fn update_grain_angles_noret(
    g: &mut PolyGraph, n: NodeIndex, syms: &Vec<Orientation>
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

fn update_angles(g: &mut PolyGraph, syms: &Vec<Orientation>) {
    for e in g.edge_indices() {
        update_angle(g, e, syms);
    }
}

fn angle_area_vec(g: &PolyGraph) -> Vec<AngleArea> {
    g.edge_weights().map(|&e| e).collect()
}

fn max_gap(sorted_pairs: &Vec<AngleArea>) -> f64 {
    let mut max_gap = 0.0;
    let mut prev_angle = sorted_pairs.first().unwrap().angle;
    for &AngleArea{ angle, .. } in sorted_pairs.iter().skip(1) {
        let gap = angle - prev_angle;
        max_gap = gap.max(max_gap);
        prev_angle = angle;
    }

    max_gap
}

#[derive(Clone, Debug)]
struct Histogram {
    pub beg: f64,
    pub end: f64,
    pub heights: Vec<f64>,
    find_bar_coef: f64,
}

impl Histogram {
    pub fn new(beg: f64, end: f64, bars: usize) -> Self {
        let hs = vec![0.0; bars];
        Histogram { 
            beg, end, heights: hs, 
            find_bar_coef: bars as f64 / (end - beg),
        }
    }

    pub fn add(&mut self, aa: AngleArea) {
        let idx = self.bar_idx(aa.angle);
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

    pub fn bar_len(&self) -> f64 {
        (self.end - self.beg) / self.bars() as f64
    }

    pub fn total_height(&self) -> f64 {
        self.heights.iter().sum()
    }

    pub fn area(&self) -> f64 {
        self.total_height() * self.bar_len()
    }

    pub fn bar_idx(&self, angle: f64) -> usize {
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
        let inv_area = 1.0 / (g.edge_weights().map(|x| x.area).sum::<f64>() * self.bar_len());
        for AngleArea{ area, .. } in g.edge_weights_mut() {
            *area *= inv_area;
        }
    }

    fn update_with_edge_new_angle(&mut self, new_aa: AngleArea, prev_angle: f64) {
        let hpos = self.bar_idx(new_aa.angle);
        let prev_hpos = self.bar_idx(prev_angle);
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
        let d = self.bar_len();
        let first = self.beg + d * 0.5;
        self.heights.clone().into_iter()
            .enumerate()
            .map(move |(i, h)| (first + i as f64 * d, h))
    }
}

fn diff_norm(hist: &Histogram, f: impl Fn(f64) -> f64) -> f64 {
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

fn swap_ori(g: &mut PolyGraph, n1: NodeIndex, n2: NodeIndex) {
    let gn1 = g[n1];
    g[n1] = g[n2];
    g[n2] = gn1;
}

fn iterate_swaps(
    g: &mut PolyGraph, hist: &mut Histogram, syms: &Vec<Orientation>,
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

fn iterate_rotations(
    g: &mut PolyGraph, hist: &mut Histogram, syms: &Vec<Orientation>,
    rng: &mut impl Rng, f: impl Fn(f64) -> f64
) -> Option<f64> {

    const MAX_ROTS: usize = 4;

    let distr = RandUniform::new(0, g.node_count() as u32);
    let n: NodeIndex = rng.sample(distr).into();
    
    let prev_ori = g[n];
    g[n] = EulerAngles::random(rng).into();
    let prev_angles = update_grain_angles(g, n, syms);
    let prev_hist = hist.update_with_grain_new_angles(g, n, &prev_angles);

    let prev_dnorm = diff_norm(&prev_hist, |x| f(x));
    let dnorm = diff_norm(hist, |x| f(x));
    if dnorm < prev_dnorm {
        Some(dnorm)
    } else {
        for _ in 0..MAX_ROTS - 1 {
            g[n] = EulerAngles::random(rng).into();
            let prev_angles = update_grain_angles(g, n, syms);
            hist.update_with_grain_new_angles_noret(g, n, &prev_angles);
            let dnorm = diff_norm(hist, |x| f(x));
            if dnorm < prev_dnorm {
                return Some(dnorm);
            }
        }

        *hist = prev_hist;
        g[n] = prev_ori;
        restore_grain_angles(g, n, prev_angles);
        None
    }
}

fn rotate_to_fund_domain(o: Orientation, syms: &Vec<Orientation>) -> Orientation {
    for s in syms {
        let q = s * o;
        let angs = EulerAngles::from(q);
        if fund_domain::euler_angles_inside(angs) {
            return q;
        }
    }
    panic!("failed to rotate to fundamental domain")
}

fn rotate_to_fund_domain_debug(o: Orientation, syms: &Vec<Orientation>) -> Orientation {
    let mut res = None;
    for s in syms {
        let q = s * o;
        let angs = EulerAngles::from(q);
        if fund_domain::euler_angles_inside(angs) {
            if res.is_none() {
                res = Some(q);
            } else {
                println!("q{:?}\nr{:?}", angs, EulerAngles::from(res.unwrap()));
                panic!("multiple orientations in fundamental domain")
            }
        }
    }
    if let Some(q) = res {
        q
    } else {
        println!("{:?}\n{:?}", o, EulerAngles::from(o));
        panic!("failed to rotate to fundamental domain")
    }
}

pub mod fund_domain {
    use crate::EulerAngles;
    use rand::prelude::*;
    use std::f64::consts::*;

    fn delta_to_alpha_rough(delta: f64) -> f64 {
        let coefs = [
            1.0616904724669767, 
            0.5304936828043207, 
            -1.6125716869506046, 
            1.3248915389845322, 
            -0.33738085998446576,
        ];
        let mut alpha = 0.002259301224771941;
        let mut x = delta;
        for c in coefs {
            alpha += c * x;
            x *= delta;
        }
        alpha.clamp(0.0, FRAC_PI_2 - f64::EPSILON)
    }

    fn delta_to_alpha(delta: f64) -> f64 {
        // max abs residual is around 1.54e-7 in both cases
        let (mut alpha, coefs) = if delta < FRAC_PI_4 {
            (-1.5412213748824816e-7, [
                1.138093530676275, 
                -0.0006002734608298991, 
                -0.290073973800705, 
                -0.03556580842760254, 
                0.27337326904565074,
                -0.18567348837960618,
                0.04264441959610007,
            ])
        } else {
            (0.2939488268782047, [
                0.5896569406998474,
                -0.3094771848329967,
                0.9258403328769952,
                -1.0243587280088193,
                0.7330805129161097,
                -0.28322639540399663,
                0.04264441961596222,
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
        3.0 * if alpha < FRAC_PI_4 { 
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
        0.3333333333333333 * (1.0 - cos_beta) / cos_beta_length(alpha)
    }
    
    fn lambda_to_cos_beta(lambda: f64, alpha: f64) -> f64 {
        1.0 - 3.0 * cos_beta_length(alpha) * lambda
    }

    fn random_delta(rng: &mut impl Rng) -> f64 {
        rng.gen_range(0.0..FRAC_PI_2)
    }

    fn random_alpha(rng: &mut impl Rng) -> f64 {
        let delta = rng.gen_range(0.0..FRAC_PI_2);
        delta_to_alpha(delta)
    }

    fn random_lambda(rng: &mut impl Rng) -> f64 {
        rng.gen_range(0.0..0.3333333333333333)
    }

    fn random_cos_beta(alpha: f64, rng: &mut impl Rng) -> f64 {
        let low = lower_bnd_fund_cos_beta(alpha);
        rng.gen_range(low..1.0)
    }

    fn random_gamma(rng: &mut impl Rng) -> f64 {
        rng.gen_range(0.0..PI*2.0)
    }

    pub fn random_euler_angles(rng: &mut impl Rng) -> EulerAngles {
        let gamma = random_gamma(rng);
        let alpha = random_alpha(rng);
        let cos_beta = random_cos_beta(alpha, rng);
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

    fn delta_idx_with_size(delta: f64, size: usize) -> usize {
        (delta * size as f64 * FRAC_2_PI)
            .clamp(0.5, size as f64 - 0.5) as usize
    }

    fn lambda_idx_with_size(lambda: f64, size: usize) -> usize {
        (lambda * size as f64 * 0.3333333333333333)
            .clamp(0.5, size as f64 - 0.5) as usize
    }
    
    fn gamma_idx_with_size(gamma: f64, size: usize) -> usize {
        (gamma * size as f64 * 0.5 * FRAC_1_PI)
            .clamp(0.5, size as f64 - 0.5) as usize
    }

    #[derive(Debug, Clone, Copy)]
    pub struct FundAngles {
        delta: f64,  // [0;Pi/2)
        lambda: f64, // [0; 1/3)
        gamma: f64,  // [0;2*Pi)
    }

    impl FundAngles {
        pub fn random(rng: &mut impl Rng) -> Self {
            Self {
                delta: random_delta(rng),
                lambda: random_lambda(rng),
                gamma: random_gamma(rng),
            }
        }
    }

    type Vec3<T> = Vec<Vec<Vec<T>>>;

    #[derive(Debug, Clone)]
    pub struct FundGrid {
        cells: Vec3<f64>,
        //...
    }

    impl FundGrid {
        //...
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EulerAngles {
    alpha: f64,    // [0;2*Pi)
    cos_beta: f64, // [0;  Pi)
    gamma: f64,    // [0;2*Pi)
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

impl From<Orientation> for EulerAngles {
    fn from(o: Orientation) -> Self {
        let cos_beta = o.w*o.w - o.i*o.i - o.j*o.j + o.k*o.k;
        if cos_beta.abs() >= 1.0 - f64::EPSILON {
            let om11 = o.w*o.w + o.i*o.i - o.j*o.j - o.k*o.k;
            let hom21 = o.i * o.j + o.w * o.k;
            let a = if hom21 < 0.0 { 2.0 * PI - om11.acos() } else { om11.acos() };
            let alpha = if a >= 2.0 * PI { a - 2.0 * PI } else { a };
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

impl From<EulerAngles> for Orientation {
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
        Orientation::new_unchecked(q)
    }
}

fn main1() {
    let mut g = parse_graph("poly-1k.stface");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();

    let (alpha, cos_beta, gamma) = (FRAC_PI_2, 1.0 - 1e-16, FRAC_PI_3 * 2.0);
    let angs = EulerAngles{ alpha, cos_beta, gamma };
    let q = Orientation::from(angs);
    let back_angs = EulerAngles::from(q);
    dbg!(angs);
    dbg!(back_angs);
    println!("alpha+gamma: {}", alpha + gamma);
}

fn main() {
    let mut g = parse_graph("poly-1k.stface");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();
    for basis in &syms {
        println!("{:?}", EulerAngles::from(*basis));
    }

    let mut dummy = Orientation::identity();
    for _ in 0..1_000_000 {
        dummy *= rotate_to_fund_domain_debug(
            fund_domain::random_euler_angles(&mut rng).into(), 
            &syms
        );
    }
    println!("dummy {:?}", dummy);

    // let (alpha, cos_beta, gamma) = (0.0, 1.0 - 1e-15, FRAC_PI_2);
    // let angs = EulerAngles{ alpha, cos_beta, gamma };
    // let q = Orientation::from(angs);
    // let bangs = EulerAngles::from(q);
    // println!("deb {:?}", bangs);
    // println!("rotated {:?}", EulerAngles::from(rotate_to_fund_domain_debug(angs.into(), &syms)));
}

fn main3() {
    let mut g = parse_graph("poly-10k.stface");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();
    for basis in &syms {
        println!("{:?}", basis);
    }
    println!("syms {}", syms.len());
    update_angles(&mut g, &syms);

    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = Histogram::new(hist_beg, hist_end, 30);
    hist.normalize_grain_boundary_area(&mut g);
    println!(
        "grains boundary area mul by bar len {}", 
        g.edge_weights().map(|e| e.area).sum::<f64>() * hist.bar_len()
    );
    let aa = angle_area_vec(&g);
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
    //         if dnorm < lognorn_stop {
    //             break
    //         }
    //     }
    // }
    // println!(
    //     "swaps alg time:        {}, norm {}", 
    //     now.elapsed().as_secs_f64(), diff_norm(&mut hist, |x| lognorm.pdf(x))
    // );

    let now = Instant::now();
    for i in 0..3_000_000 {
        if let Some(dnorm) = iterate_rotations(
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
        now.elapsed().as_secs_f64(), diff_norm(&mut hist, |x| lognorm.pdf(x))
    );

    let mut file = File::create("hist.txt").unwrap();
    for (angle, height) in hist.pairs() {
        writeln!(&mut file, "{}\t{}", angle.to_degrees(), height.to_radians()).unwrap();
    }
    // for angle in (0..30).map(|i| (i as f64 + 0.5) * (hist_end - hist_beg) / 30.0) {
    //     writeln!(&mut file, "{}\t{}", angle.to_degrees(), lognorm.pdf(angle).to_radians()).unwrap();
    // }

    write_orientations(&g, "orientations.out");
}
