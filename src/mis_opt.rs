use crate::*;

pub fn normalize_grain_boundary_area(g: &mut PolyGraph) {
    let inv_area = 1.0 / g.edge_weights().map(|x| x.area).sum::<f64>();
    for AngleArea{ area, .. } in g.edge_weights_mut() {
        *area *= inv_area;
    }
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
        self.total_height()
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

    // pub fn normalize_grain_boundary_area(&self, g: &mut PolyGraph) {
    //     let inv_area = 1.0 / (g.edge_weights().map(|x| x.area).sum::<f64>() * self.bar_len);
    //     for AngleArea{ area, .. } in g.edge_weights_mut() {
    //         *area *= inv_area;
    //     }
    // }

    fn update_with_edge_new_angle(&mut self, new_aa: AngleArea, prev_angle: f64) {
        let hpos = self.idx(new_aa.angle);
        let prev_hpos = self.idx(prev_angle);
        if hpos != prev_hpos {
            self.heights[prev_hpos] -= new_aa.area;
            self.heights[hpos] += new_aa.area;
        }
    }

    pub fn pairs(&self) -> impl Iterator<Item=(f64, f64)> {
        let bl = self.bar_len;
        let first = self.beg + bl * 0.5;
        self.heights.clone().into_iter()
            .enumerate()
            .map(move |(i, h)| (first + i as f64 * bl, h / bl))
    }
}

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

pub fn update_angles(g: &mut PolyGraph, syms: &Vec<UnitQuat>) {
    for e in g.edge_indices() {
        update_angle(g, e, syms);
    }
}

fn restore_grain_angles(g: &mut PolyGraph, n: NodeIndex, prev_angles: Vec<f64>) {
    let edges: Vec<_> = g.edges(n).map(|e| e.id()).collect();
    for (&e, a) in edges.iter().zip(prev_angles) {
        g[e].angle = a;
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
    (hist.pairs()
        .map(|(a, d)| {
            let fa = f(a);
            ((fa - d) / (fa + d)).powi(2)
        })
        .sum::<f64>() / hist.bars() as f64).sqrt()

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

#[derive(Debug, Clone)]
struct SwapperBackup {
    dnorm: f64,
    grain1_idx: NodeIndex, 
    grain2_idx: NodeIndex,
    angles1: Vec<f64>,
    angles2: Vec<f64>,
    hist: Histogram,
}

#[derive(Debug, Clone)]
pub struct Swapper<ObjF: Fn(f64) -> f64> {
    backup: Option<SwapperBackup>,
    pub obj_func: ObjF,
    pub dnorm: f64,
}

impl<ObjF: Fn(f64) -> f64> Swapper<ObjF> {
    pub fn new(hist: &Histogram, obj_func: ObjF) -> Self {
        let dnorm = diff_norm(hist, |x| obj_func(x));
        Self{ backup: None, obj_func, dnorm }
    }

    fn update_hist_with_2grains_new_angles(
        hist: &mut Histogram, g: &PolyGraph, 
        grain1_idx: NodeIndex, grain2_idx: NodeIndex, 
        prev_angles1: &Vec<f64>, prev_angles2: &Vec<f64>,
    ) -> Histogram {

        let prev_hist = hist.clone();
        for (e, &pa) in g.edges(grain1_idx).zip(prev_angles1) {
            hist.update_with_edge_new_angle(*e.weight(), pa);
        }
        for (e, &pa) in g.edges(grain2_idx).zip(prev_angles2) {
            // in petgraph v0.6.0 source is always grain2_idx even when graph is undirected
            if e.target() == grain1_idx {
                continue;
            }
            // more implementation stable version, doesn't require source to always be grain2_idx
            // if e.source() == grain2_idx && e.target() == grain1_idx ||
            //    e.source() == grain1_idx && e.target() == grain2_idx {
            //     continue;
            // }
            hist.update_with_edge_new_angle(*e.weight(), pa);
        }

        prev_hist
    }

    fn swap_ori(g: &mut PolyGraph, n1: NodeIndex, n2: NodeIndex) {
        let gn1_ori = g[n1].orientation;
        g[n1].orientation = g[n2].orientation;
        g[n2].orientation = gn1_ori;
    }

    pub fn swap(
        &mut self, grains: (NodeIndex, NodeIndex), g: &mut PolyGraph, 
        hist: &mut Histogram, syms: &Vec<UnitQuat>
    ) -> SwapOptResult {
    
        let (grain1_idx, grain2_idx) = grains;        
        Self::swap_ori(g, grain1_idx, grain2_idx);
        let prev_angles1 = update_grain_angles(g, grain1_idx, syms);
        let prev_angles2 = update_grain_angles(g, grain2_idx, syms);
        let prev_hist = Self::update_hist_with_2grains_new_angles(
            hist, g, grain1_idx, grain2_idx, &prev_angles1, &prev_angles2
        );
    
        let prev_dnorm = diff_norm(&prev_hist, |x| (self.obj_func)(x));
        self.backup = Some(SwapperBackup{ 
            dnorm: prev_dnorm,
            grain1_idx, 
            grain2_idx,
            angles1: prev_angles1,
            angles2: prev_angles2,
            hist: prev_hist,
        });

        let dnorm = diff_norm(hist, |x| (self.obj_func)(x));
        if dnorm < prev_dnorm {
            SwapOptResult::MoreOptimal(dnorm)
        } else {
            SwapOptResult::SameOrLessOptimal(dnorm)
        }
    }

    pub fn undo(&mut self, g: &mut PolyGraph, hist: &mut Histogram) {
        let SwapperBackup{ 
            dnorm: prev_dnorm,
            grain1_idx, 
            grain2_idx,
            angles1: prev_angles1,
            angles2: prev_angles2,
            hist: prev_hist,
        } = self.backup.take().unwrap();

        *hist = prev_hist;
        Self::swap_ori(g, grain1_idx, grain2_idx);
        restore_grain_angles(g, grain1_idx, prev_angles1);
        restore_grain_angles(g, grain2_idx, prev_angles2);
        self.dnorm = prev_dnorm;
    }
}

#[derive(Debug, Clone)]
struct RotatorBackup {
    dnorm: f64,
    grain_idx: NodeIndex, 
    ori: Option<GrainOrientation>,
    angles: Vec<f64>,
    hist: Histogram,
}

#[derive(Debug, Clone)]
pub struct Rotator {
    backup: Option<RotatorBackup>,
    pub dnorm: f64,
}

impl Rotator {
    pub fn new() -> Self {
        Self{ backup: None, dnorm: f64::MAX }
    }

    fn update_hist_with_grain_new_angles(
        hist: &mut Histogram, g: &PolyGraph, n: NodeIndex, prev_angles: &Vec<f64>
    ) -> Histogram {

        let prev_hist = hist.clone();
        for (e, &pa) in g.edges(n).zip(prev_angles) {
            hist.update_with_edge_new_angle(*e.weight(), pa);
        }

        prev_hist
    }

    pub fn rotate(
        &mut self, mode: RotationMode, grain_idx: NodeIndex, g: &mut PolyGraph, hist: &mut Histogram, 
        syms: &Vec<UnitQuat>, f: impl Fn(f64) -> f64, rng: &mut impl Rng, 
    ) -> RotationOptResult {
        
        let prev_ori = if let RotationMode::Start = mode {
            let prev_ori = g[grain_idx].orientation;
            g[grain_idx].orientation = GrainOrientation::random(rng);
            Some(prev_ori)
        } else {
            None
        };
        
        let prev_angles = update_grain_angles(g, grain_idx, syms);
        let prev_hist = Self::update_hist_with_grain_new_angles(hist, g, grain_idx, &prev_angles);
        let prev_dnorm = diff_norm(&prev_hist, |x| f(x));
        self.backup = Some(RotatorBackup{
            dnorm: prev_dnorm,
            grain_idx,
            ori: prev_ori,
            angles: prev_angles,
            hist: prev_hist,
        });

        let dnorm = diff_norm(hist, |x| f(x));
        use RotationOptResult::*;
        if dnorm < prev_dnorm {
            MoreOptimal{ criterion: dnorm, prev_ori }
        } else {
            SameOrLessOptimal{ criterion: dnorm, prev_ori }
        }
    }

    pub fn undo(&mut self, g: &mut PolyGraph, hist: &mut Histogram) {
        if let &RotatorBackup{ 
            grain_idx,
            ori: Some(prev_ori),
            ..
        } = self.backup.as_ref().unwrap() {
            g[grain_idx].orientation = prev_ori;
        }

        let RotatorBackup{ 
            dnorm: prev_dnorm,
            grain_idx,
            angles: prev_angles,
            hist: prev_hist,
            ..
        } = self.backup.take().unwrap();

        *hist = prev_hist;
        restore_grain_angles(g, grain_idx, prev_angles);
        self.dnorm = prev_dnorm;
    }
}
