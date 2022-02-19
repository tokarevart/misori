use core::ops::Range;

use crate::*;

pub fn normalize_grain_boundary_area(g: &mut PolyGraph) {
    let inv_area = 1.0 / g.edge_weights().map(|x| x.area).sum::<f64>();
    for AngleArea{ area, .. } in g.edge_weights_mut() {
        *area *= inv_area;
    }
}

pub fn cubic_range() -> Range<f64> {
    let end = 2.0 * ((SQRT_2 - 1.0) * (5.0 - 2.0 * SQRT_2).sqrt()).atan();
    0.0..end
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
    pub fn new(range: Range<f64>, bars: usize) -> Self {
        let (beg, end) = (range.start, range.end);
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
        let inv_bl = 1.0 / bl;
        let first = self.beg + bl * 0.5;
        self.heights.clone().into_iter()
            .enumerate()
            .map(move |(i, h)| (first + i as f64 * bl, h * inv_bl))
    }
}

fn misorientation_angle(
    o1: UnitQuat, o2: UnitQuat, 
    syms: &Vec<UnitQuat>
) -> f64 {
    let r = o1.rotation_to(&o2);
    if r.w > 1.0 - f32::EPSILON as f64 {
        0.0
    } else {
        syms.iter()
            .map(|s| (s.scalar() * r.scalar() - s.imag().dot(&r.imag())).abs())
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap()
            .acos() * 2.0
    }
}

fn update_angle(
    g: &mut PolyGraph, e: EdgeIndex, syms: &Vec<UnitQuat>
) -> f64 {

    let (n1, n2) = g.edge_endpoints(e).unwrap();
    let (o1, o2) = (g[n1].orientation.quat, g[n2].orientation.quat);
    let prev_angle = g[e].angle;
    g[e].angle = misorientation_angle(o1, o2, syms);
    if g[e].angle.is_nan() {
        dbg!((n1, n2));
    }
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

fn squared_error_at(a: f64, d: f64, f: impl Fn(f64) -> f64) -> f64 {
    let fa = f(a);
    let e = fa - d;
    e * e
}

pub fn squared_error(hist: &Histogram, f: impl Fn(f64) -> f64) -> f64 {
    hist.pairs()
        .map(|(a, d)| squared_error_at(a, d, |x| f(x)))
        .sum::<f64>()
}

pub fn mean_squared_error(hist: &Histogram, f: impl Fn(f64) -> f64) -> f64 {
    squared_error(hist, f) / hist.bars() as f64
}

type ObjFn<'a> = Box<dyn Fn(&Histogram) -> f64 + 'a>;

pub fn distribution_loss_fn<'a>(distr: &'a impl Fn(f64) -> f64) -> ObjFn<'a> {
    Box::new(|h: &Histogram| mean_squared_error(h, |x| distr(x)))
}

#[derive(Debug, Clone)]
struct SwapperBackup {
    error: f64,
    grain1_idx: NodeIndex, 
    grain2_idx: NodeIndex,
    angles1: Vec<f64>,
    angles2: Vec<f64>,
    hist: Histogram,
}

pub struct Swapper<'a> {
    backup: Option<SwapperBackup>,
    pub loss_fn: ObjFn<'a>,
    pub error: f64,
}

impl<'a> Swapper<'a> {
    pub fn new_with_distr(hist: &Histogram, distr: &'a impl Fn(f64) -> f64) -> Self {
        let loss_fn = distribution_loss_fn(distr);
        Self::new_with_loss_fn(hist, loss_fn)
    }

    pub fn new_with_loss_fn(hist: &Histogram, loss_fn: ObjFn<'a>) -> Self {
        let error = loss_fn(hist);
        Self{ backup: None, loss_fn, error }
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
    
        let prev_error = (self.loss_fn)(&prev_hist);
        self.backup = Some(SwapperBackup{ 
            error: prev_error,
            grain1_idx, 
            grain2_idx,
            angles1: prev_angles1,
            angles2: prev_angles2,
            hist: prev_hist,
        });

        let error = (self.loss_fn)(hist);
        if error < prev_error {
            SwapOptResult::MoreOptimal(error)
        } else {
            SwapOptResult::SameOrLessOptimal(error)
        }
    }

    pub fn undo(&mut self, g: &mut PolyGraph, hist: &mut Histogram) {
        let SwapperBackup{ 
            error: prev_error,
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
        self.error = prev_error;
    }
}

#[derive(Debug, Clone)]
struct RotatorBackup {
    error: f64,
    grain_idx: NodeIndex, 
    ori: Option<GrainOrientation>,
    angles: Vec<f64>,
    hist: Histogram,
}

pub struct Rotator<'a> {
    backup: Option<RotatorBackup>,
    pub loss_fn: ObjFn<'a>,
    pub error: f64,
}

impl<'a> Rotator<'a> {
    pub fn new_with_distr(hist: &Histogram, distr: &'a impl Fn(f64) -> f64) -> Self {
        let loss_fn = distribution_loss_fn(distr);
        Self::new_with_loss_fn(hist, loss_fn)
    }

    pub fn new_with_loss_fn(hist: &Histogram, loss_fn: ObjFn<'a>) -> Self {
        let error = loss_fn(hist);
        Self{ backup: None, loss_fn, error }
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
        &mut self, mode: RotationMode, grain_idx: NodeIndex, g: &mut PolyGraph, 
        hist: &mut Histogram, syms: &Vec<UnitQuat>, rng: &mut impl Rng, 
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
        let prev_error = (self.loss_fn)(&prev_hist);
        self.backup = Some(RotatorBackup{
            error: prev_error,
            grain_idx,
            ori: prev_ori,
            angles: prev_angles,
            hist: prev_hist,
        });

        let error = (self.loss_fn)(hist);
        use RotationOptResult::*;
        if error < prev_error {
            MoreOptimal{ criterion: error, prev_ori }
        } else {
            SameOrLessOptimal{ criterion: error, prev_ori }
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
            error: prev_error,
            grain_idx,
            angles: prev_angles,
            hist: prev_hist,
            ..
        } = self.backup.take().unwrap();

        *hist = prev_hist;
        restore_grain_angles(g, grain_idx, prev_angles);
        self.error = prev_error;
    }
}
