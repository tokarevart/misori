use crate::*;

pub fn normalize_grain_volumes(g: &mut PolyGraph) {
    let inv_vol = 1.0 / g.node_weights().map(|x| x.volume).sum::<f64>();
    for Grain{ volume, .. } in g.node_weights_mut() {
        *volume *= inv_vol;
    }
}

pub fn texture_index(grid: &fnd::FundGrid) -> f64 {
    grid.cells.iter().flatten().flatten()
        .map(|&x| x * x)
        .sum::<f64>() / grid.dvol
}

pub fn diff_norm(grid: &fnd::FundGrid) -> f64 {
    quad_diff_norm(grid).sqrt().sqrt()
}

pub fn quad_diff_norm(grid: &fnd::FundGrid) -> f64 {
    grid.cells.iter().flatten().flatten()
        .map(|&d| {
            let d = d / grid.dvol;
            ((1.0 - d) / (1.0 + d)).powi(4)
        })
        .sum::<f64>() * grid.dvol
}

pub fn quad_diff_norm_at(d: f64, grid: &fnd::FundGrid) -> f64 {
    let d = d / grid.dvol;
    ((1.0 - d) / (1.0 + d)).powi(4) * grid.dvol
}

#[derive(Debug, Clone, Copy)]
struct CellBackup {
    idxs: fnd::CellIdxs,
    height: f64,
}

#[derive(Debug, Clone)]
struct RotatorBackup {
    quad_dnorm: f64,
    grain_idx: NodeIndex, 
    ori: Option<GrainOrientation>,
    prev_cell_bu: CellBackup,
    cur_cell_bu: CellBackup,
}

#[derive(Debug, Clone)]
pub struct Rotator {
    backup: Option<RotatorBackup>,
    pub quad_dnorm: f64,
}

impl Rotator {
    pub fn new(grid: &fnd::FundGrid) -> Self {
        Self{ backup: None, quad_dnorm: quad_diff_norm(grid) }
    }

    pub fn rotate(
        &mut self, mode: RotationMode, grain_idx: NodeIndex, g: &mut PolyGraph, 
        grid: &mut fnd::FundGrid, rng: &mut impl Rng,
    ) -> RotationOptResult {
        
        let vol = g[grain_idx].volume;
        let prev_qdnorm = self.quad_dnorm;

        let (prev_ori, actual_prev_ori) = match mode {
            RotationMode::Start => {
                let prev_ori = g[grain_idx].orientation;
                g[grain_idx].orientation = GrainOrientation::random(rng);
                (Some(prev_ori), prev_ori)
            },
            RotationMode::Continue{ prev_ori } => (None, prev_ori),
        };
    
        let prev_idxs = grid.idxs(actual_prev_ori.fund);
        let prev_h1 = grid.at(prev_idxs);
        *grid.at_mut(prev_idxs) -= vol;
        let prev_cell_bu = CellBackup{ idxs: prev_idxs, height: prev_h1 };

        let mut cur_ori = g[grain_idx].orientation;
        let mut cur_idxs = grid.idxs(cur_ori.fund);
        if let RotationMode::Start = mode {
            while cur_idxs == prev_idxs {
                cur_ori = GrainOrientation::random(rng);
                cur_idxs = grid.idxs(cur_ori.fund);
            }
        }
        let prev_h2 = grid.at(cur_idxs);
        *grid.at_mut(cur_idxs) += vol;
        let cur_cell_bu = CellBackup{ idxs: cur_idxs, height: prev_h2 };

        self.quad_dnorm -= 
            quad_diff_norm_at(prev_h1, grid) 
            + quad_diff_norm_at(prev_h2, grid);
        self.quad_dnorm += 
            quad_diff_norm_at(grid.at(prev_idxs), grid) 
            + quad_diff_norm_at(grid.at(cur_idxs), grid);
        
        self.backup = Some(RotatorBackup{ 
            grain_idx, 
            quad_dnorm: prev_qdnorm,
            ori: prev_ori, 
            prev_cell_bu, 
            cur_cell_bu,
        });

        use RotationOptResult::*;
        if self.quad_dnorm < prev_qdnorm {
            MoreOptimal{ criterion: self.quad_dnorm.sqrt().sqrt(), prev_ori }
        } else {
            SameOrLessOptimal{ criterion: self.quad_dnorm.sqrt().sqrt(), prev_ori }
        }
    }

    pub fn undo(&mut self, g: &mut PolyGraph, grid: &mut fnd::FundGrid) {
        if let &RotatorBackup{ 
            grain_idx, 
            ori: Some(prev_ori),
            ..
        } = self.backup.as_ref().unwrap() {
            g[grain_idx].orientation = prev_ori;
        }

        let RotatorBackup{
            prev_cell_bu, 
            cur_cell_bu, 
            quad_dnorm,
            ..
        } = self.backup.take().unwrap();

        // restoration order matters in case 
        // the new orientation is in the same cell as the previous one
        *grid.at_mut(cur_cell_bu.idxs) = cur_cell_bu.height;
        *grid.at_mut(prev_cell_bu.idxs) = prev_cell_bu.height;
        self.quad_dnorm = quad_dnorm;
    }
}
