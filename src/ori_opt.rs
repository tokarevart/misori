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
        .sum::<f64>() * grid.dvol
}

fn squared_error_at(d: f64, cellidx: usize, mesh: &fnd::FundMesh) -> f64 {
    let d = d / mesh.cells[cellidx].vol;
    let e = 1.0 - d;
    e * e
}

fn squared_averror_at(d: f64, avvol: f64, mesh: &fnd::FundMesh) -> f64 {
    let d = d / avvol;
    let e = 1.0 - d;
    e * e
}

pub fn squared_error(mesh: &fnd::FundMesh) -> f64 {
    mesh.cells.iter().enumerate()
        .map(|(i, c)| squared_error_at(c.dens, i, mesh))
        .sum::<f64>()
}

pub fn squared_averror(mesh: &fnd::FundMesh) -> f64 {
    let avvol = mesh.total_volume() / mesh.cells.len() as f64;
    mesh.cells.iter().enumerate()
        .map(|(i, c)| squared_averror_at(c.dens, avvol, mesh))
        .sum::<f64>()
}

pub fn mean_squared_error(mesh: &fnd::FundMesh) -> f64 {
    mesh.cells.iter().enumerate()
        .map(|(i, c)| squared_error_at(c.dens, i, mesh) * c.vol)
        .sum::<f64>()
}

pub fn mean_squared_averror(mesh: &fnd::FundMesh) -> f64 {
    let avvol = mesh.total_volume() / mesh.cells.len() as f64;
    mesh.cells.iter().enumerate()
        .map(|(i, c)| squared_averror_at(c.dens, avvol, mesh) * c.vol)
        .sum::<f64>()
}

fn mean_squared_error_at(d: f64, cellidx: usize, mesh: &fnd::FundMesh) -> f64 {
    squared_error_at(d, cellidx, mesh) * mesh.cells[cellidx].vol
}

fn mean_squared_averror_at(d: f64, avvol: f64, mesh: &fnd::FundMesh) -> f64 {
    squared_averror_at(d, avvol, mesh) * avvol
}

#[derive(Debug, Clone, Copy)]
struct CellBackup {
    idxs: fnd::CellIdxs,
    height: f64,
}

#[derive(Debug, Clone)]
struct RotatorBackup {
    mse: f64,
    grain_idx: NodeIndex, 
    ori: Option<GrainOrientation>,
    prev_cell_bu: CellBackup,
    cur_cell_bu: CellBackup,
}

#[derive(Debug, Clone)]
pub struct Rotator {
    backup: Option<RotatorBackup>,
    pub mse: f64,
}

impl Rotator {
    pub fn new(mesh: &fnd::FundMesh) -> Self {
        Self{ backup: None, mse: mean_squared_averror(mesh) }
    }

    pub fn rotate(
        &mut self, mode: RotationMode, grain_idx: NodeIndex, g: &mut PolyGraph, 
        mesh: &mut fnd::FundMesh, rng: &mut impl Rng,
    ) -> RotationOptResult {
        
        let vol = g[grain_idx].volume;
        let prev_mse = self.mse;

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

        self.mse -= 
            mean_squared_error_at(prev_h1, grid) 
            + mean_squared_error_at(prev_h2, grid);
        self.mse += 
            mean_squared_error_at(grid.at(prev_idxs), grid) 
            + mean_squared_error_at(grid.at(cur_idxs), grid);
        
        self.backup = Some(RotatorBackup{ 
            grain_idx, 
            mse: prev_mse,
            ori: prev_ori, 
            prev_cell_bu, 
            cur_cell_bu,
        });

        use RotationOptResult::*;
        if self.mse < prev_mse {
            MoreOptimal{ criterion: self.mse, prev_ori }
        } else {
            SameOrLessOptimal{ criterion: self.mse, prev_ori }
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
            mse,
            ..
        } = self.backup.take().unwrap();

        // restoration order matters in case 
        // the new orientation is in the same cell as the previous one
        *grid.at_mut(cur_cell_bu.idxs) = cur_cell_bu.height;
        *grid.at_mut(prev_cell_bu.idxs) = prev_cell_bu.height;
        self.mse = mse;
    }
}
