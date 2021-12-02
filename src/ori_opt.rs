use crate::*;

pub fn texture_index(grid: &fnd::FundGrid) -> f64 {
    grid.cells.iter().flatten().flatten()
        .map(|&x| x * x)
        .sum::<f64>() / grid.dvol
}

#[derive(Debug, Clone, Copy)]
struct CellBackup {
    idxs: fnd::CellIdxs,
    height: f64,
}

#[derive(Debug, Clone)]
struct RotatorBackup {
    texture_index: f64,
    grain_idx: NodeIndex, 
    ori: Option<GrainOrientation>,
    prev_cell_bu: CellBackup,
    cur_cell_bu: CellBackup,
}

#[derive(Debug, Clone)]
pub struct Rotator {
    backup: Option<RotatorBackup>,
    pub texture_index: f64,
}

impl Rotator {
    pub fn new(grid: &fnd::FundGrid) -> Self {
        Self{ backup: None, texture_index: texture_index(grid) }
    }

    pub fn rotate(
        &mut self, mode: RotationMode, grain_idx: NodeIndex, g: &mut PolyGraph, 
        grid: &mut fnd::FundGrid, rng: &mut impl Rng,
    ) -> OptResult {
        
        let vol = g[grain_idx].volume;
        let prev_texidx = self.texture_index;

        // let prev_ori = g[grain_idx].orientation;
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
        while cur_idxs == prev_idxs {
            cur_ori = GrainOrientation::random(rng);
            cur_idxs = grid.idxs(cur_ori.fund);
        }
        let prev_h2 = grid.at(cur_idxs);
        *grid.at_mut(cur_idxs) += vol;
        let cur_cell_bu = CellBackup{ idxs: cur_idxs, height: prev_h2 };

        self.texture_index += 2.0 * vol * ((prev_h2 - prev_h1) + vol) / grid.dvol;
        
        let backup = RotatorBackup{ 
            grain_idx, 
            texture_index: prev_texidx,
            ori: prev_ori, 
            prev_cell_bu, 
            cur_cell_bu,
        };
        self.backup = Some(backup);

        if self.texture_index < prev_texidx {
            OptResult::MoreOptimal{ criterion: self.texture_index, prev_ori }
        } else {
            OptResult::SameOrLessOptimal{ criterion: self.texture_index, prev_ori }
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
            texture_index,
            ..
        } = self.backup.take().unwrap();

        // restoration order matters in case 
        // the new orientation is in the same cell as the previous one
        *grid.at_mut(cur_cell_bu.idxs) = cur_cell_bu.height;
        *grid.at_mut(prev_cell_bu.idxs) = prev_cell_bu.height;
        self.texture_index = texture_index;
    }
}
