use rand_pcg::Pcg64;
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use misori::*;

fn main1() {
    // let bnds = parse_bnds("bnds-10k.stface");
    // let num_vols = count_volumes_from_bnds(&bnds);
    // let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut g = misori::parse_graph("bnds-10k.stface", "vols-10k.stpoly");

    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let syms = cube_rotational_symmetry();
    mis_opt::update_angles(&mut g, &syms);

    let (hist_beg, hist_end) = (0.0, 70.0f64.to_radians());
    let mut hist = mis_opt::Histogram::new(hist_beg, hist_end, 30);
    hist.normalize_grain_boundary_area(&mut g);
    println!(
        "grains boundary area mul by bar len {}", 
        g.edge_weights().map(|e| e.area).sum::<f64>() * hist.bar_len
    );
    let aa = mis_opt::angle_area_vec(&g);
    hist.add_from_slice(&aa);

    let uni = StatUniform::new(hist_beg, hist_end).unwrap();
    let lognorm = StatLogNormal::new(-1.0, 0.5).unwrap();
    
    // let now = Instant::now();
    // for i in 0..1_000_000 {
    //     if let Some(dnorm) = iterate_swaps(
    //         &mut g, &mut hist, &syms, &mut rng, |x| lognorm.pdf(x)
    //     ) {
    //         // println!("iter {}, norm {}", i, dnorm);
    //     }
    // }
    // println!(
    //     "swaps alg time:        {}, norm {}", 
    //     now.elapsed().as_secs_f64(), diff_norm(&mut hist, |x| lognorm.pdf(x))
    // );

    let now = Instant::now();
    let mut rotator = mis_opt::Rotator::new();
    for i in 0..1_000_000 {
        if let OptResult::MoreOptimal{ criterion: dnorm, .. } = rotator.rotate(
            RotationMode::Start,
            misori::random_grain(&g, &mut rng),
            &mut g, &mut hist, &syms, |x| lognorm.pdf(x), &mut rng
        ) {
            // println!("iter {}, norm {}", i, dnorm);
        } else {
            rotator.undo(&mut g, &mut hist);
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

    write_orientations_mtex_euler(&g, "orientations-euler.out");
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
    misori::normalize_grain_volumes(&mut g);
    grid.add_from_iter(g.node_weights());

    let minmax_density = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten()
            .min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol, 
        *g.cells.iter().flatten().flatten()
            .max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol
    );

    println!("dvol: {}", grid.dvol);
    println!("min max f: {:?}", minmax_density(&grid));
    
    let now = Instant::now();
    let mut rotator = ori_opt::Rotator::new(&grid);
    println!("starting texture index: {}", rotator.texture_index);
    for i in 0..10_000_000 {
        if let OptResult::MoreOptimal{ criterion: texidx, .. } = rotator.rotate(
            RotationMode::Start,
            misori::random_grain(&g, &mut rng),
            &mut g, &mut grid, &mut rng
        ) {
            // println!("iter {}, texture index {}", i, texidx);
        } else {
            rotator.undo(&mut g, &mut grid);
        }
    }
    println!(
        "rotations alg time: {} s, texture index {}", 
        now.elapsed().as_secs_f64(), rotator.texture_index
    );

    println!("min max f: {:?}", minmax_density(&grid));

    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_mtex_euler(g.node_count(), "orientations-euler.out", &mut rng);
}

fn main3() {
    let bnds = parse_bnds("bnds-10k.stface");
    let num_vols = count_volumes_from_bnds(&bnds);
    let mut g = build_graph(bnds, vec![1.0; num_vols]);
    // let mut g = parse_graph("bnds-10k.stface", "vols-10k.stpoly");
    println!("nodes {}, edges {}", g.node_count(), g.edge_count());
    let mut rng = Pcg64::seed_from_u64(0);
    set_random_orientations(&mut g, &mut rng);

    let segms1 = 19;
    let segms2 = 21;
    let mut grid1 = fnd::FundGrid::new(segms1);
    let mut grid2 = fnd::FundGrid::new(segms2);
    misori::normalize_grain_volumes(&mut g);
    grid1.add_from_iter(g.node_weights());
    grid2.add_from_iter(g.node_weights());

    let minmax_density = |g: &fnd::FundGrid| (
        *g.cells.iter().flatten().flatten().
            min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol, 
        *g.cells.iter().flatten().flatten().
            max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() / g.dvol,
    );

    println!("min max density1: {:?}", minmax_density(&grid1));
    println!("min max density2: {:?}", minmax_density(&grid2));
    
    let now = Instant::now();
    let mut rotator1 = ori_opt::Rotator::new(&grid1);
    let mut rotator2 = ori_opt::Rotator::new(&grid2);
    println!("starting texture index1: {}", rotator1.texture_index);
    println!("starting texture index2: {}", rotator2.texture_index);
    for i in 0..10_000_000 {
        let grain_idx = misori::random_grain(&g, &mut rng);
        let prev_ori = if let OptResult::MoreOptimal{ 
            criterion: texidx, prev_ori 
        } = rotator1.rotate(RotationMode::Start,grain_idx, &mut g, &mut grid1, &mut rng) {
            // println!("iter {}, texture index {}", i, texidx);
            prev_ori.unwrap()
        } else {
            rotator1.undo(&mut g, &mut grid1);
            continue;
        };
        if let OptResult::MoreOptimal{ criterion: texidx, .. } = rotator2.rotate(
            RotationMode::Continue{ prev_ori },
            grain_idx, &mut g, &mut grid2, &mut rng
        ) {
            // println!("iter {}, texture index {}", i, texidx);
        } else {
            rotator1.undo(&mut g, &mut grid1);
            rotator2.undo(&mut g, &mut grid2);
            continue;
        }

        println!(
            "iter {}, texture index1 {}, texture index2 {}", 
            i, rotator1.texture_index, rotator2.texture_index
        );
    }
    println!(
        "rotations alg time: {} s, texture index1 {}, texture index2 {}", 
        now.elapsed().as_secs_f64(), rotator1.texture_index, rotator2.texture_index
    );

    println!("min max density1: {:?}", minmax_density(&grid1));
    println!("min max density2: {:?}", minmax_density(&grid2));

    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_mtex_euler(g.node_count(), "orientations-euler.out", &mut rng);
}
