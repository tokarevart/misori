use rand_pcg::Pcg64;
use statrs::distribution::Continuous;
use statrs::distribution::Uniform as StatUniform;
use statrs::distribution::LogNormal as StatLogNormal;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use misori::*;

fn main() {
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
        if let mis_opt::OptResult::MoreOptimal(dnorm) = rotator.rotate_random_grain_ori(
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

fn main2() {
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
        if let ori_opt::OptResult::MoreOptimal(texidx) = rotator.rotate_random_grain_ori(
            &mut g, &mut grid, &mut rng
        ) {
            // println!("iter {}, texture index {}", i, texidx);
        } else {
            rotator.undo(&mut g, &mut grid);
        }
    }
    println!(
        "rotations alg time: {} s, texture index {}", 
        now.elapsed().as_secs_f64(), rotator.texture_index(&grid)
    );

    println!("min max f: {:?}", minmax(&grid));

    write_orientations_mtex_euler(&g, "orientations-euler.out");
    // write_random_orientations_mtex_euler(g.node_count(), "orientations-euler.out", &mut rng);
}
